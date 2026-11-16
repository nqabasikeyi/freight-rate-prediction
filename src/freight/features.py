"""Feature engineering. All label-derived encodings are fit on training rows only."""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

US_HOLIDAYS = holidays.US(years=[2024, 2025, 2026])
HOLIDAY_DATES = pd.to_datetime(sorted(US_HOLIDAYS.keys()))
EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def _days_to_nearest_holiday(dates: pd.Series) -> np.ndarray:
    d = dates.to_numpy("datetime64[D]")
    h = HOLIDAY_DATES.to_numpy("datetime64[D]")
    diff = (d[:, None] - h[None, :]).astype(int)
    return np.abs(diff).min(axis=1)


def quarter_end_ramp(dates: pd.Series) -> np.ndarray:
    """0 outside quarter-end months; inside Mar/Jun/Sep/Dec rises linearly from ~0 to 1
    across the month. In the training data rates climb ~3.5% from the start to the end of
    every quarter-end month, while other months are flat."""
    qe = dates.dt.month.isin([3, 6, 9, 12]).to_numpy()
    frac = ((dates.dt.day - 1) / (dates.dt.days_in_month - 1)).to_numpy()
    return np.where(qe, frac, 0.0)


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features that need no labels."""
    f = pd.DataFrame(index=df.index)
    f["distance"] = df["distance"]
    f["log_distance"] = np.log(df["distance"])
    f["straight_miles"] = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    f["circuity"] = df["distance"] / f["straight_miles"].clip(lower=1)
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        f[c] = df[c]
    f["bearing_lat"] = df["delivery_lat"] - df["pickup_lat"]
    f["bearing_lon"] = df["delivery_lon"] - df["pickup_lon"]
    f["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT)
    f["weight"] = df["weight"]
    f["weight_missing"] = df["weight_missing"]
    f["weight_capped"] = df["weight_capped"]
    f["market_index"] = df["market_index"]
    f["quote_signal"] = df["quote_signal"]
    f["day_of_week"] = df["date"].dt.dayofweek
    f["is_weekend"] = (f["day_of_week"] >= 5).astype(int)
    f["quarter_end_ramp"] = quarter_end_ramp(df["date"])
    # Holiday features were tested and removed: no measurable holiday effect in training data.
    return f


class TargetEncoder:
    """Smoothed mean of log rate-per-mile for pickup, delivery and lane.

    Fit on training rows; training rows get out-of-fold values so the model never
    sees its own label through the encoding. Unseen keys fall back to NaN and the
    model leans on coordinates instead.
    """

    KEYS = {"pickup_te": ["pickup"], "delivery_te": ["delivery"], "lane_te": ["pickup", "delivery"], "lane_eq_te": ["pickup", "delivery", "equipment"]}

    def __init__(self, smoothing: float = 20.0, n_splits: int = 5, seed: int = 0):
        self.smoothing, self.n_splits, self.seed = smoothing, n_splits, seed

    @staticmethod
    def _key(df, cols):
        return df[cols].astype(str).agg("|".join, axis=1)

    def _table(self, keys: pd.Series, y: pd.Series, prior: float) -> pd.Series:
        g = y.groupby(keys).agg(["sum", "count"])
        return (g["sum"] + self.smoothing * prior) / (g["count"] + self.smoothing)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.prior_ = float(y.mean())
        self.tables_ = {name: self._table(self._key(df, cols), y, self.prior_) for name, cols in self.KEYS.items()}
        out = pd.DataFrame(index=df.index, columns=list(self.KEYS), dtype=float)
        for tr, va in KFold(self.n_splits, shuffle=True, random_state=self.seed).split(df):
            for name, cols in self.KEYS.items():
                table = self._table(self._key(df.iloc[tr], cols), y.iloc[tr], self.prior_)
                out.iloc[va, out.columns.get_loc(name)] = self._key(df.iloc[va], cols).map(table).to_numpy()
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: self._key(df, cols).map(self.tables_[name]) for name, cols in self.KEYS.items()}, index=df.index)
