"""Loading and cleaning of the freight load data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
WEIGHT_CAP = 47_500.0

# A load is treated as a corrupted label when its posted rate is this far from
# what a simple distance/equipment/month model expects. The real distribution of
# ratios sits in ~[0.8, 1.45]; the corrupted rows sit at ~0.2-0.4x and ~2.2-5x,
# with an empty gap in between, so the exact thresholds are not sensitive.
OUTLIER_LOW, OUTLIER_HIGH = 0.6, 1.7


def load_train(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "train_test.csv", parse_dates=["date"])


def load_validation(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "validation.csv", parse_dates=["date"])


def clean_features(df: pd.DataFrame, daily_market: pd.Series | None = None) -> pd.DataFrame:
    """Fix feature-level data quality issues. Safe for train, validation and December rows.

    - Negative weights are sign errors: take the absolute value.
    - Weight is capped at 47,500 lb in the source data: keep a flag for capped rows.
    - Missing weight: equipment-level median (with a missing flag).
    - Missing market_index: market_index is a daily market level (std within a day
      is ~0.025 vs ~0.17 overall), so fill it with the mean of other loads on the same date.
    """
    out = df.copy()
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].fillna(out.groupby("equipment")["weight"].transform("median"))
    out["weight"] = out["weight"].fillna(out["weight"].median())
    out["weight_capped"] = (out["weight"] >= WEIGHT_CAP).astype(int)

    if "market_index" not in out:
        out["market_index"] = np.nan
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    same_day = out.groupby("date")["market_index"].transform("mean")
    out["market_index"] = out["market_index"].fillna(same_day)
    if daily_market is not None:
        out["market_index"] = out["market_index"].fillna(out["date"].map(daily_market))
    return out


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Mean market_index per date across all available rows (features only, no labels)."""
    allrows = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    return allrows.groupby("date")["market_index"].mean()


def flag_label_outliers(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask of rows whose posted_rate looks corrupted."""
    X = np.column_stack(
        [
            np.ones(len(df)),
            np.log(df["distance"]),
            (df["equipment"] == "Reefer").astype(float),
            (df["equipment"] == "Flatbed").astype(float),
            pd.get_dummies(df["date"].dt.month, drop_first=True).to_numpy(float),
        ]
    )
    y = np.log(df["posted_rate"].to_numpy())
    # Two passes of least squares: refit after dropping gross outliers.
    keep = np.ones(len(df), dtype=bool)
    for _ in range(2):
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        ratio = np.exp(y - X @ beta)
        keep = (ratio > OUTLIER_LOW) & (ratio < OUTLIER_HIGH)
    return pd.Series(~keep, index=df.index, name="label_outlier")
