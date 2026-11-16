"""Model definitions: baselines and the LightGBM rate model."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .features import TargetEncoder, base_features, quarter_end_ramp

LGB_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    n_estimators=300,  # chosen on rolling folds; trees only learn corrections on top of MarketTrend
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(np.mean(np.abs(err) / y_true) * 100),
        "MedAPE_%": float(np.median(np.abs(err) / y_true) * 100),
        "n": int(len(y_true)),
    }


class RatePerMileBaseline:
    """Median rate-per-mile by equipment x distance band, times distance."""

    BANDS = [0, 250, 500, 1000, 1500, 2000, 1e9]

    def fit(self, df, y):
        band = pd.cut(df["distance"], self.BANDS)
        self.table_ = (y / df["distance"]).groupby([df["equipment"], band], observed=True).median()
        self.global_ = float((y / df["distance"]).median())
        return self

    def predict(self, df):
        band = pd.cut(df["distance"], self.BANDS)
        idx = pd.MultiIndex.from_arrays([df["equipment"], band])
        rpm = pd.Series(self.table_.reindex(idx).to_numpy(), index=df.index).fillna(self.global_)
        return (rpm * df["distance"]).to_numpy()


class LogLinearModel:
    """Ridge regression on log(rate): log distance, equipment, weight, market_index, quote_signal."""

    def _X(self, df):
        f = base_features(df)
        X = pd.get_dummies(f[["log_distance", "weight", "market_index", "quote_signal", "equipment", "day_of_week"]], columns=["equipment", "day_of_week"], dtype=float)
        X["weight"] /= 1e4
        return X

    def fit(self, df, y):
        X = self._X(df)
        self.cols_ = X.columns
        self.m_ = Ridge(alpha=1.0).fit(X, np.log(y))
        return self

    def predict(self, df):
        return np.exp(self.m_.predict(self._X(df).reindex(columns=self.cols_, fill_value=0)))


T0 = pd.Timestamp("2025-01-01")


class MarketTrend:
    """Linear model of log(rate) on the market drivers that trees cannot extrapolate.

    log(rate) ~ log(distance) + log(distance)^2 + equipment + weight + market_index + months_since_start

    In the training data rates rise ~0.6%/month beyond what market_index explains. A tree
    model trained on Jan-Aug would predict Sep-Dec at the level of the last training months,
    so the trend is handled here, linearly, and LightGBM models the residual.
    """

    def _X(self, df):
        ld = np.log(df["distance"].to_numpy())
        t = (df["date"] - T0).dt.days.to_numpy() / 30.4
        return np.column_stack(
            [
                np.ones(len(df)), ld, ld**2,
                (df["equipment"] == "Reefer").to_numpy(float), (df["equipment"] == "Flatbed").to_numpy(float),
                df["weight"].to_numpy() / 1e4, quarter_end_ramp(df["date"]),
                df["market_index"].to_numpy(), t,
            ]
        )

    def fit(self, df, y_log):
        self.beta_, *_ = np.linalg.lstsq(self._X(df), np.asarray(y_log), rcond=None)
        return self

    def predict_log(self, df):
        return self._X(df) @ self.beta_

    @property
    def monthly_trend_pct(self) -> float:
        return float(np.expm1(self.beta_[-1]) * 100)

    @property
    def quarter_end_effect_pct(self) -> float:
        """Rate uplift on the last day of a quarter-end month."""
        return float(np.expm1(self.beta_[-3]) * 100)

    @property
    def market_index_effect(self) -> float:
        return float(self.beta_[-2])


class FreightRateModel:
    """LightGBM on log(posted_rate) with row features and out-of-fold target encodings.

    With trend=True (default) the model is a hybrid: MarketTrend gives the starting
    prediction (LightGBM init_score) and the trees learn corrections on top of it.
    """

    def __init__(self, use_target_encoding: bool = True, trend: bool = True, params: dict | None = None, n_estimators: int | None = None):
        self.use_te = use_target_encoding
        self.trend = trend
        self.params = {**LGB_PARAMS, **(params or {})}
        if n_estimators:
            self.params["n_estimators"] = n_estimators

    def _features(self, df, te: pd.DataFrame | None):
        X = base_features(df)
        if te is not None:
            X = X.join(te)
        return X

    def fit(self, df: pd.DataFrame, y: pd.Series):
        y_log = np.log(y)
        te = None
        if self.use_te:
            self.encoder_ = TargetEncoder()
            te = self.encoder_.fit_transform(df, np.log(y / df["distance"]))
        X = self._features(df, te)
        self.feature_names_ = list(X.columns)
        init = None
        if self.trend:
            self.market_ = MarketTrend().fit(df, y_log)
            init = self.market_.predict_log(df)
        self.model_ = lgb.LGBMRegressor(**self.params).fit(X, y_log, init_score=init)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        te = self.encoder_.transform(df) if self.use_te else None
        X = self._features(df, te)[self.feature_names_]
        log_pred = self.model_.predict(X, raw_score=True)
        if self.trend:
            log_pred = log_pred + self.market_.predict_log(df)
        # exp(log prediction) estimates the conditional median, which is what MAE rewards.
        return np.exp(log_pred)

    def feature_importance(self) -> pd.Series:
        gain = self.model_.booster_.feature_importance("gain")
        return pd.Series(gain / gain.sum(), index=self.feature_names_).sort_values(ascending=False)
