"""Time-based validation of baselines and the LightGBM model.

Split design (mirrors the real task: train on the past, predict the next two months):
  * Primary holdout: train Jan-Aug 2025, test Sep-Oct 2025.
  * Rolling-origin backtest: three folds, each training on all months before a cutoff and
    testing on the following two months.
Label outliers are removed from training data only. Test metrics are reported both on
clean labels (what the model is designed to predict) and on all labels (raw).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freight.data import clean_features, daily_market_index, flag_label_outliers, load_train, load_validation  # noqa: E402
from freight.model import FreightRateModel, LogLinearModel, MarketTrend, RatePerMileBaseline, metrics  # noqa: E402


class TrendOnly:
    def fit(self, df, y):
        self.m = MarketTrend().fit(df, np.log(y))
        return self

    def predict(self, df):
        return np.exp(self.m.predict_log(df))

OUT = Path(__file__).resolve().parents[1] / "outputs"

FOLDS = [  # (train_end_exclusive, test_end_exclusive)
    ("2025-07-01", "2025-09-01"),
    ("2025-08-01", "2025-10-01"),
    ("2025-09-01", "2025-11-01"),  # primary holdout: Sep-Oct
]

MODELS = {
    "Baseline: rate-per-mile table": lambda: RatePerMileBaseline(),
    "Log-linear ridge": lambda: LogLinearModel(),
    "Market trend only (linear)": "trend",
    "LightGBM only (no trend component)": lambda: FreightRateModel(trend=False, n_estimators=1500),
    "Hybrid, no target encoding": lambda: FreightRateModel(use_target_encoding=False),
    "Hybrid trend + LightGBM (final)": lambda: FreightRateModel(),
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = load_train()
    daily = daily_market_index(raw, load_validation())
    df = clean_features(raw, daily)
    df["label_outlier"] = flag_label_outliers(df)
    print(f"Label outliers flagged: {df['label_outlier'].sum()} of {len(df)} ({df['label_outlier'].mean():.2%})")

    rows, preds_holdout = [], None
    for train_end, test_end in FOLDS:
        tr = df[df["date"] < train_end]
        te = df[(df["date"] >= train_end) & (df["date"] < test_end)]
        tr_clean = tr[~tr["label_outlier"]]
        te_clean = te[~te["label_outlier"]]
        fold = f"train<{train_end} | test {train_end}..{test_end}"
        variants = dict(MODELS)
        if test_end == "2025-11-01":
            variants["Final model trained on raw labels"] = "raw"
        for name, make in variants.items():
            if make == "raw":
                model = FreightRateModel().fit(tr, tr["posted_rate"])
            elif make == "trend":
                model = TrendOnly().fit(tr_clean, tr_clean["posted_rate"])
            else:
                model = make().fit(tr_clean, tr_clean["posted_rate"])
            p = pd.Series(model.predict(te), index=te.index)
            rows.append({"fold": fold, "model": name, "labels": "clean", **metrics(te_clean["posted_rate"], p[te_clean.index])})
            rows.append({"fold": fold, "model": name, "labels": "raw", **metrics(te["posted_rate"], p)})
            if test_end == "2025-11-01" and name == "Hybrid trend + LightGBM (final)":
                preds_holdout = te.assign(predicted_rate=p)
                importance = model.feature_importance()
            rows[-2]["bias_%"] = rows[-1]["bias_%"] = float(np.mean(np.log(p[te_clean.index] / te_clean["posted_rate"])) * 100)
            print(f"{fold:45s} {name:40s} MAE clean={rows[-2]['MAE']:.2f}  MAPE clean={rows[-2]['MAPE_%']:.2f}%")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "validation_metrics.csv", index=False)
    summary = (
        res[res["labels"] == "clean"].groupby("model")[["MAE", "RMSE", "MAPE_%", "bias_%"]].mean().sort_values("MAE")
    )
    print("\nMean over rolling folds (clean labels):\n", summary.round(2))

    # Holdout breakdowns for the report
    h = preds_holdout[~preds_holdout["label_outlier"]].copy()
    seen_lanes = set(zip(df.loc[df["date"] < "2025-09-01", "pickup"], df.loc[df["date"] < "2025-09-01", "delivery"]))
    h["lane_seen"] = [(a, b) in seen_lanes for a, b in zip(h["pickup"], h["delivery"])]
    breakdown = {
        "by_equipment": {k: metrics(g["posted_rate"], g["predicted_rate"]) for k, g in h.groupby("equipment")},
        "by_month": {str(k): metrics(g["posted_rate"], g["predicted_rate"]) for k, g in h.groupby(h["date"].dt.month)},
        "by_lane_seen": {str(k): metrics(g["posted_rate"], g["predicted_rate"]) for k, g in h.groupby("lane_seen")},
    }
    json.dump(
        {
            "label_outliers": int(df["label_outlier"].sum()),
            "rows": len(df),
            "summary_clean": summary.round(4).to_dict("index"),
            "holdout_breakdown": breakdown,
            "feature_importance": importance.round(4).to_dict(),
        },
        open(OUT / "validation_summary.json", "w"),
        indent=2,
    )
    preds_holdout[["load_id", "date", "pickup", "delivery", "equipment", "distance", "posted_rate", "predicted_rate", "label_outlier"]].to_csv(
        OUT / "holdout_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
