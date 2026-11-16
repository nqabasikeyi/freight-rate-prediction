"""Train the final model on all clean Jan-Oct 2025 loads and write both prediction files.

Outputs:
  validation_predictions.csv          load_id,predicted_rate for the 12,000 validation loads
  data/december_chart_inputs.csv      the 31 December rows with predicted_rate filled in
  outputs/model.joblib, outputs/final_model_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freight.data import DATA_DIR, clean_features, daily_market_index, flag_label_outliers, load_train, load_validation  # noqa: E402
from freight.model import FreightRateModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def city_coordinates(*frames: pd.DataFrame) -> dict[str, tuple[float, float]]:
    rows = []
    for f in frames:
        rows.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        rows.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    coords = pd.concat(rows).groupby("city")[["lat", "lon"]].median()
    return {c: (r.lat, r.lon) for c, r in coords.iterrows()}


def build_december_frame(template: pd.DataFrame, validation: pd.DataFrame, coords: dict, daily_market: pd.Series) -> pd.DataFrame:
    """The December chart inputs only give lane, distance, equipment, weight and date.

    - Coordinates come from the city coordinates in the provided data.
    - market_index is a daily market level, so each date gets the mean market_index of the
      ~200 validation loads on that same date (a feature of the scoring set, not a label).
    - quote_signal is a per-load signal with little effect; each date gets the daily median
      of validation loads.
    """
    d = template.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["pickup_lat"], d["pickup_lon"] = zip(*d["pickup"].map(coords))
    d["delivery_lat"], d["delivery_lon"] = zip(*d["delivery"].map(coords))
    d["market_index"] = d["date"].map(daily_market)
    d["quote_signal"] = d["date"].map(validation.groupby("date")["quote_signal"].median())
    return clean_features(d, daily_market)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw_train, raw_val = load_train(), load_validation()
    daily = daily_market_index(raw_train, raw_val)
    train = clean_features(raw_train, daily)
    val = clean_features(raw_val, daily)
    outlier = flag_label_outliers(train)
    train_clean = train[~outlier]
    print(f"Training on {len(train_clean):,} clean loads (removed {outlier.sum()} corrupted labels)")

    model = FreightRateModel().fit(train_clean, train_clean["posted_rate"])
    joblib.dump(model, OUT / "model.joblib")

    # Validation predictions in template order
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    pred = pd.Series(model.predict(val), index=val["load_id"]).round(2)
    template["predicted_rate"] = template["load_id"].map(pred)
    assert template["predicted_rate"].notna().all() and (template["predicted_rate"] > 0).all()
    template.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv ({len(template):,} rows)")

    # December chart
    dec_path = DATA_DIR / "december_chart_inputs.csv"
    dec_template = pd.read_csv(dec_path)
    dec = build_december_frame(dec_template, raw_val, city_coordinates(raw_train, raw_val), daily)
    dec_template["predicted_rate"] = np.round(model.predict(dec), 2)
    dec_template.to_csv(dec_path, index=False)
    dec.assign(predicted_rate=dec_template["predicted_rate"].to_numpy())[["date", "market_index", "quote_signal", "predicted_rate"]].to_csv(
        OUT / "december_detail.csv", index=False
    )
    print(f"Filled {dec_path.relative_to(ROOT)} (Dec range ${dec_template.predicted_rate.min():,.2f} - ${dec_template.predicted_rate.max():,.2f})")

    summary = {
        "train_rows_clean": int(len(train_clean)),
        "label_outliers_removed": int(outlier.sum()),
        "monthly_trend_pct": model.market_.monthly_trend_pct,
        "quarter_end_uplift_pct": model.market_.quarter_end_effect_pct,
        "market_index_log_effect": model.market_.market_index_effect,
        "feature_importance": model.feature_importance().round(4).to_dict(),
        "validation_prediction_stats": template["predicted_rate"].describe().round(2).to_dict(),
    }
    json.dump(summary, open(OUT / "final_model_summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "feature_importance"}, indent=2))


if __name__ == "__main__":
    main()
