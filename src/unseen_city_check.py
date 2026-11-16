"""Robustness check: how well does the model price loads touching cities it has never seen?

validation.csv contains 8 cities absent from train_test.csv. To estimate error on those loads
we repeat the primary split (train Jan-Aug, test Sep-Oct) but also remove every training load
that touches 8 randomly chosen cities, then score test loads touching those cities.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freight.data import clean_features, daily_market_index, flag_label_outliers, load_train, load_validation  # noqa: E402
from freight.model import FreightRateModel, metrics  # noqa: E402

raw, val = load_train(), load_validation()
df = clean_features(raw, daily_market_index(raw, val))
df = df[~flag_label_outliers(df)]
unseen = set(val.pickup) | set(val.delivery)
unseen -= set(raw.pickup) | set(raw.delivery)
touch_val = val.pickup.isin(unseen) | val.delivery.isin(unseen)
print(f"Validation loads touching an unseen city: {touch_val.sum()} ({touch_val.mean():.1%})")

cities = np.array(sorted(set(df.pickup) | set(df.delivery)))
results = []
for seed in range(3):
    held = set(np.random.default_rng(seed).choice(cities, 8, replace=False))
    touches = df.pickup.isin(held) | df.delivery.isin(held)
    tr = df[(df.date < "2025-09-01") & ~touches]
    te = df[(df.date >= "2025-09-01")]
    p = FreightRateModel().fit(tr, tr.posted_rate).predict(te)
    m_new = metrics(te.posted_rate[touches.loc[te.index]], p[touches.loc[te.index].to_numpy()])
    m_old = metrics(te.posted_rate[~touches.loc[te.index]], p[~touches.loc[te.index].to_numpy()])
    results.append({"seed": seed, "unseen_city_MAPE": m_new["MAPE_%"], "seen_MAPE": m_old["MAPE_%"], "n_unseen": m_new["n"]})
    print(results[-1])
out = {"validation_unseen_city_share": float(touch_val.mean()), "runs": results,
       "mean_unseen_city_MAPE": float(np.mean([r["unseen_city_MAPE"] for r in results])),
       "mean_seen_MAPE": float(np.mean([r["seen_MAPE"] for r in results]))}
json.dump(out, open(Path(__file__).resolve().parents[1] / "outputs" / "unseen_city_check.json", "w"), indent=2)
print(out)
