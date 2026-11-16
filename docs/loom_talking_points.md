# Loom talking points (2–3 minutes, camera on)

**1. Key findings (~35s)**
- Distance and equipment explain ~99% of variation in log rate; Reefer ~13% and Flatbed ~8% above Dry Van.
- market_index is a *daily* market level with a weekly cycle; one unit ≈ 15% on rate.
- Rates drift up ~0.6%/month beyond the market index.
- Biggest discovery: every quarter-end month (Mar, Jun, Sep) rates ramp from normal to about +4% by month end. December is a quarter-end month. *(show Figure 1)*

**2. Data quality (~30s)**
- 677 loads (1.4%) have corrupted rates, 0.2–0.45× or 2.2–5.5× expected, with a clean gap from real loads. Removed from training only. *(show Figure 2)* Training on raw labels: 2.09% vs 1.39% MAPE.
- Negative weights are sign flips → absolute value. Missing weight → equipment median. Missing market_index → same-day mean.
- 8 validation cities never seen in training (12% of loads) → model relies on coordinates; tested with leave-cities-out: 1.79% vs 1.39%.

**3. Model choice (~35s)**
- Plain LightGBM can't extrapolate the trend → underpredicts future months by 2–3%.
- Hybrid: linear MarketTrend (market index, trend, quarter-end ramp) as LightGBM's starting point; 300 trees learn lane/location corrections. *(show Figure 3)*
- Predicts log rate, so errors are proportional.

**4. Split & validation (~35s)**
- Data is Jan–Oct, target is Nov–Dec → random split would leak market conditions; always train on past, test on next two months.
- Primary holdout: train Jan–Aug, test Sep–Oct (1.39% MAPE, $32 MAE). Two more rolling folds → 1.62% average. Baseline 3.8%.
- Outlier detection, target encodings and trend model fit inside each training fold only; target encoding is out-of-fold.

**5. Code walkthrough (~30s)**
- `src/freight/data.py` – cleaning + `flag_label_outliers`
- `src/freight/features.py` – `quarter_end_ramp`, out-of-fold `TargetEncoder`
- `src/freight/model.py` – `MarketTrend` and `FreightRateModel` (init_score hybrid)
- `src/validate.py` → `src/predict.py` → `score.py`
- Close on the December chart: weekly market waves + quarter-end climb, $826 → $853, consistent with the lane's history (~$2.26/mile).
