# Loom talking points (2–3 minutes, camera on). Screen-share the notebook

**1. Key findings: notebook section 1 (~35s)**
- Show `.info()` / `.describe()`: train is Jan–Oct, we predict Nov–Dec, so this is forecasting.
- Distance and equipment drive most of the rate (Reefer $2.31/mile, Flatbed $2.22, Dry Van $2.05).
- market_index is a *daily* market level with a weekly cycle; +1 index ≈ +15% on rate.
- Rates drift up ~0.6%/month, and in every quarter-end month they ramp to about +4% by month end. *(scroll to the quarter-end chart)* December is a quarter-end month.

**2. Data quality: section 1.4 (~30s)**
- 677 corrupted rates (0.17–0.45× or 2.2–5.3× expected), with a clear gap from real loads. Removed from training only. *(histogram)*
- Negative weights are sign flips → absolute value. Missing weight → equipment median. Missing market_index → same-day mean.
- 8 validation cities never appear in training (12% of loads): the robustness check gives 1.73% vs 1.41%.

**3. Split and validation: section 3 (~30s)**
- A random split would leak market conditions; always train on the past, test on the next two months.
- Three rolling folds; Sep–Oct is the primary holdout because it mirrors Nov–Dec, including a quarter-end month.
- Outlier detection, target encodings and the trend model are fitted inside each training fold.

**4. Model choice: sections 4–5 (~40s)**
- Plain LightGBM underpredicts future months by 2–3%: trees can't extrapolate the trend.
- So a linear market-trend baseline handles trend and quarter-end, and four families learn the residual: Ridge, LightGBM, Random Forest, KNN.
- Voting is for classification; for regression we average predictions (VotingRegressor equivalent). Every combination was tested.
- LightGBM + Random Forest wins: 1.57% average MAPE vs 3.87% for the rate-per-mile baseline. Adding Ridge or KNN makes it worse.
- No confusion matrix: this is regression. Show predicted-vs-actual, residuals, and the price-band table instead.

**5. Results and December (~25s)**
- Sep–Oct: 1.41% MAPE, $33 MAE, 75% of loads within ±2%.
- December chart: weekly market waves plus the quarter-end climb, $828 → $867, in line with the lane's ~$2.23/mile history.
