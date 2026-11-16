# Freight Rate Prediction

Predicts the posted rate of a freight load from lane, distance, equipment, weight, date and
market signals. The final model is a **hybrid**: a linear "market" component (market index,
long-run trend, quarter-end ramp) gives the baseline, and LightGBM learns corrections on top.

**Validation (rolling time folds, train on the past → predict the next two months):**
MAPE 1.62% / MAE $38.60 on average; primary Sep–Oct holdout MAPE 1.39% / MAE $32.34.
Rate-per-mile baseline: 3.87%. Plain LightGBM: 2.91%.

## Run it

```bash
python -m pip install -r requirements.txt

python src/validate.py            # time-based validation, writes outputs/validation_*.{csv,json}
python src/predict.py             # trains on all clean Jan–Oct data, writes validation_predictions.csv
                                  # and fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv   # creates scorer_results/candidate_december.png

# optional
python src/unseen_city_check.py   # error on cities the model has never seen
python src/report_figures.py      # figures used in the report
```

Python 3.10+. Full pipeline runs in about 3 minutes on a laptop.

## Layout

```
data/                       provided inputs (december_chart_inputs.csv is filled by predict.py)
src/freight/data.py         loading, cleaning, label-outlier detection
src/freight/features.py     features, quarter-end ramp, out-of-fold target encoding
src/freight/model.py        baselines, MarketTrend (linear) and FreightRateModel (hybrid)
src/validate.py             rolling time-based validation + model comparison
src/predict.py              final training and both prediction files
src/unseen_city_check.py    leave-cities-out robustness check
src/report_figures.py       report charts
validation_predictions.csv  submission file (load_id,predicted_rate)
scorer_results/             chart produced by score.py
outputs/                    metrics, summaries, figures
```

## Key decisions

- **Time-based split, not random.** Training data covers Jan–Oct 2025; the loads to predict are
  Nov–Dec 2025. Validation always trains on earlier months and tests on the next two.
- **Corrupted labels removed from training (677 rows, 1.4%).** Their rates are 0.2–0.45× or
  2.2–5.5× the expected rate, separated from normal loads by a clear gap.
- **Feature fixes:** negative weights are sign errors (absolute value); missing weight filled with
  the equipment median; missing market_index filled with the same-day mean (it is a daily market level).
- **Trend and quarter-end handled linearly** because tree models cannot extrapolate beyond the
  training period: rates rise ~0.6%/month beyond the market index, and climb ~4% across every
  quarter-end month (Mar/Jun/Sep, and therefore December).
- **December chart inputs** lack market_index and quote_signal; they are filled with the daily
  mean/median of the validation loads on the same date (scoring-set features, not labels).
