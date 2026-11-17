# Freight Rate Prediction

Everything is in one notebook: **`freight_rate_model.ipynb`**. It covers EDA, cleaning, time-based
validation, model comparison, ensembling, evaluation, the submission files and the December chart.
Outputs are saved in the notebook, so it reads on GitHub without running it.

**Final model:** a linear market-trend baseline (market index, monthly trend, quarter-end surge),
plus an equal-weight average of LightGBM and Random Forest that learn the remaining lane-level effects.

| Validation (train on past months, predict the next two) | MAPE | MAE |
|---|---|---|
| Final ensemble, average of 3 rolling folds | 1.57% | $37.35 |
| Final ensemble, primary holdout (train Jan–Aug, test Sep–Oct) | 1.41% | $32.90 |
| Rate-per-mile baseline, primary holdout | 3.82% | $82.78 |
| LightGBM without the trend baseline, primary holdout | 3.01% | — |

## Run it

```bash
python -m pip install -r requirements.txt
jupyter notebook freight_rate_model.ipynb      # then Run All (about 5 minutes on a 2-core laptop)
# or headless:
jupyter nbconvert --to notebook --execute --inplace freight_rate_model.ipynb
```

The notebook writes `validation_predictions.csv`, fills `data/december_chart_inputs.csv`, and runs
`score.py`, which creates `scorer_results/candidate_december.png`. To run the scorer separately:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Layout

```
freight_rate_model.ipynb    the whole solution
data/                       provided inputs (december_chart_inputs.csv is filled by the notebook)
score.py                    provided scorer
validation_predictions.csv  submission file (load_id,predicted_rate)
scorer_results/             December chart from score.py
outputs/figures/            charts saved by the notebook (used in the report)
report/                     Freight_Rate_Report.docx and the script that builds it
docs/                       Loom talking points
tools/build_notebook.py     generates the notebook from source (keeps diffs readable)
```
