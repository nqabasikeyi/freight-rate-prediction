// Builds report/Freight_Rate_Report.docx. Run from the repo root: node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, Footer, PageNumber,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FONT = "Arial";
const CONTENT_W = 9360; // US Letter, 1" margins
const ACCENT = "1F4E79";

const p = (text, opts = {}) =>
  new Paragraph({ spacing: { after: 120 }, ...opts, children: runs(text) });
// **bold** support inside plain strings
function runs(text) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((t) =>
    t.startsWith("**") ? new TextRun({ text: t.slice(2, -2), bold: true }) : new TextRun(t));
}
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: runs(t) });
const num = (t) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: runs(t) });
const caption = (t) => new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: t, italics: true, size: 18, color: "555555" })] });

function image(rel, widthIn) {
  const file = path.join(ROOT, rel);
  const buf = fs.readFileSync(file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); // PNG header
  const width = widthIn * 96;
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: buf, transformation: { width, height: Math.round((width * h) / w) } })],
  });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths, highlightRow = -1) {
  const cell = (text, w, { head = false, hl = false, right = false } = {}) =>
    new TableCell({
      width: { size: w, type: WidthType.DXA }, borders,
      shading: head ? { fill: ACCENT, type: ShadingType.CLEAR, color: "auto" } : hl ? { fill: "E8F0F8", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ alignment: right ? AlignmentType.RIGHT : AlignmentType.LEFT, children: [new TextRun({ text, bold: head || hl, color: head ? "FFFFFF" : "000000", size: 19 })] })],
    });
  return new Table({
    width: { size: widths.reduce((a, b) => a + b), type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, widths[i], { head: true, right: i > 0 && widths.length > 2 && !isNaN(parseFloat(rows[0][i])) })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((t, i) => cell(t, widths[i], { hl: ri === highlightRow, right: i > 0 && !isNaN(parseFloat(t.replace(/[$%+]/g, ""))) })) })),
    ],
  });
}
const spacer = () => new Paragraph({ spacing: { after: 120 }, children: [] });

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Freight Rate Prediction")] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: "Machine Learning Engineer assessment — approach, validation and results", color: "555555", size: 24 })] }),

  h1("1. Summary"),
  p("The task is to predict the posted rate for 12,000 freight loads from November–December 2025, using 48,000 labelled loads from January–October 2025. Because the loads to predict are in the future, every validation result in this report comes from **training on earlier months and testing on later months**, never from a random split."),
  p("The final model has two layers. A small linear model captures the market-wide drivers that change over time: the daily market index, a steady upward trend and a quarter-end rate surge. On top of it, an **equal-weight average of LightGBM and Random Forest** learns lane-, distance- and equipment-specific effects. That pair was chosen from four model families and every combination of them. All the work is in a single notebook, freight_rate_model.ipynb."),
  table(
    ["Result", "Value"],
    [
      ["Primary holdout (train Jan–Aug, test Sep–Oct): MAPE / MAE", "1.41%  /  $32.90"],
      ["Average over 3 rolling time folds: MAPE / MAE", "1.57%  /  $37.35"],
      ["Loads within ±2% / ±5% of actual (primary holdout)", "75% / 99.5%"],
      ["Rate-per-mile baseline, primary holdout: MAPE", "3.82%"],
      ["LightGBM without the trend baseline, primary holdout: MAPE", "3.01%"],
      ["Corrupted labels found and removed from training", "677 loads (1.4%)"],
    ],
    [6360, 3000],
  ),
  spacer(),

  h1("2. Key findings from the data"),
  bullet("**Distance and equipment explain most of the rate.** Rate per mile falls from about $2.55 on the shortest hauls to $1.91 on the longest. Median rate per mile is $2.05 for Dry Van, $2.22 for Flatbed and $2.31 for Reefer."),
  bullet("**market_index is a daily market level, not a per-load value.** It varies 0.17 overall but only 0.025 within a single day, and it follows a weekly cycle. +1 index moves rates by about 15%. Its row-level correlation with rate per mile is only 0.08, which hides this until distance and equipment effects are removed."),
  bullet("**Rates rise about 0.6% per month beyond what the market index explains.** Tree models cannot extrapolate this into November–December on their own."),
  bullet("**Rates climb through every quarter-end month.** In March, June and September, rates start the month at normal levels and rise steadily to about +4% by the last days. Other months are flat. The pattern repeats almost identically each quarter, and December is also a quarter-end month."),
  bullet("**No holiday effect** (Juneteenth looks high only because it falls inside the June ramp). Day-of-week effects are under ±0.5%. quote_signal is weak on its own but correlates 0.17 with what is left after other effects, so the models keep it."),
  bullet("**8 cities in the validation set never appear in training.** They touch 12% of the loads to predict, so the model must generalise through coordinates rather than memorising lanes."),
  image("outputs/figures/quarter_end_ramp.png", 6.0),
  caption("Figure 1. Rate relative to normal by day of month, after removing distance, equipment, weight, market index and trend effects."),

  h1("3. Data-quality issues and how they were handled"),
  table(
    ["Issue", "Rows (train / validation)", "Fix"],
    [
      ["Corrupted posted_rate: 0.17–0.45× or 2.2–5.3× the expected rate", "677 / n.a.", "Flagged with a robust distance + equipment + month fit; removed from training only. Normal loads sit at 0.85–1.2×, so the thresholds (0.6× / 1.7×) fall in an empty gap and the result is not sensitive to them."],
      ["Negative weight (sign error; magnitudes match normal loads)", "292 / 145", "Absolute value, plus a flag column"],
      ["Missing weight", "300 / 165", "Equipment-level median, plus a flag column"],
      ["Weight capped at 47,500 lb", "1,191 at the cap", "Kept; flag column marks capped rows"],
      ["Missing market_index", "374 / 249", "Mean market_index of all loads on the same date (it is a daily value)"],
      ["Road distance > 1.5× straight-line distance", "214", "Checked: all short hauls (70–224 miles) where roads are naturally less direct. Not an error; kept as a circuity feature."],
    ],
    [3200, 1900, 4260],
  ),
  spacer(),
  image("outputs/figures/label_outliers.png", 6.0),
  caption("Figure 2. Posted rate divided by expected rate. Corrupted labels (orange) are clearly separated from normal loads."),

  h1("4. Train/test split and validation approach"),
  h2("Why the split is by time"),
  p("The training data runs from 1 January to 31 October 2025 and the loads to predict run from 1 November to 31 December 2025. A random 80/20 split would let the model learn from loads on the same days as the test loads, so it would see the same market conditions it is being tested on. That overstates accuracy, and it would hide the most important risk here: tree models cannot extrapolate a trend into months they have not seen. Every split therefore trains on the past and tests on the following two months, mirroring the real task."),
  h2("Folds"),
  table(
    ["Fold", "Train on", "Test on", "Final ensemble MAPE"],
    [
      ["1", "Jan–Jun", "Jul–Aug", "1.83%"],
      ["2", "Jan–Jul", "Aug–Sep", "1.48%"],
      ["3 (primary holdout)", "Jan–Aug", "Sep–Oct", "1.41%"],
      ["Average", "", "", "1.57%"],
    ],
    [2760, 2000, 2000, 2600],
    2,
  ),
  spacer(),
  bullet("**Primary holdout:** Sep–Oct is the closest stand-in for Nov–Dec. It is two months long, it follows the training period directly and it contains a quarter-end month, just like the real task."),
  bullet("**Rolling folds** check that the result is not a one-off, and they are what the model and the ensemble were selected on. Fold 1 is the weakest because with only six months of training data the trend and the rising market index are hard to separate."),
  bullet("**No leakage inside folds.** Corrupted-label detection, lane target encodings and the linear market model are all fitted on the training months only. Target encodings use 5-fold out-of-fold values so a load never sees its own rate."),
  bullet("**Metrics:** MAPE to compare models, plus MAE in dollars and bias. They are computed on clean labels (what the model is designed to predict) and, for the final model, also on all labels."),
  bullet("**Final model:** after selection, the ensemble is retrained on all 47,323 clean January–October loads."),

  h1("5. Model selection"),
  h2("Baselines and four model families"),
  p("Three baselines set the bar: a rate-per-mile lookup table, the linear market-trend model alone, and a standard LightGBM predicting log rate directly. That LightGBM underpredicts every future period by 2–3%, confirming that tree models need the trend handled separately. So four cheap model families were each trained on the **residual** left after the market-trend baseline:"),
  table(
    ["Model", "Family", "Fold 1", "Fold 2", "Fold 3", "Average"],
    [
      ["LightGBM + Random Forest (average)", "Ensemble", "1.83%", "1.48%", "1.41%", "1.57%"],
      ["Random Forest, 150 trees", "Bagged trees", "1.85%", "1.50%", "1.49%", "1.61%"],
      ["LightGBM, 300 trees", "Boosted trees", "1.89%", "1.57%", "1.38%", "1.61%"],
      ["KNN, k=25", "Nearest neighbours", "1.98%", "1.69%", "1.67%", "1.78%"],
      ["Ridge", "Linear", "2.12%", "1.80%", "1.83%", "1.92%"],
      ["Market trend only", "Baseline", "2.80%", "2.65%", "2.65%", "2.70%"],
      ["LightGBM without trend", "Baseline", "3.04%", "3.00%", "3.01%", "3.02%"],
      ["Rate-per-mile table", "Baseline", "3.97%", "3.83%", "3.82%", "3.87%"],
    ],
    [2860, 1700, 1200, 1200, 1200, 1200],
    0,
  ),
  caption("Table: MAPE by time fold. All four families take under 30 seconds to train across the three folds."),
  h2("Ensembling"),
  p("Soft and hard voting are classification techniques: they combine predicted classes or class probabilities. Here the target is a dollar amount, so the regression equivalent is averaging the models' predictions (scikit-learn's VotingRegressor). Predictions are averaged in log space, a geometric mean that suits rates from $200 to $7,000. Every combination of 2, 3 and 4 families was scored with equal weights."),
  bullet("**LightGBM + Random Forest is the best combination** (1.57% average). Bagging and boosting make different errors, so averaging them helps."),
  bullet("**Adding Ridge or KNN makes it worse.** Diversity only helps when the extra models are also accurate."),
  bullet("**On the Sep–Oct fold alone, LightGBM is marginally better** (1.38% vs 1.41%). The ensemble was chosen because it is better on average and more consistent across periods."),
  bullet("**Weights are kept equal.** Tuning weights on three folds would risk overfitting the validation set for a tiny gain."),
  image("outputs/figures/model_comparison.png", 6.0),
  caption("Figure 3. Error on the Sep–Oct holdout, all models trained on Jan–Aug."),

  h1("6. Evaluation of the final model"),
  p("A confusion matrix needs classes, and this is a regression problem, so the evaluation uses the regression equivalents: error breakdowns, predicted-vs-actual and residual plots."),
  image("outputs/figures/holdout_diagnostics.png", 6.5),
  caption("Figure 4. Primary holdout (Sep–Oct). Errors are centred near zero and show no pattern against distance."),
  bullet("**Overall:** 1.41% MAPE ($32.90 MAE) on clean labels, with 75% of loads within ±2% and 99.5% within ±5%."),
  bullet("**Balanced across segments:** by equipment, MAPE is 1.34–1.51%; by distance band 1.35–1.55%; by price band 1.38–1.53%. Binning rates into five price bands for illustration, 97.9% of loads land in the correct band, and the rest are near a band boundary."),
  bullet("**Error grows with the forecast horizon:** 1.30% in September, 1.51% in October (October runs about 1% low). December will be the hardest month."),
  bullet("**Unseen cities:** repeating the split with 8 random cities removed from training (3 repeats), loads involving those cities scored 1.73% MAPE against 1.41% for known cities. The model degrades gracefully because coordinates carry most of the location signal."),
  bullet("**All labels, including corrupted ones:** 3.93% MAPE / $89 MAE. The gap comes almost entirely from the 1.4% of loads whose recorded rate is off by 2–5×."),
  image("outputs/figures/holdout_daily.png", 6.0),
  caption("Figure 5. Daily average rate per mile on the Sep–Oct holdout, model trained on Jan–Aug only."),

  h1("7. December chart"),
  p("The chart fixes the load (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb) and changes only the date. The chart inputs do not include market_index or quote_signal. Each December date therefore uses the mean market_index and median quote_signal of the ~200 validation loads on that same date. These are features of the loads being scored, not labels."),
  image("scorer_results/candidate_december.png", 6.5),
  caption("Figure 6. Produced by the provided score.py."),
  bullet("**Weekly waves** come from the daily market index, which follows a weekly cycle in both the training data and the December validation loads."),
  bullet("**The upward drift**, from $828 to $867, is the quarter-end surge seen in every previous quarter-end month, plus the 0.6%/month trend."),
  bullet("**Sanity check:** Dry Van loads on this lane in training ranged $758–$934 (median $2.23 per mile). December predictions of $2.30–$2.41 per mile sit slightly above that, as expected given the trend and the quarter-end surge."),

  h1("8. Limitations and next steps"),
  bullet("**The trend is assumed to continue linearly into December.** It is stable across training windows, but a two-month extrapolation is the main source of risk. With live data I would re-estimate it weekly."),
  bullet("**December has no labelled precedent.** The quarter-end ramp is learned from March, June and September; December may also have year-end effects that no training month shows."),
  bullet("**Corrupted labels in the scoring data.** If the final ground truth contains the same ~1.4% of corrupted rates, measured error will be higher (about $89 MAE rather than $33 on the holdout). The model predicts the true market rate rather than trying to guess the corruption."),
  bullet("**Possible improvements:** prediction intervals with quantile models; a lagged market-index feature; and drift monitoring for the trend and quarter-end effect."),

  h1("9. Reproducing the results"),
  p("Install requirements.txt, open freight_rate_model.ipynb and run all cells (about five minutes). The notebook writes validation_predictions.csv, fills data/december_chart_inputs.csv and runs score.py to create the December chart."),
];

const doc = new Document({
  creator: "Candidate",
  title: "Freight Rate Prediction Report",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 44, bold: true, color: ACCENT }, paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 30, bold: true, color: ACCENT }, paragraph: { spacing: { before: 320, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: "333333" }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: "777777" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "report", "Freight_Rate_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote", out);
});
