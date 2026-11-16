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
  p("The final model is a **hybrid**: a small linear model captures the market-wide drivers that change over time (the daily market index, a steady upward trend and a quarter-end rate surge), and LightGBM learns the lane-, distance- and equipment-specific corrections on top."),
  table(
    ["Result", "Value"],
    [
      ["Primary holdout (train Jan–Aug, test Sep–Oct): MAPE / MAE", "1.39%  /  $32.34"],
      ["Average over 3 rolling time folds: MAPE / MAE", "1.62%  /  $38.60"],
      ["Simple rate-per-mile baseline, same holdout: MAPE", "3.82%"],
      ["Standard LightGBM without the trend component: MAPE", "3.02%"],
      ["Corrupted labels found and removed from training", "677 loads (1.4%)"],
    ],
    [6360, 3000],
  ),
  spacer(),

  h1("2. Key findings from the data"),
  bullet("**Distance and equipment explain most of the rate.** A log-linear fit on distance and equipment alone explains 99.4% of the variance in log rate. Reefer runs ~13% and Flatbed ~8% above Dry Van; rate per mile falls from about $2.55 on short hauls to $1.91 on the longest."),
  bullet("**market_index is a daily market level, not a per-load value.** It varies 0.17 overall but only 0.025 within a single day, and it follows a weekly cycle. A one-unit change moves rates by about 15%."),
  bullet("**Rates rise about 0.6% per month beyond what the market index explains.** In January and October the market index was similar, yet October rates were about 6% higher. The estimate is stable (0.53–0.72%/month) once seven or more months of data are used."),
  bullet("**Rates climb through every quarter-end month.** In March, June and September, rates start the month at normal levels and rise steadily to about +4% by the last days. Other months are flat. This repeats almost identically in all three quarters, and December is also a quarter-end month."),
  bullet("**No measurable holiday effect** (Jan 1, Memorial Day, July 4, Labor Day). Day-of-week effects are small (about ±1%). **quote_signal carries little information** (correlation with rate per mile 0.05)."),
  bullet("**8 cities in the validation set never appear in training**, and 12% of validation loads touch one of them, so the model must generalise through coordinates rather than memorising lanes."),
  image("outputs/figures/quarter_end_ramp.png", 6.0),
  caption("Figure 1. Rate relative to normal by day of month, after removing distance, equipment, weight, market index and trend effects."),

  h1("3. Data-quality issues and how they were handled"),
  table(
    ["Issue", "Rows (train / validation)", "Fix"],
    [
      ["Corrupted posted_rate: 0.2–0.45× or 2.2–5.5× the expected rate", "677 / n.a.", "Flagged with a robust distance + equipment + month fit; removed from training only. A wide empty gap separates them from real loads, so the thresholds (0.6× / 1.7×) are not sensitive."],
      ["Negative weight (sign error; magnitudes match normal loads)", "292 / present", "Absolute value, plus a flag column"],
      ["Missing weight", "300 / 165", "Equipment-level median, plus a flag column"],
      ["Weight capped at 47,500 lb", "1,191 at the cap", "Kept; flag column marks capped rows"],
      ["Missing market_index", "374 / 249", "Mean market_index of other loads on the same date (it is a daily value)"],
      ["Road distance vs straight-line distance", "214 ratios > 1.5", "Checked: all short hauls where road circuity is naturally higher; no fix needed. Circuity kept as a feature."],
    ],
    [3200, 1900, 4260],
  ),
  spacer(),
  image("outputs/figures/label_outliers.png", 6.0),
  caption("Figure 2. Posted rate divided by expected rate. Corrupted labels (orange) are clearly separated from normal loads."),
  p("Removing corrupted labels matters: the same model trained on raw labels scores 2.09% MAPE on the Sep–Oct holdout instead of 1.39%, because squared-error training is pulled towards the extreme values."),

  h1("4. Train/test split and validation approach"),
  h2("Why the split is by time"),
  p("The training data runs from 1 January to 31 October 2025 and the loads to predict run from 1 November to 31 December 2025. A random 80/20 split would let the model learn from loads on the same days as the test loads, so it would see the same market conditions it is being tested on. That overstates accuracy, and it would hide the most important failure in this problem: tree models cannot extrapolate a trend into months they have not seen. Every split in this project therefore trains on the past and tests on the following two months, mirroring the real task."),
  h2("Folds"),
  table(
    ["Fold", "Train on", "Test on", "Final model MAPE", "Final model MAE"],
    [
      ["1", "Jan–Jun", "Jul–Aug", "1.90%", "$46.74"],
      ["2", "Jan–Jul", "Aug–Sep", "1.57%", "$36.72"],
      ["3 (primary holdout)", "Jan–Aug", "Sep–Oct", "1.39%", "$32.34"],
      ["Average", "", "", "1.62%", "$38.60"],
    ],
    [2160, 1700, 1700, 1900, 1900],
    2,
  ),
  spacer(),
  bullet("**Primary holdout:** Sep–Oct is the closest stand-in for Nov–Dec. It is two months long, it follows the training period directly and it contains a quarter-end month, just like the real task."),
  bullet("**Rolling folds** check that the result is not a one-off. Fold 1 is the weakest because with only six months of training data the trend and the rising market index move together and are hard to separate; accuracy improves as training data grows."),
  bullet("**No leakage inside folds.** Corrupted-label detection, lane target encodings and the linear market model are all fitted on the training months only. Target encodings use 5-fold out-of-fold values so a load never sees its own rate."),
  bullet("**Metrics** are MAE, RMSE, MAPE and bias, reported on clean labels (what the model is designed to predict) and also on all labels. Including corrupted labels, the primary holdout scores 3.91% MAPE / $88 MAE; that error comes almost entirely from the 1.4% of loads whose recorded rate is wrong."),
  bullet("**Unseen cities:** to test the 8 new cities, I repeated the primary split while also removing every training load that touches 8 randomly chosen cities (3 repeats). Loads involving those cities scored 1.79% MAPE versus 1.39% for known cities, so the model degrades gracefully."),
  bullet("**Final model:** after validation, the model is retrained on all clean January–October loads with the same settings."),

  h1("5. Model choice"),
  p("I compared six approaches on identical splits:"),
  image("outputs/figures/model_comparison.png", 6.0),
  caption("Figure 3. Error on the Sep–Oct holdout, all models trained on Jan–Aug."),
  p("Standard LightGBM is accurate within the training period but underpredicts later months by 2–3%, because a tree model can only output values it has seen and the market trend keeps rising. The linear market model extrapolates the trend correctly but cannot capture lane-level detail. The hybrid uses each where it is strong:"),
  num("**MarketTrend (linear, log space):** log distance and its square, equipment, weight, market_index, months since January and the quarter-end ramp (0 outside Mar/Jun/Sep/Dec, rising from 0 to 1 across those months)."),
  num("**LightGBM (300 trees):** starts from the MarketTrend prediction (init_score) and learns corrections from coordinates, lane/pickup/delivery target encodings, distance, circuity, weight, equipment, market_index, quote_signal and day of week. Predicting log rate means errors are proportional, which suits loads ranging from $200 to $7,000."),
  p("Tree count was chosen on the rolling folds (100–300 trees perform similarly; more trees overfit the small residual signal). Target encoding adds a small but consistent gain. Feature importance is led by coordinates and lane encodings, which is expected: the linear component already handles the market-wide effects, so the trees focus on location."),
  image("outputs/figures/holdout_daily.png", 6.0),
  caption("Figure 4. Daily average rate per mile on the Sep–Oct holdout, model trained on Jan–Aug only."),
  p("By month, holdout error is 1.23% in September and 1.53% in October: error grows with the forecast horizon, and October runs about 1% low because the trend estimated from eight months was slightly below the full-data estimate. The final model, trained on ten months, uses a trend of 0.60%/month, a quarter-end uplift of 4.1% and a market-index effect of 0.142 (log scale)."),

  h1("6. December chart"),
  p("The chart fixes the load (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb) and changes only the date. The chart inputs do not include market_index or quote_signal, so each December date uses the mean market_index and median quote_signal of the ~200 validation loads on that same date. These are features of the loads being scored, not labels."),
  image("scorer_results/candidate_december.png", 6.5),
  caption("Figure 5. Produced by the provided score.py."),
  bullet("**Weekly waves** come from the daily market index, which follows a weekly cycle in both the training data and the December validation loads."),
  bullet("**The upward drift**, from about $826 to $853, is the quarter-end surge seen in every previous quarter-end month, plus the small monthly trend."),
  bullet("**Sanity check:** recent Dry Van loads on this lane in training ranged $778–$865 (about $2.26 per mile), and the model's error on this lane's historical loads is about 1%. The December predictions of $825–$853 ($2.29–$2.37 per mile) are in line with that."),

  h1("7. Limitations and next steps"),
  bullet("**The trend is assumed to continue linearly into December.** It is stable across training windows, but a two-month extrapolation is the main source of risk. With live data, I would re-estimate it weekly."),
  bullet("**December has no labelled precedent.** The quarter-end ramp is learned from March, June and September; December may also have year-end or holiday effects that the training data cannot show."),
  bullet("**Corrupted labels in the scoring data.** If the final ground truth contains the same ~1.4% of corrupted rates, measured MAE will be higher (about $88 rather than $32 on the holdout). The model predicts the true market rate rather than trying to guess the corruption."),
  bullet("**Possible improvements:** prediction intervals via quantile LightGBM; a lagged or smoothed market-index feature; and monitoring for drift in the trend and quarter-end effect."),

  h1("8. Reproducing the results"),
  p("The repository README has full instructions. In short: pip install -r requirements.txt, then run src/validate.py, src/predict.py and score.py. The full pipeline runs in about three minutes."),
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
