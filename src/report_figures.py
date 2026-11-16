"""Figures used in the written report. Run after validate.py and predict.py."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freight.data import clean_features, daily_market_index, flag_label_outliers, load_train, load_validation  # noqa: E402
from freight.model import MarketTrend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#b4b2a9", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 12, "figure.dpi": 160,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_comparison():
    res = pd.read_csv(ROOT / "outputs" / "validation_metrics.csv")
    primary = res["fold"].str.contains("test 2025-09-01")
    s = res[(res["labels"] == "clean") & primary].set_index("model")["MAPE_%"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = [BLUE if "final" in m else GRAY for m in s.index]
    ax.barh(s.index, s.values, color=colors, height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v + 0.05, y, f"{v:.2f}%", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Mean absolute % error on Sep–Oct holdout (trained on Jan–Aug)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Validation error by model (lower is better)")
    save(fig, "model_comparison.png")


def label_outliers(df):
    X = df.copy()
    m = MarketTrend().fit(X[~X.label_outlier], np.log(X.loc[~X.label_outlier, "posted_rate"]))
    ratio = X["posted_rate"] / np.exp(m.predict_log(X))
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    bins = np.logspace(np.log10(0.15), np.log10(6), 90)
    ax.hist(ratio[~X.label_outlier], bins=bins, color=BLUE, label="Kept (98.6%)")
    ax.hist(ratio[X.label_outlier], bins=bins, color=ORANGE, label="Flagged as corrupted (677 rows)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5], ["0.2x", "0.5x", "1x", "2x", "5x"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Posted rate ÷ expected rate for that distance, equipment and date")
    ax.set_ylabel("Loads (log scale)")
    ax.legend(frameon=False)
    ax.set_title("Corrupted rate labels sit far from the normal range")
    save(fig, "label_outliers.png")


def quarter_end(df):
    X = df[~df.label_outlier].copy()
    ld = np.log(X.distance)
    A = np.column_stack([np.ones(len(X)), ld, ld**2, X.equipment.eq("Reefer"), X.equipment.eq("Flatbed"), X.weight / 1e4,
                         X.market_index, (X.date - pd.Timestamp("2025-01-01")).dt.days / 30.4])
    b, *_ = np.linalg.lstsq(A.astype(float), np.log(X.posted_rate), rcond=None)
    X["res"] = (np.exp(np.log(X.posted_rate) - A.astype(float) @ b) - 1) * 100
    X["qe"] = X.date.dt.month.isin([3, 6, 9])
    g = X.groupby(["qe", X.date.dt.day]).res.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(g.index, g[True] - g[False].mean(), color=BLUE, lw=2, marker="o", ms=3)
    ax.plot(g.index, g[False] - g[False].mean(), color=ORANGE, lw=2, marker="o", ms=3)
    ax.text(31.4, (g[True] - g[False].mean()).iloc[-3:].mean(), "Quarter-end months\n(Mar, Jun, Sep)", color=INK, fontsize=9, va="center")
    ax.text(31.4, 0, "Other months", color=INK, fontsize=9, va="center")
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlim(0.5, 31.5)
    ax.set_xlabel("Day of month")
    ax.set_ylabel("Rate vs normal (%)")
    ax.set_title("Rates climb through every quarter-end month")
    save(fig, "quarter_end_ramp.png")


def holdout_daily():
    h = pd.read_csv(ROOT / "outputs" / "holdout_predictions.csv", parse_dates=["date"])
    h = h[~h.label_outlier]
    d = h.assign(a=h.posted_rate / h.distance, p=h.predicted_rate / h.distance).groupby("date")[["a", "p"]].mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(d.index, d.a, color=GRAY, lw=2, label="Actual")
    ax.plot(d.index, d.p, color=BLUE, lw=2, label="Predicted (model trained on Jan–Aug only)")
    ax.set_ylabel("Avg rate per mile ($)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Sep–Oct holdout: daily average rate per mile")
    fig.autofmt_xdate()
    save(fig, "holdout_daily.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_train()
    df = clean_features(raw, daily_market_index(raw, load_validation()))
    df["label_outlier"] = flag_label_outliers(df)
    model_comparison()
    label_outliers(df)
    quarter_end(df)
    holdout_daily()
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()
