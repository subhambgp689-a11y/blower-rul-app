"""Matplotlib charts for the RUL app (one consistent style, light theme)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rul_engine import CAP, LABELS, SENSORS, VELOCITY

BLUE, BLUE_BAND, INK, MUTED, GRID, GRAY = "#2a78d6", "#cfe0f6", "#0b0b0b", "#52514e", "#e6e5e0", "#b9b8b2"
STATUS_COLOR = {"Critical": "#d03b3b", "Plan maintenance": "#ec835a", "Watch": "#fab219", "Healthy": "#0ca30c"}
STATUS_MARK = {"Critical": "X", "Plan maintenance": "D", "Watch": "^", "Healthy": "o"}

plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 110, "savefig.bbox": "tight", "legend.frameon": False})


def _hline(ax, y, text, color, ls=":"):
    ax.axhline(y, color=color, lw=1, ls=ls, zorder=1)
    ax.annotate(text, xy=(1, y), xycoords=("axes fraction", "data"), xytext=(-2, 2), textcoords="offset points",
                ha="right", va="bottom", fontsize=7.5, color=MUTED)


def rul_chart(m: pd.DataFrame, true: pd.Series | None = None):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = m["op_hours"].values
    ax.fill_between(x, m["lo_h"], m["hi_h"], color=BLUE_BAND, lw=0, label="90 % interval")
    ax.plot(x, m["pred_rul_h"], color=BLUE, lw=2, label="Predicted RUL")
    if true is not None:
        ax.plot(x, np.minimum(true.values, CAP), color=INK, lw=1.4, ls="--", label="True RUL, capped (demo data)")
    _hline(ax, CAP, "cap 1,500 h", GRAY, "--")
    _hline(ax, 720, "30 days", STATUS_COLOR["Plan maintenance"])
    _hline(ax, 240, "10 days", STATUS_COLOR["Critical"])
    ax.set_ylim(0, CAP * 1.08)
    ax.set_xlabel("Operating hours"); ax.set_ylabel("RUL (h)")
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    return fig


def health_chart(m: pd.DataFrame, thr: float):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    x = m["op_hours"].values
    hi = m["health_index"].values
    alarm = m["hi_alarm"].values
    ymax = max(hi.max() * 1.1, thr * 2)
    ax.fill_between(x, 0, ymax, where=alarm, color=STATUS_COLOR["Watch"], alpha=0.18, lw=0, step="mid",
                    label="Alarm active (3 consecutive snapshots above threshold)")
    ax.plot(x, hi, color=BLUE, lw=1.6, label="Health index")
    _hline(ax, thr, "alarm threshold (healthy 99th percentile)", MUTED, "--")
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Operating hours"); ax.set_ylabel("Mahalanobis distance")
    ax.legend(loc="upper left", fontsize=8)
    return fig


def sensor_grid(m: pd.DataFrame, cols: list[str]):
    rows = int(np.ceil(len(cols) / 2))
    fig, axs = plt.subplots(rows, 2, figsize=(9, 1.9 * rows + 0.3), sharex=True, squeeze=False)
    x = m["op_hours"].values
    for ax, c in zip(axs.flat, cols):
        ax.plot(x, m[c].values, color=BLUE, lw=1)
        base = m[c].iloc[:20].median()
        ax.axhline(base, color=GRAY, lw=1, ls="--")
        if c in VELOCITY:
            for lim, z in [(1.4, "A/B"), (2.8, "B/C"), (4.5, "C/D")]:
                if lim < max(m[c].max() * 1.15, 3.0):
                    ax.axhline(lim, color=GRID if lim < 2.8 else "#d9c7b8", lw=0.9, ls=":")
                    ax.annotate(f"ISO {z}", xy=(0, lim), xycoords=("axes fraction", "data"), xytext=(2, 1),
                                textcoords="offset points", fontsize=6.5, color=MUTED)
        name, unit = LABELS[c]
        ax.set_title(f"{name} ({unit})", loc="left", fontsize=8.5)
    for ax in axs.flat[len(cols):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("Operating hours")
    fig.tight_layout(h_pad=0.6)
    return fig


def fleet_chart(summary: pd.DataFrame, n: int = 20):
    d = summary.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(0.3 * len(d) + 0.9, 1.8)))
    y = np.arange(len(d))
    for status in STATUS_COLOR:
        s = d["status"].values == status
        if s.any():
            ax.hlines(y[s], d["lo_h"].values[s], d["hi_h"].values[s], color=STATUS_COLOR[status], lw=3, alpha=0.35)
            ax.scatter(d["pred_rul_h"].values[s], y[s], color=STATUS_COLOR[status], marker=STATUS_MARK[status],
                       s=46, zorder=3, edgecolors="white", linewidths=0.8, label=status)
    ax.set_yticks(y); ax.set_yticklabels(d["machine_id"]); ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.axvline(720, color=STATUS_COLOR["Plan maintenance"], lw=1, ls=":")
    ax.axvline(240, color=STATUS_COLOR["Critical"], lw=1, ls=":")
    ax.set_xlim(0, CAP * 1.03)
    ax.set_xlabel("Predicted RUL (h) with 90 % interval    ·    dotted lines: 10 and 30 days")
    ax.legend(loc="lower right", fontsize=8, ncol=4, bbox_to_anchor=(1, 1.0))
    return fig


def quick_bar(res: dict):
    fig, ax = plt.subplots(figsize=(9, 1.25))
    ax.axvspan(0, 240, color=STATUS_COLOR["Critical"], alpha=0.10, lw=0)
    ax.axvspan(240, 720, color=STATUS_COLOR["Plan maintenance"], alpha=0.10, lw=0)
    ax.hlines(0, res["lo"], res["hi"], color=BLUE, lw=8, alpha=0.3)
    ax.plot(res["pred"], 0, marker="o", color=BLUE, ms=11, markeredgecolor="white")
    for x, t in [(120, "< 10 d"), (480, "10–30 d")]:
        ax.text(x, 0.32, t, ha="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, CAP * 1.02); ax.set_ylim(-0.5, 0.6)
    ax.set_yticks([]); ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
    ax.set_xlabel("RUL (h): dot = estimate, bar = 90 % interval")
    return fig


def deviation_bars(z: dict, top: str):
    s = pd.Series(z).reindex(SENSORS)[::-1]
    fig, ax = plt.subplots(figsize=(9, 3.0))
    colors = [BLUE if k == top else GRAY for k in s.index]
    ax.barh([LABELS[k][0] for k in s.index], s.values, color=colors, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rise above healthy level (in healthy standard deviations, ≥ 0)")
    return fig


def importance_chart(imp: dict):
    s = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.barh([LABELS[k][0] for k in s.index], s.values, color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in test R² when the feature is shuffled (permutation importance)")
    return fig
