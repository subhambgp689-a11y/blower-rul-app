"""Matplotlib charts for the plant early-warning app."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import plant_engine as E  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#008300", "#e34948"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
STATUS_COLOR = {"Inspect": RED, "Watch": YELLOW, "Healthy": GREEN}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "grid.linestyle": "-", "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "figure.dpi": 110, "savefig.dpi": 110,
})


def _datefmt(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _ref_line(ax, y, text, color=MUTED, ls=(0, (4, 3))):
    ax.axhline(y, color=color, lw=0.9, ls=ls, zorder=1)
    ax.annotate(text, xy=(1, y), xycoords=("axes fraction", "data"), xytext=(3, 0), textcoords="offset points",
                va="center", ha="left", fontsize=7.5, color=INK2)


def fleet_chart(fs: pd.DataFrame):
    """Horizontal bars: predicted RUL of each fan, coloured by status, most urgent at the top."""
    d = fs.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, max(2.8, 0.42 * len(d) + 1.0)))
    y = np.arange(len(d))
    ax.barh(y, d["pred_h"], height=0.55, color=[STATUS_COLOR[s] for s in d["status"]], zorder=2)
    for yi, (_, r) in zip(y, d.iterrows()):
        lab = E.fmt_rul(r["pred_h"]) + ("   (warning active)" if r["warning_now"] else "")
        ax.text(min(r["pred_h"], E.CAP) + 18, yi, lab, va="center", fontsize=8, color=INK,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0), zorder=3)
    ax.set_yticks(y, [f"{r.fan}" + ("  (old data)" if r.stale else "") for r in d.itertuples()])
    ax.set_xlim(0, E.CAP * 1.22)
    ax.set_xlabel("Predicted remaining useful life (running hours, capped at 1,500)")
    for x, t in ((E.INSPECT_H, "Inspect < 240 h"), (E.WATCH_H, "Watch < 1,350 h")):
        ax.axvline(x, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.text(x, len(d) - 0.35, t, fontsize=7.5, color=INK2, ha="center", va="bottom")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLOR[s]) for s in E.STATUS_ORDER]
    ax.legend(handles, E.STATUS_ORDER, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.16 - 0.4 / len(d)))
    fig.tight_layout()
    return fig


def trend_chart(g: pd.DataFrame, events=(), show_true: bool = False):
    """Three stacked panels: vibration, temperature, predicted RUL with warning flags."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.25, 0.7, 1.25]})
    a1, a2, a3 = axs
    a1.plot(g["t"], g["vel_h"], color=BLUE, lw=1.3, label="V_H (horizontal)")
    a1.plot(g["t"], g["vel_v"], color=ORANGE, lw=1.3, label="V_V (vertical)")
    top = max(np.nanpercentile(g[["vel_h", "vel_v"]].max(axis=1), 99.5) * 1.15, 3.0) if len(g) else 5
    a1.set_ylim(0, top)
    for v, t in zip(E.ISO_LIMITS, ("A/B", "B/C", "C/D")):
        if v < top:
            _ref_line(a1, v, f"ISO {t}")
    a1.set_ylabel("mm/s")
    a1.set_title("Vibration velocity (12-hour running snapshots)")

    a2.plot(g["t"], g["temp"], color=AQUA, lw=1.3)
    a2.set_ylabel("°C")
    a2.set_title("Temperature")

    a3.plot(g["t"], g["pred"], color=BLUE, lw=1.6, label="Predicted RUL")
    if show_true and "RUL_h" in g and g["RUL_h"].notna().any():
        a3.plot(g["t"], np.minimum(g["RUL_h"], E.CAP), color=INK, lw=1.2, ls=(0, (5, 3)),
                label="RUL from inferred events (capped)")
    al = g[g["alarm"]]
    if len(al):
        a3.scatter(al["t"], al["pred"], s=26, color=RED, edgecolor=SURFACE, linewidth=1.2, zorder=4,
                   label="Early warning (2 snapshots < 120 h)")
    _ref_line(a3, E.WARN_H, "120 h", color=RED)
    _ref_line(a3, E.INSPECT_H, "240 h")
    a3.set_ylim(0, E.CAP * 1.07)
    a3.set_ylabel("running hours")
    a3.set_title("Predicted remaining useful life")
    a3.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    for t in events:
        for ax in axs:
            ax.axvline(t, color=RED, lw=1.0, ls=(0, (2, 2)), zorder=1)
    h, lab = a1.get_legend_handles_labels()
    if len(events):
        h.append(plt.Line2D([], [], color=RED, lw=1.0, ls=(0, (2, 2))))
        lab.append("Inferred failure/repair event")
    a1.legend(h, lab, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    _datefmt(a3)
    fig.tight_layout(h_pad=0.8)
    return fig


def event_chart(g: pd.DataFrame, t_event, strength: str, lead_h=None):
    """80 running days before and 15 after one inferred event."""
    g = g.sort_values("t").reset_index(drop=True)
    i = int((g["t"] - t_event).abs().idxmin())
    x = g.iloc[max(0, i - 160):i + 31]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    a1.plot(x["t"], x["vel_h"], color=BLUE, lw=1.3, label="V_H")
    a1.plot(x["t"], x["vel_v"], color=ORANGE, lw=1.3, label="V_V")
    a1.set_ylabel("mm/s")
    a1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    a1.set_title(f"Inferred event {pd.Timestamp(t_event):%d %b %Y} ({strength})")
    a2.plot(x["t"], x["pred"], color=BLUE, lw=1.6, label="Predicted RUL")
    if "RUL_h" in x:
        a2.plot(x["t"], np.minimum(x["RUL_h"], E.CAP), color=INK, lw=1.2, ls=(0, (5, 3)), label="RUL from event")
    al = x[x["alarm"]]
    if len(al):
        a2.scatter(al["t"], al["pred"], s=26, color=RED, edgecolor=SURFACE, linewidth=1.2, zorder=4, label="Warning")
    _ref_line(a2, E.WARN_H, "120 h", color=RED)
    a2.set_ylim(0, E.CAP * 1.07)
    a2.set_ylabel("running hours")
    a2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    for ax in (a1, a2):
        ax.axvline(t_event, color=RED, lw=1.0, ls=(0, (2, 2)))
    if lead_h is not None and np.isfinite(lead_h):
        a2.set_title(f"First warning {lead_h:,.0f} running hours ({lead_h / 24:.0f} running days) before the event")
    else:
        a2.set_title("No warning in the last 30 running days before the event")
    _datefmt(a2)
    fig.tight_layout(h_pad=0.8)
    return fig


def roc_chart(fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.plot(fpr, tpr, color=BLUE, lw=2)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"Fails within 10 running days?  AUC {auc:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    return fig


def ratio_chart(ratio):
    d = pd.DataFrame(ratio).sort_values("h", ascending=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = d["h"] / 24
    ax.plot(x, d["real"], color=BLUE, marker="o", ms=6, mec=SURFACE, mew=1.5, label="Plant fans (9 events)")
    ax.plot(x, d["syn"], color=ORANGE, marker="o", ms=6, mec=SURFACE, mew=1.5,
            label="Synthetic fleet (observable modes)")
    ax.invert_xaxis()
    ax.set_xlabel("Running days before failure")
    ax.set_ylabel("V_H: last 2 days ÷ cycle start")
    ax.set_title("How fast vibration rises before failure")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def importance_chart(imp: pd.Series):
    d = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.barh(range(len(d)), d.values, height=0.6, color=BLUE)
    ax.set_yticks(range(len(d)), [E.FEATURE_TEXT[f].split(" (")[0] for f in d.index], fontsize=8)
    for i, v in enumerate(d.values):
        ax.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, d.max() * 1.18)
    fig.tight_layout()
    return fig
