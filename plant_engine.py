"""
Plant RC-fan early-warning engine (Chapter 5 of the dissertation).

Pipeline: plant export (one row per axis) -> cleaned wide records -> 12-running-hour snapshots ->
trailing-reference features (v2) -> Extra Trees trained on the observable failure modes of the synthetic fleet ->
predicted RUL, warning rule and status for each fan.

Only the three channels the plant records are used: motor drive-end vibration velocity H and V, and temperature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, roc_curve

CAP = 1500.0          # RUL cap (running hours)
SNAP_H = 12.0         # running hours per snapshot (same as the synthetic fleet)
RUN_THR = 0.15        # mm/s: a sample counts as running when max(V_H, V_V) is above this
GAP_CAP_H = 1.1       # a single sample never counts for more than 1.1 running hours (bridges logger gaps)
WARN_H = 120.0        # warning rule: predicted RUL below 120 h ...
WARN_N = 2            # ... on 2 consecutive snapshots
INSPECT_H = 240.0     # status rules (as in the report): < 240 h inspect, < 1,350 h watch
WATCH_H = 1350.0
ISO_LIMITS = (1.4, 2.8, 4.5)   # ISO 10816-3 zone limits (mm/s) for the A/B, B/C, C/D boundaries
STATUS_ORDER = ["Inspect", "Watch", "Healthy"]
STATUS_TEXT = {
    "Inspect": "Predicted RUL below 240 running hours (about 10 running days), or the early warning is active",
    "Watch": "Predicted RUL below 1,350 running hours, or vibration in ISO zone D",
    "Healthy": "No rise in vibration relative to the fan's own recent level",
}
STATUS_ACTION = {"Inspect": "Inspect the fan and plan the intervention now",
                 "Watch": "Check the trend more often and confirm at the next round",
                 "Healthy": "Normal monitoring"}

# failure / repair events inferred from the plant record and accepted after visual review (Chapter 5.3)
EVENTS = [
    ("SQF4_RC_fan_Motor", "2024-10-12 14:53:01.200", "moderate"),
    ("SQF5_RC_fan_Motor", "2025-02-24 19:38:07.680", "moderate"),
    ("SQF5_RC_fan_Motor", "2025-12-21 08:39:53.670", "clear"),
    ("SQF6_RC_fan_Motor", "2026-04-30 12:14:30.600", "moderate"),
    ("SQF7_RC_fan_Motor", "2026-03-29 09:07:40.627", "clear"),
    ("SQF8_RC_fan_Motor", "2026-01-08 03:24:39.747", "moderate"),
    ("SQF9_RC_fan_Motor", "2026-02-02 03:44:10.993", "abrupt"),
    ("SQF9_RC_fan_Motor", "2026-04-26 00:59:34.220", "clear"),
    ("SQF10_RC_fan_Motor", "2025-10-15 09:06:06.063", "clear"),
]

FEATURES = ["rh_m4", "rv_m4", "rmax_m4", "rh_slope", "rv_slope", "rh_cummax14", "hv_rel", "temp_rel",
            "vmax_abs", "vmax_std7"]
FEATURE_TEXT = {
    "rh_m4": "V_H, last 2 days vs. its level 14–60 days earlier (log ratio)",
    "rv_m4": "V_V, last 2 days vs. its level 14–60 days earlier (log ratio)",
    "rmax_m4": "max(V_H, V_V), last 2 days vs. 14–60 days earlier (log ratio)",
    "rh_slope": "Change of the V_H ratio over the last 3 days",
    "rv_slope": "Change of the V_V ratio over the last 3 days",
    "rh_cummax14": "Highest V_H ratio in the last 7 days",
    "hv_rel": "Change of the H/V balance vs. 14–60 days earlier",
    "temp_rel": "Temperature, last 2 days minus 14–60 days earlier (°C)",
    "vmax_abs": "max(V_H, V_V), 2-day mean (mm/s)",
    "vmax_std7": "Variability of max(V_H, V_V) over the last 7 days (mm/s)",
}


def short_name(m: str) -> str:
    """'SQF10_RC_fan_Motor' -> 'SQF10'; other names unchanged."""
    m = str(m)
    return m.split("_RC_fan")[0] if "_RC_fan" in m else m


def iso_zone(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    return "ABCD"[int(np.digitize(v, ISO_LIMITS))]


def fmt_rul(h: float) -> str:
    if not np.isfinite(h):
        return "–"
    if h >= CAP - 1:
        return "≥ 1,500 h"
    return f"{h:,.0f} h"


# ------------------------------------------------------------------------------------------ input handling
EXPORT_COLS = {"DATE_TIME", "Equipment_Name", "Velocity", "Axis"}
WIDE_ALIASES = {
    "machine": ["machine", "equipment_name", "machine_id", "equipment", "fan"],
    "timestamp": ["timestamp", "date_time", "datetime", "time", "t"],
    "vel_h": ["vel_h", "v_h", "velocity_h", "vh", "motor_de_vel_h_mms"],
    "vel_v": ["vel_v", "v_v", "velocity_v", "vv", "motor_de_vel_v_mms"],
    "temp": ["temp", "temperature", "temp_c", "motor_de_temp_c"],
}


def detect_format(df: pd.DataFrame) -> str | None:
    cols = {c.strip() for c in df.columns}
    if EXPORT_COLS <= cols:
        return "export"
    low = {c.strip().lower() for c in df.columns}
    if all(any(a in low for a in al) for al in WIDE_ALIASES.values()):
        return "wide"
    return None


def format_help() -> str:
    return ("**Option 1: the plant export** (same layout as `All_Machines_combined.csv`): columns `DATE_TIME`, "
            "`Equipment_Name`, `Velocity`, `Temperature` and `Axis` (`H` or `V`), one row per axis. Other columns are "
            "ignored.\n\n**Option 2: one row per reading**: columns `machine`, `timestamp`, `vel_h`, `vel_v` (mm/s) and "
            "`temp` (°C).\n\nReadings can be at any interval (e.g. every 15 min). The app keeps only running periods "
            "and turns them into 12-running-hour snapshots. At least 60 running days per fan are recommended, "
            "because the features compare the last 2 days with the level 14–60 days earlier.")


def to_wide(df: pd.DataFrame, fmt: str) -> tuple[pd.DataFrame, dict]:
    """Return one row per machine and timestamp with vel_h, vel_v, temp; plus an audit dictionary."""
    audit = {"rows_in": int(len(df))}
    df = df.rename(columns=lambda c: c.strip())
    if fmt == "export":
        d = df[["DATE_TIME", "Equipment_Name", "Velocity", "Axis"] +
               (["Temperature"] if "Temperature" in df.columns else [])].copy()
        if "Temperature" not in d:
            d["Temperature"] = np.nan
        d["Equipment_Name"] = d["Equipment_Name"].astype(str).str.strip()
        d["Axis"] = d["Axis"].astype(str).str.strip().str.upper()
        d["DATE_TIME"] = pd.to_datetime(d["DATE_TIME"], errors="coerce")
        d["Velocity"] = pd.to_numeric(d["Velocity"], errors="coerce")
        d["Temperature"] = pd.to_numeric(d["Temperature"], errors="coerce")
        n0 = len(d)
        d = d.drop_duplicates()
        audit["duplicates"] = int(n0 - len(d))
        d = d.dropna(subset=["DATE_TIME"])
        d = d[d["Axis"].isin(["H", "V"])]
        w = (d.pivot_table(index=["Equipment_Name", "DATE_TIME"], columns="Axis", values="Velocity", aggfunc="mean")
             .join(d.groupby(["Equipment_Name", "DATE_TIME"])["Temperature"].mean()).reset_index()
             .rename(columns={"Equipment_Name": "machine", "DATE_TIME": "timestamp", "H": "vel_h", "V": "vel_v",
                              "Temperature": "temp"}))
        for c in ("vel_h", "vel_v"):
            if c not in w:
                w[c] = np.nan
    else:
        low = {c.lower(): c for c in df.columns}
        pick = {k: low[next(a for a in al if a in low)] for k, al in WIDE_ALIASES.items()}
        w = df[[pick[k] for k in WIDE_ALIASES]].copy()
        w.columns = list(WIDE_ALIASES)
        w["machine"] = w["machine"].astype(str).str.strip()
        w["timestamp"] = pd.to_datetime(w["timestamp"], errors="coerce")
        for c in ("vel_h", "vel_v", "temp"):
            w[c] = pd.to_numeric(w[c], errors="coerce")
        n0 = len(w)
        w = w.dropna(subset=["timestamp"]).drop_duplicates()
        audit["duplicates"] = int(n0 - len(w))
    w = w[["machine", "timestamp", "vel_h", "vel_v", "temp"]]
    audit["timestamps"] = int(len(w))
    drop = (w["vel_h"].fillna(0) <= 0) & (w["vel_v"].fillna(0) <= 0)
    audit["dropouts"] = int(drop.sum())
    w = w[~drop].copy()
    audit["negative_velocity"] = int(((w["vel_h"] < 0) | (w["vel_v"] < 0)).sum())
    w.loc[w["vel_h"] < 0, "vel_h"] = np.nan
    w.loc[w["vel_v"] < 0, "vel_v"] = np.nan
    audit["temp_le0"] = int((w["temp"] <= 0).sum())
    w.loc[w["temp"] <= 0, "temp"] = np.nan
    w = w.sort_values(["machine", "timestamp"]).reset_index(drop=True)
    return w, audit


def to_snapshots(w: pd.DataFrame, audit: dict | None = None) -> pd.DataFrame:
    """Median of every 12 running hours, per machine. Stops do not age the fan."""
    audit = audit if audit is not None else {}
    w = w.copy()
    w["vmax"] = w[["vel_h", "vel_v"]].max(axis=1)
    audit["running_pct"] = float((w["vmax"] > RUN_THR).mean() * 100) if len(w) else 0.0
    r = w[w["vmax"] > RUN_THR].copy()
    if r.empty:
        return pd.DataFrame(columns=["machine", "snap", "t", "op_h", "vel_h", "vel_v", "temp", "n"])
    r["dt_h"] = np.minimum(r.groupby("machine")["timestamp"].diff(-1).dt.total_seconds().abs() / 3600,
                           GAP_CAP_H).fillna(0.25)
    r["run_h"] = r.groupby("machine")["dt_h"].cumsum()
    r["snap"] = (r["run_h"] // SNAP_H).astype(int)
    S = (r.groupby(["machine", "snap"])
         .agg(t=("timestamp", "first"), vel_h=("vel_h", "median"), vel_v=("vel_v", "median"),
              temp=("temp", "median"), n=("vmax", "size")).reset_index())
    S["op_h"] = S["snap"] * SNAP_H
    audit["snapshots"] = int(len(S))
    return S


def split_at_events(S: pd.DataFrame, events=EVENTS) -> pd.DataFrame:
    """Give each snapshot a life-cycle id; a new cycle starts after each known event of that machine."""
    S = S.sort_values(["machine", "t"]).reset_index(drop=True)
    S["cycle_id"] = S["machine"].map(short_name) + "#1"
    for m, g in S.groupby("machine"):
        ev = sorted(pd.Timestamp(d) for mm, d, _ in events if mm == m)
        c = np.ones(len(g), int)
        for d in ev:
            c += (g["t"] > d).values.astype(int)
        S.loc[g.index, "cycle_id"] = [f"{short_name(m)}#{k}" for k in c]
    return S


# ------------------------------------------------------------------------------------------ features
def _clean(df: pd.DataFrame, cols, key):
    df = df.sort_values([key, "snap"]).reset_index(drop=True)
    df[cols] = df.groupby(key)[cols].transform(lambda x: x.interpolate(limit_direction="both"))
    df[cols] = df.groupby(key)[cols].rolling(3, center=True, min_periods=1).median().reset_index(level=0, drop=True)
    return df


def features(df: pd.DataFrame, key: str = "cycle_id"):
    """Trailing-reference features: the last 2 days compared with the fan's own level 14–60 days earlier.
    df needs columns key, snap, vel_h, vel_v, temp (one row per 12-h running snapshot)."""
    df = _clean(df, ["vel_h", "vel_v", "temp"], key)
    k = df[key]
    eps = 0.05
    vmax = df[["vel_h", "vel_v"]].max(axis=1)

    def ref(s):   # median over snapshots t-120 .. t-28 (60 to 14 running days ago); fallback: expanding median
        r = s.groupby(k).transform(lambda x: x.shift(28).rolling(92, min_periods=10).median())
        fb = s.groupby(k).transform(lambda x: x.expanding().median())
        return r.fillna(fb)

    def m(s, n):
        return s.groupby(k).transform(lambda x: x.rolling(n, min_periods=1).mean())

    F = pd.DataFrame(index=df.index)
    F["rh_m4"] = np.log((m(df["vel_h"], 4) + eps) / (ref(df["vel_h"]) + eps))
    F["rv_m4"] = np.log((m(df["vel_v"], 4) + eps) / (ref(df["vel_v"]) + eps))
    F["rmax_m4"] = np.log((m(vmax, 4) + eps) / (ref(vmax) + eps))
    F["rh_slope"] = F["rh_m4"].groupby(k).diff(6).fillna(0) / 6
    F["rv_slope"] = F["rv_m4"].groupby(k).diff(6).fillna(0) / 6
    F["rh_cummax14"] = F["rh_m4"].groupby(k).transform(lambda x: x.rolling(14, min_periods=1).max())
    hv = np.log((df["vel_h"] + eps) / (df["vel_v"] + eps))
    F["hv_rel"] = m(hv, 4) - ref(hv)
    F["temp_rel"] = (m(df["temp"], 4) - ref(df["temp"])).fillna(0)
    F["vmax_abs"] = m(vmax, 4)
    F["vmax_std7"] = vmax.groupby(k).transform(lambda x: x.rolling(14, min_periods=2).std()).fillna(0)
    return df, F[FEATURES]


# ------------------------------------------------------------------------------------------ model
def fit_model(train: pd.DataFrame, n_trees: int = 150) -> dict:
    """Extra Trees on the synthetic machines whose failure mode the three plant channels can see
    (motor-DE bearing, misalignment, impeller)."""
    s = train.rename(columns={"machine_id": "cycle_id", "cycle": "snap", "motor_de_vel_h_mms": "vel_h",
                              "motor_de_vel_v_mms": "vel_v", "motor_de_temp_c": "temp", "RUL_hours": "RUL_h"})
    s, X = features(s[["cycle_id", "snap", "vel_h", "vel_v", "temp", "RUL_h"]].copy(), "cycle_id")
    y = np.minimum(s["RUL_h"].values, CAP)
    mdl = ExtraTreesRegressor(n_estimators=n_trees, min_samples_leaf=5, max_features=0.7, n_jobs=-1, random_state=0)
    mdl.fit(X, y)
    imp = pd.Series(mdl.feature_importances_, index=FEATURES).sort_values(ascending=False)
    healthy = X[y >= CAP]
    ranges = pd.DataFrame({"p05": healthy.quantile(0.05), "p50": healthy.median(), "p95": healthy.quantile(0.95)})
    late = X[(s["RUL_h"].values <= 240)]
    ranges["late_median"] = late.median()
    return {"model": mdl, "n_trees": n_trees, "train_machines": int(s["cycle_id"].nunique()),
            "train_rows": int(len(s)), "importance": imp, "ranges": ranges,
            "modes": train.drop_duplicates("machine_id")["failure_mode"].value_counts().to_dict()
            if "failure_mode" in train else {}}


def predict(engine: dict, S: pd.DataFrame, key: str = "cycle_id") -> pd.DataFrame:
    """Predicted RUL and warning flags for every snapshot. `key` groups the snapshots into life-cycles."""
    cols = [c for c in S.columns if c not in ("vel_h", "vel_v", "temp")]
    d, X = features(S.copy(), key)
    d = d.reset_index(drop=True)
    d["pred"] = np.clip(engine["model"].predict(X), 0, CAP)
    for c in FEATURES:
        d[c] = X[c].values
    d["vmax"] = d[["vel_h", "vel_v"]].max(axis=1)
    d = d.sort_values(["machine", "t"]).reset_index(drop=True)
    d["below"] = d["pred"] < WARN_H
    d["alarm"] = d.groupby(key)["below"].transform(lambda x: x.rolling(WARN_N, min_periods=WARN_N).sum() >= WARN_N)
    d["alarm"] = d["alarm"].fillna(False).astype(bool)
    d["zone"] = [iso_zone(v) for v in d["vmax"]]
    _ = cols
    return d


def status_of(pred_h: float, alarm_now: bool, zone: str) -> str:
    if alarm_now or pred_h < INSPECT_H:
        return "Inspect"
    if pred_h < WATCH_H or zone == "D":
        return "Watch"
    return "Healthy"


def fleet_status(P: pd.DataFrame, stale_days: float = 14) -> pd.DataFrame:
    """Latest snapshot of each machine with its status, most urgent first."""
    end = P["t"].max()
    rows = []
    for m, g in P.groupby("machine"):
        g = g.sort_values("t")
        last = g.iloc[-1]
        recent = g[g["t"] >= last["t"] - pd.Timedelta(days=30)]
        first_alarm = g.loc[g["alarm"] & (g["t"] >= last["t"] - pd.Timedelta(days=30)), "t"]
        s = status_of(last["pred"], bool(last["alarm"]), last["zone"])
        rows.append(dict(
            machine=m, fan=short_name(m), status=s, last_snapshot=last["t"],
            stale=(end - last["t"]).total_seconds() / 86400 > stale_days,
            vel_h=last["vel_h"], vel_v=last["vel_v"], temp=last["temp"], zone=last["zone"],
            pred_h=float(last["pred"]), min_pred_30d=float(recent["pred"].min()),
            warning_now=bool(last["alarm"]), warnings_30d=int(recent["alarm"].sum()),
            first_warning_30d=first_alarm.iloc[0] if len(first_alarm) else pd.NaT,
            snapshots=int(len(g)), running_h=float(len(g) * SNAP_H)))
    out = pd.DataFrame(rows)
    out["_o"] = out["status"].map({s: i for i, s in enumerate(STATUS_ORDER)})
    return out.sort_values(["_o", "pred_h"]).drop(columns="_o").reset_index(drop=True)


# ------------------------------------------------------------------------------------------ evaluation (plant data with inferred events)
def evaluate(P: pd.DataFrame, key: str = "cycle_id") -> dict:
    """Early-warning metrics on snapshots with known RUL (inferred events), as in Chapter 5."""
    lab = P[P["RUL_h"].notna()].copy()
    y = lab["RUL_h"].values
    within = (y <= 240).astype(int)
    auc = roc_auc_score(within, -lab["pred"].values) if 0 < within.sum() < len(within) else np.nan
    fpr, tpr, _ = roc_curve(within, -lab["pred"].values) if np.isfinite(auc) else (np.array([0, 1]),) * 2 + (None,)
    ev = []
    for cid, g in lab[~lab["censored"]].groupby(key, sort=False):
        a = g[g["alarm"] & (g["RUL_h"] <= 720)]
        ev.append(dict(cycle=cid, machine=g["machine"].iloc[0], event_time=g["t"].iloc[-1],
                       strength=g["event_strength"].iloc[0], caught=bool(len(a)),
                       lead_h=float(a["RUL_h"].iloc[0]) if len(a) else np.nan,
                       first_warning=a["t"].iloc[0] if len(a) else pd.NaT,
                       early=int((g["alarm"] & (g["RUL_h"] > 720)).sum()),
                       peak_vh=float(g.loc[g["RUL_h"] <= 240, "vel_h"].max())))
    ev = pd.DataFrame(ev).sort_values("event_time").reset_index(drop=True)
    healthy = lab["RUL_h"] >= 720
    return dict(auc=auc, fpr=fpr, tpr=tpr, events=ev, caught=int(ev["caught"].sum()), n_events=len(ev),
                lead_median=float(ev["lead_h"].median()) if ev["caught"].any() else np.nan,
                false_alarm_pct=float(lab.loc[healthy, "alarm"].mean() * 100),
                censored_alarms=int(lab.loc[lab["censored"], "alarm"].sum()),
                censored_snaps=int(lab["censored"].sum()), labelled=int(len(lab)))


def clopper_pearson(k: int, n: int, conf: float = 0.95):
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return lo, hi
