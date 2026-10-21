"""
RUL prediction engine for motor-driven centrifugal blowers.

Implements the framework of the dissertation "A Machine Learning Framework for Predictive Health
Monitoring and Remaining Useful Life Estimation of Manufacturing Machinery Using Multi-Sensor Data":

  1. Cleaning           per-machine interpolation of gaps + 3-point rolling median (despiking)
  2. Load compensation  each sensor ~ load + load^2, fitted on healthy data (first 20 % of life)
  3. Healthy baseline   subtract the machine's own median over its first 20 snapshots (10 days)
  4. Smoothing          trailing 7-day (14-snapshot) rolling mean
  5. RUL model          Extra Trees on exactly 10 features, RUL capped at 1,500 h
  6. Uncertainty        split-conformal 90 % intervals, half-width per band of predicted RUL
  7. Health index       degradation-directed Mahalanobis distance (Ledoit-Wolf), alarm after
                        3 consecutive snapshots above the healthy 99th percentile

No Streamlit imports here, so the engine can be tested and reused on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["load_pct", "motor_de_vel_h_mms", "motor_de_vel_v_mms", "motor_de_vel_a_mms",
            "motor_de_acc_g", "motor_de_temp_c", "fan_de_vel_h_mms", "fan_de_temp_c",
            "winding_temp_c", "current_unbalance_pct"]
SENSORS = FEATURES[1:]
VELOCITY = ["motor_de_vel_h_mms", "motor_de_vel_v_mms", "motor_de_vel_a_mms", "fan_de_vel_h_mms"]

LABELS = {
    "load_pct": ("Load", "%"),
    "motor_de_vel_h_mms": ("Motor DE velocity, horizontal", "mm/s"),
    "motor_de_vel_v_mms": ("Motor DE velocity, vertical", "mm/s"),
    "motor_de_vel_a_mms": ("Motor DE velocity, axial", "mm/s"),
    "motor_de_acc_g": ("Motor DE acceleration", "g"),
    "motor_de_temp_c": ("Motor DE bearing temperature", "°C"),
    "fan_de_vel_h_mms": ("Fan DE velocity, horizontal", "mm/s"),
    "fan_de_temp_c": ("Fan DE bearing temperature", "°C"),
    "winding_temp_c": ("Winding temperature", "°C"),
    "current_unbalance_pct": ("Current unbalance", "%"),
}
# what a large deviation on each sensor usually points to (used only as an inspection hint)
FAULT_HINT = {
    "motor_de_vel_h_mms": "Motor DE bearing or rotor imbalance",
    "motor_de_vel_v_mms": "Misalignment / looseness or motor DE bearing",
    "motor_de_vel_a_mms": "Coupling misalignment",
    "motor_de_acc_g": "Motor DE bearing (early damage)",
    "motor_de_temp_c": "Motor DE bearing friction or misalignment",
    "fan_de_vel_h_mms": "Impeller fouling / imbalance or fan bearing",
    "fan_de_temp_c": "Fan DE bearing",
    "winding_temp_c": "Stator winding or overload",
    "current_unbalance_pct": "Stator winding insulation",
}

CAP = 1500.0           # hours: piece-wise linear RUL cap
SNAPSHOT_H = 12.0      # hours between snapshots in the training data
BASELINE_N = 20        # snapshots defining the healthy commissioning baseline (10 days)
SMOOTH_N = 14          # 7-day trailing mean
CONF_BINS = np.array([0.0, 400.0, 800.0, 1200.0, 1450.0, np.inf])
CONF_ALPHA = 0.10
ISO_ZONES = [(1.4, "A"), (2.8, "B"), (4.5, "C"), (np.inf, "D")]  # ISO 10816-3 group 2, rigid foundation

STATUS_ORDER = ["Critical", "Plan maintenance", "Watch", "Healthy"]
STATUS_TEXT = {
    "Critical": "Predicted RUL below 10 days: inspect and plan the repair now",
    "Plan maintenance": "Predicted RUL below 30 days, or interval reaches below 10 days: schedule maintenance",
    "Watch": "Degradation detected: increase monitoring",
    "Healthy": "No degradation detected",
}


# ----------------------------------------------------------------------------------------- data
def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with an uploaded table (empty list = OK)."""
    problems = []
    missing = [c for c in ["machine_id"] + FEATURES if c not in df.columns]
    if missing:
        problems.append("Missing columns: " + ", ".join(missing))
    if "cycle" not in df.columns and "timestamp" not in df.columns:
        problems.append("Need a 'cycle' (snapshot number) or a 'timestamp' column to order the readings")
    return problems


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Order readings, create cycle/op_hours if missing and coerce numeric columns."""
    df = df.copy()
    df["machine_id"] = df["machine_id"].astype(str)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "cycle" not in df.columns:
        df = df.sort_values(["machine_id", "timestamp"])
        df["cycle"] = df.groupby("machine_id").cumcount() + 1
    for c in FEATURES + ["cycle"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["machine_id", "cycle"]).reset_index(drop=True)
    if "op_hours" not in df.columns:
        if "timestamp" in df.columns and df["timestamp"].notna().all():
            t0 = df.groupby("machine_id")["timestamp"].transform("min")
            df["op_hours"] = (df["timestamp"] - t0).dt.total_seconds() / 3600.0
        else:
            df["op_hours"] = (df["cycle"] - 1) * SNAPSHOT_H
    df["load_pct"] = df.groupby("machine_id")["load_pct"].transform(lambda x: x.interpolate(limit_direction="both"))
    return df


def needs_resampling(df: pd.DataFrame) -> bool:
    """True when timestamps are much denser than the 12-hour snapshots the model was trained on."""
    if "timestamp" not in df.columns:
        return False
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    step = ts.groupby(df["machine_id"].astype(str)).diff().dt.total_seconds().median() / 3600.0
    return bool(np.isfinite(step) and step < 0.75 * SNAPSHOT_H)


def resample_snapshots(df: pd.DataFrame, hours: float = SNAPSHOT_H) -> pd.DataFrame:
    """Average dense readings (e.g. every 10 min) into one snapshot per `hours` of time per machine."""
    df = df.copy()
    df["machine_id"] = df["machine_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    t0 = df.groupby("machine_id")["timestamp"].transform("min")
    df["_bin"] = ((df["timestamp"] - t0).dt.total_seconds() // (hours * 3600)).astype(int)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df.groupby(["machine_id", "_bin"], as_index=False).agg(
        timestamp=("timestamp", "min"), **{c: (c, "mean") for c in FEATURES})
    out["cycle"] = out.groupby("machine_id").cumcount() + 1
    out["op_hours"] = out["_bin"] * hours
    return out.drop(columns="_bin")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "cycle"]).reset_index(drop=True)
    df[SENSORS] = df.groupby("machine_id")[SENSORS].transform(lambda x: x.interpolate(limit_direction="both"))
    df[SENSORS] = (df.groupby("machine_id")[SENSORS].rolling(3, center=True, min_periods=1).median()
                   .reset_index(level=0, drop=True))
    return df


def _design(load: pd.Series | np.ndarray) -> np.ndarray:
    load = np.asarray(load, dtype=float)
    return np.column_stack([np.ones_like(load), load, load ** 2])


def _load_effect(engine: dict, load) -> pd.DataFrame:
    coef = engine["load_coef"]                                  # (3, n_sensors)
    return pd.DataFrame(_design(load) @ coef, columns=SENSORS)


def features(engine: dict, df_clean: pd.DataFrame, smooth: int = SMOOTH_N) -> pd.DataFrame:
    """Load-compensated, baseline-referenced, smoothed features (exactly the 10 model inputs)."""
    resid = df_clean[SENSORS].reset_index(drop=True) - _load_effect(engine, df_clean["load_pct"]).values
    g = df_clean["machine_id"].reset_index(drop=True)
    base = resid.groupby(g).transform(lambda x: x.iloc[:BASELINE_N].median())
    dev = resid - base
    if smooth > 1:
        dev = dev.groupby(g).transform(lambda x: x.rolling(smooth, min_periods=1).mean())
    out = pd.DataFrame({"load_pct": df_clean["load_pct"].values})
    for s in SENSORS:
        out[s] = dev[s].values
    return out[FEATURES]


# ----------------------------------------------------------------------------------------- fit
def fit_engine(train: pd.DataFrame, n_trees: int = 150, seed: int = 0, calib_machines: int = 30) -> dict:
    """Fit every component of the framework on run-to-failure training data (needs RUL_hours)."""
    train = clean(prepare(train))
    life = train.groupby("machine_id")["cycle"].transform("max")
    healthy = (train["cycle"] <= 0.2 * life).values

    X_h = _design(train.loc[healthy, "load_pct"])
    coef, *_ = np.linalg.lstsq(X_h, train.loc[healthy, SENSORS].values, rcond=None)
    engine = {"load_coef": coef}

    X = features(engine, train)
    y = train["RUL_hours"].clip(upper=CAP).values

    # split-conformal calibration on the last `calib_machines` machines
    ids = train["machine_id"].unique()
    is_cal = train["machine_id"].isin(set(ids[-calib_machines:])).values
    proper = ExtraTreesRegressor(n_estimators=n_trees, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    proper.fit(X[~is_cal], y[~is_cal])
    p_cal = np.clip(proper.predict(X[is_cal]), 0, CAP)
    res = np.abs(y[is_cal] - p_cal)
    band = np.digitize(p_cal, CONF_BINS) - 1
    half = np.zeros(len(CONF_BINS) - 1)
    for k in range(len(half)):
        r = np.sort(res[band == k])
        n = len(r)
        half[k] = r[min(n - 1, int(np.ceil((n + 1) * (1 - CONF_ALPHA))) - 1)] if n else CAP
    del proper

    model = ExtraTreesRegressor(n_estimators=n_trees, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    model.fit(X, y)

    # degradation-directed health index on healthy data
    Z = X[SENSORS].values[healthy]
    mu, sd = Z.mean(axis=0), Z.std(axis=0, ddof=1)
    zz = np.clip((Z - mu) / sd, 0, None)
    lw = LedoitWolf().fit(zz)
    centre = zz.mean(axis=0)
    d_h = np.sqrt(np.einsum("ij,jk,ik->i", zz - centre, lw.precision_, zz - centre))

    # commissioning reference values (median of first 20 snapshots, all machines) for the quick-estimate form
    first = train[train["cycle"] <= BASELINE_N]
    engine.update(model=model, conf_half=half, hi_mu=mu, hi_sd=sd, hi_centre=centre, hi_prec=lw.precision_,
                  hi_thr=float(np.percentile(d_h, 99)), typical_baseline=first[FEATURES].median().to_dict(),
                  n_trees=n_trees, train_machines=int(len(ids)), train_rows=int(len(train)))
    return engine


# ----------------------------------------------------------------------------------------- predict
def health_index(engine: dict, X: pd.DataFrame) -> np.ndarray:
    zz = np.clip((X[SENSORS].values - engine["hi_mu"]) / engine["hi_sd"], 0, None)
    d = zz - engine["hi_centre"]
    return np.sqrt(np.einsum("ij,jk,ik->i", d, engine["hi_prec"], d))


def iso_zone(v: float) -> str:
    if v is None or np.isnan(v):
        return "–"
    return next(z for lim, z in ISO_ZONES if v < lim)


def predict(engine: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise predictions for a table of readings from one or more machines."""
    data = clean(prepare(df))
    X = features(engine, data)
    pred = np.clip(engine["model"].predict(X), 0, CAP)
    band = np.digitize(pred, CONF_BINS) - 1
    half = engine["conf_half"][band]
    out = data.copy()
    out["pred_rul_h"] = pred
    out["lo_h"] = np.clip(pred - half, 0, CAP)
    out["hi_h"] = np.clip(pred + half, 0, CAP)
    out["health_index"] = health_index(engine, X)
    above = (out["health_index"] > engine["hi_thr"]).astype(int)
    new_block = (above != above.shift()) | (out["machine_id"] != out["machine_id"].shift())
    run = above.groupby(new_block.cumsum()).cumsum()
    out["hi_alarm"] = (run >= 3) & (above == 1)
    out["hi_alarm_ever"] = out.groupby("machine_id")["hi_alarm"].cummax()
    out["vel_max"] = out[VELOCITY].max(axis=1)
    zz = np.clip((X[SENSORS].values - engine["hi_mu"]) / engine["hi_sd"], 0, None)
    out["top_dev_sensor"] = np.array(SENSORS)[zz.argmax(axis=1)]
    out["top_dev_z"] = zz.max(axis=1)
    for s in SENSORS:
        out[f"dev_{s}"] = X[s].values
    return out


def status_of(row: pd.Series) -> str:
    if row["pred_rul_h"] < 240:
        return "Critical"
    if row["pred_rul_h"] < 720 or row["lo_h"] < 240:
        return "Plan maintenance"
    if row["pred_rul_h"] < 0.9 * CAP or bool(row["hi_alarm_ever"]):
        return "Watch"
    return "Healthy"


def fleet_summary(pred: pd.DataFrame) -> pd.DataFrame:
    last = pred.groupby("machine_id", sort=False).tail(1).copy()
    n = pred.groupby("machine_id").size()
    last["snapshots"] = last["machine_id"].map(n).values
    last["status"] = last.apply(status_of, axis=1)
    last["iso_zone"] = last["vel_max"].apply(iso_zone)
    last["hint"] = np.where(last["status"] == "Healthy", "–", last["top_dev_sensor"].map(FAULT_HINT))
    last["rank"] = last["status"].map({s: i for i, s in enumerate(STATUS_ORDER)})
    return last.sort_values(["rank", "pred_rul_h"]).drop(columns="rank").reset_index(drop=True)


def quick_estimate(engine: dict, current: dict, baseline: dict) -> dict:
    """Single-snapshot estimate from current readings and the machine's commissioning readings.
    No smoothing is possible, so this is noisier than a history-based prediction."""
    cur = pd.Series(current, dtype=float)
    base = pd.Series(baseline, dtype=float)
    eff_c = _load_effect(engine, [cur["load_pct"]]).iloc[0]
    eff_b = _load_effect(engine, [base["load_pct"]]).iloc[0]
    dev = (cur[SENSORS] - eff_c) - (base[SENSORS] - eff_b)
    X = pd.DataFrame([[cur["load_pct"], *dev.values]], columns=FEATURES)
    pred = float(np.clip(engine["model"].predict(X)[0], 0, CAP))
    half = float(engine["conf_half"][np.digitize([pred], CONF_BINS)[0] - 1])
    hi = float(health_index(engine, X)[0])
    zz = np.clip((dev.values - engine["hi_mu"]) / engine["hi_sd"], 0, None)
    return dict(pred=pred, lo=max(0.0, pred - half), hi=min(CAP, pred + half), health_index=hi,
                hi_above=hi > engine["hi_thr"], top=SENSORS[int(zz.argmax())], top_z=float(zz.max()),
                dev=dev.to_dict(), vel_max=float(cur[VELOCITY].max()))


HINT_MODE = {"motor_de_vel_h_mms": "BEARING", "motor_de_acc_g": "BEARING", "motor_de_temp_c": "BEARING",
             "fan_de_temp_c": "BEARING", "fan_de_vel_h_mms": "IMPELLER", "motor_de_vel_v_mms": "MISALIGNMENT",
             "motor_de_vel_a_mms": "MISALIGNMENT", "winding_temp_c": "WINDING", "current_unbalance_pct": "WINDING"}


def hint_accuracy(summary: pd.DataFrame, meta: pd.DataFrame) -> tuple[float, int]:
    """Share of non-healthy machines whose largest-deviation sensor points to the true failure mode."""
    d = summary[summary["status"] != "Healthy"]
    true = d["machine_id"].map(meta.set_index("machine_id")["failure_mode"])
    return float((d["top_dev_sensor"].map(HINT_MODE) == true).mean() * 100), int(len(d))


def evaluate(engine: dict, pred: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Accuracy of the deployed engine against true RUL labels (demo fleet)."""
    m = pred.merge(labels[["machine_id", "cycle", "RUL_hours"]], on=["machine_id", "cycle"], how="inner")
    y = m["RUL_hours"].clip(upper=CAP).values
    p = m["pred_rul_h"].values
    last = m.groupby("machine_id").tail(1).index
    cov = ((y >= m["lo_h"].values) & (y <= m["hi_h"].values))
    return dict(r2=r2_score(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))), mae=mean_absolute_error(y, p),
                r2_last=r2_score(y[last], p[last]), coverage=float(cov.mean() * 100),
                machines=int(m["machine_id"].nunique()), rows=int(len(m)))


def fmt_rul(h: float) -> str:
    if h >= 0.97 * CAP:
        return "≥ 1,500 h"
    return f"{h:,.0f} h ({h / 24:.0f} d)"
