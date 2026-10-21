"""
Blower RUL Monitor: Streamlit front end for the dissertation
"A Machine Learning Framework for Predictive Health Monitoring and Remaining Useful Life Estimation
of Manufacturing Machinery Using Multi-Sensor Data" (Subham Mishra, BITS Pilani WILP, 2026).

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import charts
import rul_engine as E

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
ICON = {"Critical": "⛔", "Plan maintenance": "🔧", "Watch": "👁", "Healthy": "✅"}
DEMO = "Demo fleet (50 test blowers)"
UPLOAD = "Upload my own CSV"

st.set_page_config(page_title="Blower RUL Monitor", page_icon="⚙️", layout="wide")


# ------------------------------------------------------------------------------------------ cache
@st.cache_resource(show_spinner="Training the RUL model on 150 run-to-failure histories (first start only)…")
def get_engine():
    return E.fit_engine(pd.read_csv(DATA / "train.csv"))


@st.cache_data(show_spinner=False)
def load_demo():
    return (pd.read_csv(DATA / "test.csv"), pd.read_csv(DATA / "test_full_labels.csv"),
            pd.read_csv(DATA / "machines_metadata.csv"))


@st.cache_data(show_spinner="Computing RUL, intervals and health index…")
def run_predictions(_engine, df: pd.DataFrame, key: str):
    pred = E.predict(_engine, df)
    return pred, E.fleet_summary(pred)


@st.cache_data(show_spinner=False)
def report_results():
    return json.loads((ASSETS / "report_results.json").read_text())


@st.cache_data(show_spinner=False)
def example_readings():
    """Typical readings ~10–15 days before failure for each failure mode (from the training fleet)."""
    tr = pd.read_csv(DATA / "train.csv")
    meta = pd.read_csv(DATA / "machines_metadata.csv").set_index("machine_id")
    tr["fm"] = tr["machine_id"].map(meta["failure_mode"])
    tr["loc"] = tr["machine_id"].map(meta["fault_location"])
    late = tr[tr["RUL_hours"].between(240, 360)]
    ex = {"Healthy machine (commissioning values)": None}
    names = {"BEARING": "Example: motor DE bearing wear", "IMPELLER": "Example: impeller fouling / imbalance",
             "MISALIGNMENT": "Example: coupling misalignment", "WINDING": "Example: stator winding degradation"}
    for mode, name in names.items():
        sel = late[(late["fm"] == mode) & ((late["loc"] == "MOTOR_DE") if mode == "BEARING" else True)]
        ex[name] = sel[E.FEATURES].median().round(3).to_dict()
    return ex


def status_label(s: str) -> str:
    return f"{ICON[s]} {s}"


def show_table(df: pd.DataFrame):
    """Full-width table on both older and newer Streamlit versions."""
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True)


def show_image(path: str):
    try:
        st.image(path, width="stretch")
    except Exception:
        try:
            st.image(path, use_container_width=True)
        except Exception:
            st.image(path, use_column_width=True)


# ------------------------------------------------------------------------------------------ sidebar
st.sidebar.title("⚙️ Blower RUL Monitor")
st.sidebar.caption("Predictive health monitoring and remaining-useful-life estimation for motor-driven "
                   "centrifugal blowers (RC fan motors).")
page = st.sidebar.radio("Page", ["Fleet overview", "Machine detail", "Quick estimate", "Model & results"])
st.sidebar.divider()
source = st.sidebar.radio("Data source", [DEMO, UPLOAD])

engine = get_engine()
demo_df, demo_labels, demo_meta = load_demo()
labels = None
data, data_key = None, ""

if source == DEMO:
    data, data_key, labels = demo_df, "demo", demo_labels
else:
    up = st.sidebar.file_uploader("CSV with the 10 sensor features", type=["csv"])
    with st.sidebar.expander("Required columns"):
        st.markdown("`machine_id`, `cycle` **or** `timestamp`, and the ten features:\n\n"
                    + "\n".join(f"- `{c}` ({E.LABELS[c][1]})" for c in E.FEATURES))
        st.download_button("Download a sample file", (DATA / "sample_upload.csv").read_bytes(),
                           "sample_upload.csv", "text/csv")
    if up is not None:
        raw = pd.read_csv(up)
        problems = E.validate(raw)
        if problems:
            for p in problems:
                st.sidebar.error(p)
        else:
            if E.needs_resampling(raw):
                agg = st.sidebar.checkbox("Average readings into 12-hour snapshots (recommended)", value=True,
                                          help="The model was trained on one snapshot every 12 operating hours. "
                                               "Denser readings (e.g. every 10 min) should be averaged first.")
                if agg:
                    raw = E.resample_snapshots(raw)
            data, data_key = raw, f"upload-{up.name}-{up.size}-{len(raw)}"

st.sidebar.divider()
st.sidebar.caption("Model trained on a physics-informed **simulated** fleet calibrated to the plant RC fan. "
                   "See *Model & results*.")

needs_data = page in ("Fleet overview", "Machine detail")
if needs_data and data is None:
    st.title(page)
    st.info("Upload a CSV in the sidebar, or switch the data source to the demo fleet.")
    st.stop()
if data is not None:
    pred, summary = run_predictions(engine, data, data_key)


# ------------------------------------------------------------------------------------------ pages
def page_fleet():
    st.title("Fleet overview")
    st.caption("Latest prediction for every machine, most urgent first. RUL is capped at 1,500 h: "
               "'≥ 1,500 h' means no degradation is visible yet.")
    counts = summary["status"].value_counts()
    cols = st.columns(5)
    cols[0].metric("Machines", len(summary))
    for col, s in zip(cols[1:], E.STATUS_ORDER):
        col.metric(status_label(s), int(counts.get(s, 0)))

    short = summary[summary["snapshots"] < E.BASELINE_N]
    if len(short):
        st.warning(f"{len(short)} machine(s) have fewer than {E.BASELINE_N} snapshots, so their healthy baseline "
                   "is incomplete and their estimates are less reliable.")

    st.subheader("Most urgent machines")
    st.pyplot(charts.fleet_chart(summary, n=20), clear_figure=True)

    st.subheader("All machines")
    table = pd.DataFrame({
        "Machine": summary["machine_id"],
        "Status": summary["status"].map(status_label),
        "Predicted RUL": summary["pred_rul_h"].map(E.fmt_rul),
        "90 % interval (h)": [f"{lo:,.0f} – {hi:,.0f}" for lo, hi in zip(summary["lo_h"], summary["hi_h"])],
        "ISO zone": summary["iso_zone"],
        "Health alarm": np.where(summary["hi_alarm_ever"], "Yes", "No"),
        "Largest deviation → check": summary["hint"],
        "Snapshots": summary["snapshots"],
        "Operating hours": summary["op_hours"].round(0),
    })
    show_table(table)
    out = summary[["machine_id", "status", "pred_rul_h", "lo_h", "hi_h", "health_index", "hi_alarm_ever",
                   "iso_zone", "top_dev_sensor", "hint", "snapshots", "op_hours"]].round(1)
    st.download_button("Download fleet predictions (CSV)", out.to_csv(index=False).encode(), "fleet_rul.csv",
                       "text/csv")

    with st.expander("Status rules"):
        st.table(pd.DataFrame({"Status": [status_label(s) for s in E.STATUS_ORDER],
                               "Rule": [E.STATUS_TEXT[s] for s in E.STATUS_ORDER]}).set_index("Status"))
        st.caption("ISO zone uses the highest of the four velocity channels against ISO 10816-3 limits "
                   "(A/B 1.4, B/C 2.8, C/D 4.5 mm/s). The 'check' column names the sensor that has risen most "
                   "above its healthy level, mapped to the fault it usually indicates. It is an inspection hint, "
                   "not a diagnosis.")
    if labels is not None:
        ev = E.evaluate(engine, pred, labels)
        with st.expander("Accuracy on this demo fleet (true RUL known)"):
            c = st.columns(4)
            c[0].metric("R², all snapshots", f"{ev['r2']:.2f}")
            c[1].metric("R², last snapshot", f"{ev['r2_last']:.2f}")
            c[2].metric("RMSE", f"{ev['rmse']:.0f} h")
            c[3].metric("Interval coverage", f"{ev['coverage']:.0f} %", help="Target 90 %")
            acc, n = E.hint_accuracy(summary, demo_meta)
            st.caption(f"{ev['machines']} machines, {ev['rows']:,} snapshots never used in training. "
                       f"The largest-deviation hint matched the true failure mode for {acc:.0f} % of the "
                       f"{n} machines that are not healthy.")


def page_machine():
    st.title("Machine detail")
    ids = summary["machine_id"].tolist()
    st_of = dict(zip(summary["machine_id"], summary["status"]))
    mid = st.selectbox("Machine (most urgent first)", ids, format_func=lambda m: f"{m}  ·  {status_label(st_of[m])}")
    row = summary.set_index("machine_id").loc[mid]
    m = pred[pred["machine_id"] == mid].reset_index(drop=True)

    box = {"Critical": st.error, "Plan maintenance": st.warning, "Watch": st.info, "Healthy": st.success}
    msg = f"**{status_label(row['status'])}**: {E.STATUS_TEXT[row['status']]}."
    if row["status"] != "Healthy":
        msg += f" Largest deviation: *{E.LABELS[row['top_dev_sensor']][0]}* → check **{row['hint']}**."
    box[row["status"]](msg)

    c = st.columns(4)
    c[0].metric("Predicted RUL", E.fmt_rul(row["pred_rul_h"]))
    c[1].metric("90 % interval", f"{row['lo_h']:,.0f} – {row['hi_h']:,.0f} h")
    c[2].metric("Health index", f"{row['health_index']:.1f}", help=f"Alarm threshold {engine['hi_thr']:.1f}",
                delta="above threshold" if row["health_index"] > engine["hi_thr"] else "below threshold",
                delta_color="inverse" if row["health_index"] > engine["hi_thr"] else "off")
    c[3].metric("ISO 10816-3 zone", row["iso_zone"], help=f"Highest velocity {row['vel_max']:.2f} mm/s")

    true = None
    if labels is not None:
        lab = labels[labels["machine_id"] == mid].set_index("cycle")["RUL_hours"]
        if st.toggle("Show true RUL (demo data only)", value=True):
            true = m["cycle"].map(lab)
    if row["snapshots"] < E.BASELINE_N:
        st.warning(f"Only {row['snapshots']} snapshots: the healthy baseline normally uses the first "
                   f"{E.BASELINE_N} (10 days).")

    t1, t2, t3, t4 = st.tabs(["RUL trend", "Health index", "Sensors", "Latest readings"])
    with t1:
        st.pyplot(charts.rul_chart(m, true), clear_figure=True)
        st.caption("The blue band is the split-conformal 90 % interval. For planning, use its lower edge.")
    with t2:
        st.pyplot(charts.health_chart(m, engine["hi_thr"]), clear_figure=True)
        first = m.loc[m["hi_alarm"], "op_hours"]
        st.caption("Degradation-directed Mahalanobis distance of the nine sensor deviations from the healthy "
                   "fleet. " + (f"First alarm at {first.iloc[0]:,.0f} operating hours." if len(first)
                                else "No alarm so far."))
    with t3:
        sel = st.multiselect("Sensors", E.FEATURES, default=E.FEATURES, format_func=lambda c: E.LABELS[c][0])
        if sel:
            st.pyplot(charts.sensor_grid(m, sel), clear_figure=True)
        st.caption("Cleaned readings (gaps filled, spikes removed). Dashed grey line: median of the first 10 days. "
                   "Dotted lines: ISO 10816-3 zone limits.")
    with t4:
        base = m[E.SENSORS].iloc[:E.BASELINE_N].median()
        now = m[E.SENSORS].iloc[-1]
        z = np.clip((m[[f"dev_{s}" for s in E.SENSORS]].iloc[-1].values - engine["hi_mu"]) / engine["hi_sd"], 0, None)
        show_table(pd.DataFrame({
            "Sensor": [E.LABELS[s][0] for s in E.SENSORS], "Unit": [E.LABELS[s][1] for s in E.SENSORS],
            "First 10 days (median)": base.values.round(2), "Latest": now.values.round(2),
            "Change": (now - base).values.round(2), "Rise above healthy (σ)": z.round(1)}))
        st.caption("'Rise above healthy' is after load compensation and 7-day smoothing, in units of the healthy "
                   "fleet's standard deviation. Values above about 3 are unusual.")


def page_quick():
    st.title("Quick estimate")
    st.caption("For a single machine without a reading history: enter its readings from commissioning (healthy) "
               "and its readings now. The model compares the two after correcting for load.")
    st.warning("A single snapshot cannot be smoothed, so this estimate is noisier than the history-based "
               "estimate on the other pages. Use it for a first look, not for planning.", icon="⚠️")
    ex = example_readings()
    choice = st.selectbox("Prefill current readings with", list(ex))
    typical = {k: round(v, 3) for k, v in engine["typical_baseline"].items()}
    now_default = typical if ex[choice] is None else ex[choice]

    with st.form("quick"):
        cols = st.columns([2, 1, 1])
        cols[0].markdown("**Feature**")
        cols[1].markdown("**At commissioning**")
        cols[2].markdown("**Now**")
        base_vals, now_vals = {}, {}
        for f in E.FEATURES:
            c = st.columns([2, 1, 1])
            c[0].markdown(f"{E.LABELS[f][0]} ({E.LABELS[f][1]})")
            base_vals[f] = c[1].number_input(f"base {f}", value=float(typical[f]), format="%.3f",
                                             label_visibility="collapsed", key=f"b_{f}")
            now_vals[f] = c[2].number_input(f"now {f}", value=float(now_default[f]), format="%.3f",
                                            label_visibility="collapsed", key=f"n_{f}_{choice}")
        go = st.form_submit_button("Estimate RUL", type="primary")

    if go:
        res = E.quick_estimate(engine, now_vals, base_vals)
        c = st.columns(4)
        c[0].metric("Estimated RUL", E.fmt_rul(res["pred"]))
        c[1].metric("90 % interval", f"{res['lo']:,.0f} – {res['hi']:,.0f} h")
        c[2].metric("Health index", f"{res['health_index']:.1f}",
                    delta="above threshold" if res["hi_above"] else "below threshold",
                    delta_color="inverse" if res["hi_above"] else "off", help=f"Threshold {engine['hi_thr']:.1f}")
        c[3].metric("ISO 10816-3 zone", E.iso_zone(res["vel_max"]), help=f"Highest velocity {res['vel_max']:.2f} mm/s")
        st.pyplot(charts.quick_bar(res), clear_figure=True)
        z = dict(zip(E.SENSORS, np.clip((np.array([res["dev"][s] for s in E.SENSORS]) - engine["hi_mu"])
                                         / engine["hi_sd"], 0, None)))
        st.subheader("Which sensors have changed")
        st.pyplot(charts.deviation_bars(z, res["top"]), clear_figure=True)
        if res["top_z"] > 3:
            st.info(f"Largest rise: **{E.LABELS[res['top']][0]}**. This usually points to "
                    f"**{E.FAULT_HINT[res['top']]}**. It is an inspection hint, not a diagnosis.")
        else:
            st.success("No sensor has risen clearly above its healthy level.")


def page_model():
    st.title("Model & results")
    R = report_results()
    st.markdown(
        "The app implements the framework of the M.Tech dissertation *A Machine Learning Framework for Predictive "
        "Health Monitoring and Remaining Useful Life Estimation of Manufacturing Machinery Using Multi-Sensor Data* "
        "(Subham Mishra, BITS Pilani WILP; carried out at Tata Motors, Jamshedpur).")
    st.warning("**The model is trained on simulated data.** It learned from 150 run-to-failure histories of a "
               "physics-informed simulated blower fleet, calibrated to the plant RC fan, because the plant data "
               "contain only one recorded failure. Validate it on real machines before using it for maintenance "
               "decisions.", icon="⚠️")

    demo_pred, demo_sum = run_predictions(engine, demo_df, "demo")
    ev = E.evaluate(engine, demo_pred, demo_labels)
    st.subheader("Deployed model on 50 unseen test machines")
    c = st.columns(5)
    c[0].metric("R², all snapshots", f"{ev['r2']:.2f}")
    c[1].metric("R², last snapshot", f"{ev['r2_last']:.2f}")
    c[2].metric("RMSE", f"{ev['rmse']:.0f} h")
    c[3].metric("MAE", f"{ev['mae']:.0f} h")
    c[4].metric("90 % interval coverage", f"{ev['coverage']:.0f} %")
    st.caption(f"Extra Trees with {engine['n_trees']} trees (the report used 300; test accuracy is the same to "
               f"three decimals), trained on {engine['train_machines']} machines / {engine['train_rows']:,} snapshots "
               "each time the app starts. RUL is capped at 1,500 h.")

    t1, t2, t3, t4 = st.tabs(["Model benchmark", "Feature importance", "Early warning", "Method"])
    with t1:
        b = pd.DataFrame(R["benchmark"]).sort_values("cv_rmse_deg")
        b.insert(0, "Selected", np.where(b["model"] == R["final_model"], "✓", ""))
        b = b.rename(columns={"model": "Model", "cv_r2": "CV R²", "cv_rmse_deg": "CV RMSE, degradation zone (h)",
                              "test_r2": "Test R²", "test_rmse": "Test RMSE (h)", "test_r2_last": "Test R², last",
                              "test_phm": "Test PHM score"})
        show_table(b.round(2))
        st.caption("From the final report. 5-fold GroupKFold by machine on 150 training machines; the test set is 50 "
                   "machines never used for fitting, tuning or selection. The final model is the candidate with the "
                   "lowest cross-validated degradation-zone RMSE, a rule fixed before looking at test results.")
    with t2:
        st.pyplot(charts.importance_chart(R["importance"]), clear_figure=True)
        st.caption("Fan horizontal velocity, current unbalance and motor horizontal velocity carry most of the "
                   "information. Together they cover impeller, winding and bearing faults.")
    with t3:
        a = R["alarms"]
        rows = [[k.title() if k != "ALL" else "All machines", v["n"], f"{v['hi_detected_pct']:.0f} %",
                 f"{v['hi_median_lead_d']:.0f}", f"{v['iso_detected_pct']:.0f} %",
                 "–" if v["iso_median_lead_d"] is None else f"{v['iso_median_lead_d']:.0f}"] for k, v in a.items()]
        show_table(pd.DataFrame(rows, columns=["Failure mode", "Machines", "Health index: detected",
                                               "Health index: median warning (days)", "ISO zone C: detected",
                                               "ISO zone C: median warning (days)"]))
        st.caption("The health index warns about 110 days before failure and also catches winding faults, which do "
                   "not raise vibration. About 10 % of machines raised an alarm during their healthy period.")
    with t4:
        show_image(str(ASSETS / "framework.png"))
        st.markdown(
            "1. **Cleaning:** gaps are interpolated per machine; spikes are removed with a 3-point rolling median.\n"
            "2. **Load compensation:** each sensor is corrected for load (load and load², fitted on healthy data).\n"
            "3. **Healthy baseline:** each machine is compared with its own first 10 days.\n"
            "4. **Smoothing:** a trailing 7-day mean.\n"
            "5. **RUL:** Extra Trees on exactly ten features, RUL capped at 1,500 h.\n"
            "6. **Uncertainty:** split-conformal 90 % intervals, with a separate width for each band of predicted RUL.\n"
            "7. **Health index:** degradation-directed Mahalanobis distance (Ledoit-Wolf). An alarm needs three "
            "consecutive snapshots above the healthy 99th percentile.")


{"Fleet overview": page_fleet, "Machine detail": page_machine, "Quick estimate": page_quick,
 "Model & results": page_model}[page]()
