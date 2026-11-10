"""
Plant view of the RUL Monitor: three-channel early-warning model, for Chapter 5 of the M.Tech dissertation
"A Machine Learning Framework for Predictive Health Monitoring and Remaining Useful Life Estimation
of Manufacturing Machinery Using Multi-Sensor Data" (Subham Mishra, BITS Pilani WILP, 2026).

The model uses only the three channels the plant records (motor DE vibration velocity H and V, temperature),
is trained on the synthetic fleet's observable failure modes, and is validated on nine failure events inferred
from the plant record of the RC fan motors SQF3-SQF10.

Imported by app.py; call render(source).
"""
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import plant_charts as charts
import plant_engine as E

ROOT = Path(__file__).parent
REQUIRED = ["synthetic_train_observable.csv", "plant_snapshots.csv", "sample_plant_export.csv",
            "validation_results.json"]
ICON = {"Inspect": "⛔", "Watch": "👁", "Healthy": "✅"}
PLANT = "Plant record (SQF3–SQF10, to 9 Oct 2026)"
UPLOAD = "Upload a newer export"
PAGES = ["Fleet status", "Fan detail", "Validation on plant data", "Method & data"]



def find_file(name: str):
    """Locate a data/asset file in data/ or assets/ beside app.py, beside app.py, or in any sub-folder."""
    for p in (ROOT / "data" / name, ROOT / "assets" / name, ROOT / name):
        if p.is_file():
            return p
    hits = sorted(p for p in ROOT.rglob(name) if ".git" not in p.parts and ".venv" not in p.parts)
    return hits[0] if hits else None


def _check_files():
    _missing = [n for n in REQUIRED if find_file(n) is None]
    if _missing:
        st.title("🌀 Plant Fan RUL Monitor")
        st.error("Some files the app needs were not found next to app.py: **" + ", ".join(_missing) + "**.")
        st.markdown("Put the CSV files in a `data` folder and `validation_results.json` in an `assets` folder beside "
                    "`app.py` (or directly beside it). On GitHub: **Add file → Upload files**, drag the missing files, "
                    "commit, then reboot the app from **Manage app**.")
        st.stop()


# ------------------------------------------------------------------------------------------ cached work
@st.cache_resource(show_spinner="Training the model on 91 synthetic run-to-failure histories (first start only)…")
def get_engine():
    return E.fit_model(pd.read_csv(find_file("synthetic_train_observable.csv")))


@st.cache_data(show_spinner=False)
def load_plant():
    P = pd.read_csv(find_file("plant_snapshots.csv"), parse_dates=["t"])
    P["censored"] = P["censored"].astype(bool)
    return P


@st.cache_data(show_spinner="Computing predicted RUL and warnings…")
def run(_engine, S: pd.DataFrame, key: str):
    P = E.predict(_engine, S, "cycle_id")
    return P, E.fleet_status(P)


@st.cache_data(show_spinner="Evaluating against the inferred events…")
def run_eval(_engine, key: str):
    P, _ = run(_engine, load_plant(), "plant")
    return E.evaluate(P)


@st.cache_data(show_spinner="Reading and cleaning the export…", max_entries=3)
def process_upload(data: bytes, name: str, include_other: bool):
    try:
        raw = pd.read_csv(io.BytesIO(data), low_memory=False)
    except Exception as exc:  # noqa: BLE001
        return None, {"error": f"Could not read {name} as CSV ({exc})."}, None
    fmt = E.detect_format(raw)
    if fmt is None:
        return None, {"error": "The columns were not recognised. Found: " + ", ".join(map(str, raw.columns[:15]))}, None
    w, audit = E.to_wide(raw, fmt)
    machines = sorted(w["machine"].unique())
    audit["machines_in_file"] = machines
    if not include_other:
        other = [m for m in machines if "fan" not in m.lower()]
        w = w[~w["machine"].isin(other)]
        audit["excluded"] = other
    S = E.to_snapshots(w, audit)
    if S.empty:
        return None, {**audit, "error": "No running periods found (max(V_H, V_V) never above 0.15 mm/s)."}, fmt
    S = E.split_at_events(S)
    return S, audit, fmt


@st.cache_data(show_spinner=False)
def stored_results():
    return json.loads(find_file("validation_results.json").read_text(encoding="utf-8"))


def show_table(df: pd.DataFrame, height=None):
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:  # older Streamlit
        st.dataframe(df, use_container_width=True, **kw)


def show_fig(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def label(s: str) -> str:
    return f"{ICON[s]} {s}"


def run_days(h: float) -> str:
    return "–" if not np.isfinite(h) else ("≥ 62" if h >= E.CAP - 1 else f"{h / 24:.0f}")


# ------------------------------------------------------------------------------------------ pages
def page_fleet():
    st.title("Fleet status")
    end = P["t"].max()
    st.caption(f"Latest snapshot in the data: **{end:%d %b %Y, %H:%M}**. Each fan's status uses its own latest "
               "12-running-hour snapshot; most urgent first.")
    counts = fs["status"].value_counts()
    c = st.columns(5)
    c[0].metric("Fans", len(fs))
    for col, s in zip(c[1:4], E.STATUS_ORDER):
        col.metric(label(s), int(counts.get(s, 0)))
    c[4].metric("Early warnings active", int(fs["warning_now"].sum()),
                help="Predicted RUL below 120 h on the last two snapshots")

    for _, r in fs[fs["status"] == "Inspect"].iterrows():
        st.error(f"**{r.fan}**: predicted RUL **{E.fmt_rul(r.pred_h)}** (about {run_days(r.pred_h)} running days), "
                 f"V_H {r.vel_h:.2f} mm/s, V_V {r.vel_v:.2f} mm/s (ISO zone {r.zone})."
                 + (" Early warning active." if r.warning_now else "")
                 + f" {E.STATUS_ACTION['Inspect']}.", icon="⛔")
    stale = fs[fs["stale"]]
    if len(stale):
        st.warning("No recent running data for: " + ", ".join(f"**{r.fan}** (last {r.last_snapshot:%d %b %Y})"
                                                              for r in stale.itertuples())
                   + ". The fan may be stopped or its sensor offline, so its status is out of date.")

    show_fig(charts.fleet_chart(fs))

    table = pd.DataFrame({
        "Fan": fs["fan"], "Status": fs["status"].map(label),
        "Predicted RUL": fs["pred_h"].map(E.fmt_rul),
        "≈ running days": fs["pred_h"].map(run_days),
        "Lowest RUL, last 30 days": fs["min_pred_30d"].map(E.fmt_rul),
        "Early warning": np.where(fs["warning_now"], "Active",
                                  np.where(fs["warnings_30d"] > 0, "In last 30 days", "No")),
        "V_H (mm/s)": fs["vel_h"].round(2), "V_V (mm/s)": fs["vel_v"].round(2), "Temp (°C)": fs["temp"].round(1),
        "ISO zone": fs["zone"], "Last snapshot": fs["last_snapshot"].dt.strftime("%Y-%m-%d %H:%M"),
        "Running hours on record": fs["running_h"].round(0).astype(int),
    })
    show_table(table)
    out = fs.drop(columns=["machine"]).copy()
    for col in ("last_snapshot", "first_warning_30d"):
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d %H:%M")
    st.download_button("Download fleet status (CSV)", out.round(2).to_csv(index=False).encode(), "fleet_status.csv",
                       "text/csv")

    with st.expander("Status rules and how to read the numbers"):
        show_table(pd.DataFrame({"Status": [label(s) for s in E.STATUS_ORDER],
                                 "Rule": [E.STATUS_TEXT[s] for s in E.STATUS_ORDER],
                                 "Action": [E.STATUS_ACTION[s] for s in E.STATUS_ORDER]}))
        st.markdown(
            "- **Predicted RUL** is in running hours. A fan that is stopped does not age, so calendar time to "
            "failure can be longer. **≥ 1,500 h** means no rise in vibration is visible.\n"
            "- The model compares the last 2 running days with the fan's own level 14–60 days earlier. A fan that "
            "has always run at a high level is therefore not flagged by the model, which is why the ISO zone is "
            "shown too (zone D alone makes a fan *Watch*).\n"
            "- **Early warning**: predicted RUL below 120 h on two consecutive snapshots. On the plant record it "
            "caught 6 of 9 inferred failures, a median of 234 running hours ahead, with no alarm on healthy fans.")
    if audit:
        with st.expander("What happened to the uploaded file"):
            st.json({k: v for k, v in audit.items() if k != "error"})


def page_fan():
    st.title("Fan detail")
    st_of = dict(zip(fs["machine"], fs["status"]))
    mid = st.selectbox("Fan (most urgent first)", fs["machine"].tolist(),
                       format_func=lambda m: f"{E.short_name(m)}  ·  {label(st_of[m])}")
    r = fs.set_index("machine").loc[mid]
    g = P[P["machine"] == mid].sort_values("t").reset_index(drop=True)
    events = [(pd.Timestamp(d), s) for m, d, s in E.EVENTS if m == mid]

    box = {"Inspect": st.error, "Watch": st.warning, "Healthy": st.success}[r["status"]]
    box(f"**{label(r['status'])}**: {E.STATUS_TEXT[r['status']]}. {E.STATUS_ACTION[r['status']]}."
        + (f" Last snapshot {r['last_snapshot']:%d %b %Y}: data may be out of date." if r["stale"] else ""))
    c = st.columns(5)
    c[0].metric("Predicted RUL", E.fmt_rul(r["pred_h"]), help="Running hours, capped at 1,500")
    c[1].metric("≈ running days", run_days(r["pred_h"]))
    c[2].metric("V_H / V_V", f"{r['vel_h']:.2f} / {r['vel_v']:.2f}", help="mm/s, latest snapshot")
    c[3].metric("ISO 10816-3 zone", r["zone"], help="From the higher of V_H and V_V")
    c[4].metric("Temperature", "–" if not np.isfinite(r["temp"]) else f"{r['temp']:.1f} °C")

    span = st.radio("Period", ["Last 90 days", "Last 12 months", "Whole record"], horizontal=True)
    days = {"Last 90 days": 90, "Last 12 months": 365, "Whole record": None}[span]
    gv = g if days is None else g[g["t"] >= g["t"].max() - pd.Timedelta(days=days)]
    ev_in = [t for t, _ in events if gv["t"].min() <= t <= gv["t"].max()] if len(gv) else []

    tabs = st.tabs(["Trend", "Inferred events", "Latest snapshots", "What the model sees"])
    with tabs[0]:
        show_true = False
        if has_labels and g["RUL_h"].notna().any():
            show_true = st.toggle("Show RUL implied by the inferred events", value=False)
        show_fig(charts.trend_chart(gv, ev_in, show_true))
        st.caption("Red dots: early warning (predicted RUL below 120 h on two consecutive snapshots). Dashed red "
                   "lines: inferred failure/repair events. Snapshots are medians of 12 running hours, so periods "
                   "when the fan was stopped are skipped.")
    with tabs[1]:
        if not events:
            st.info("No failure/repair event has been inferred for this fan.")
        else:
            if ev_all is not None:
                e = ev_all["events"][ev_all["events"]["machine"] == mid].reset_index(drop=True)
                show_table(pd.DataFrame({
                    "Event (last snapshot before the drop)": e["event_time"].dt.strftime("%Y-%m-%d %H:%M"),
                    "Grade": e["strength"], "Caught": np.where(e["caught"], "Yes", "No"),
                    "First warning": e["first_warning"].dt.strftime("%Y-%m-%d").fillna("–"),
                    "Warning lead (running h)": e["lead_h"].map(lambda v: "–" if not np.isfinite(v) else f"{v:.0f}"),
                    "Peak V_H, last 10 days (mm/s)": e["peak_vh"].round(2),
                    "Warnings > 30 days early": e["early"]}))
                pick = st.selectbox("Show event", range(len(e)),
                                    format_func=lambda i: f"{e['event_time'][i]:%d %b %Y} ({e['strength'][i]})")
                show_fig(charts.event_chart(g, e["event_time"][pick], e["strength"][pick], e["lead_h"][pick]))
            else:
                st.markdown("Known events for this fan: " + ", ".join(f"{t:%d %b %Y} ({s})" for t, s in events))
                t0, s0 = events[-1]
                if g["t"].min() <= t0 <= g["t"].max():
                    show_fig(charts.event_chart(g, t0, s0))
            st.caption("Events were inferred from the vibration record (a rise to at least 2 mm/s followed by a "
                       "lasting drop after an intervention) and reviewed visually. Confirm them against work orders.")
    with tabs[2]:
        last = g.tail(20).iloc[::-1]
        show_table(pd.DataFrame({
            "Snapshot start": last["t"].dt.strftime("%Y-%m-%d %H:%M"),
            "V_H (mm/s)": last["vel_h"].round(2), "V_V (mm/s)": last["vel_v"].round(2),
            "Temp (°C)": last["temp"].round(1), "ISO zone": last["zone"],
            "Predicted RUL": last["pred"].map(E.fmt_rul), "Below 120 h": np.where(last["below"], "Yes", ""),
            "Warning": np.where(last["alarm"], "⛔", "")}))
        st.caption("Last 20 snapshots, newest first. Values are cleaned (gaps interpolated, 3-point median).")
        csv = g[["t", "op_h", "vel_h", "vel_v", "temp", "zone", "pred", "alarm"] + E.FEATURES].copy()
        csv["t"] = csv["t"].dt.strftime("%Y-%m-%d %H:%M")
        st.download_button(f"Download all {len(g):,} snapshots of {E.short_name(mid)} (CSV)",
                           csv.round(4).to_csv(index=False).encode(), f"{E.short_name(mid)}_snapshots.csv", "text/csv")
    with tabs[3]:
        rg = engine["ranges"]
        now = g[E.FEATURES].iloc[-1]
        outside = (now > rg["p95"]) | (now < rg["p05"])
        show_table(pd.DataFrame({
            "Feature": [E.FEATURE_TEXT[f] for f in E.FEATURES],
            "Now": now.values.round(3),
            "Healthy range (5–95 %)": [f"{rg.loc[f, 'p05']:.3f} – {rg.loc[f, 'p95']:.3f}" for f in E.FEATURES],
            "Typical 10 days before failure": rg["late_median"].round(3).values,
            "Outside healthy range": np.where(outside, "Yes", ""),
            "Importance": engine["importance"].reindex(E.FEATURES).round(2).values}))
        st.caption("Healthy range and typical pre-failure value come from the synthetic training fleet "
                   "(observable failure modes). A log ratio of 0.69 means the level has doubled.")


def page_validation():
    st.title("Validation on plant data")
    st.markdown("The model never saw plant labels. It was trained only on the synthetic fleet and then applied to "
                "the plant record of SQF3–SQF10 (zero-shot transfer, experiment E2 in the report). Everything on "
                "this page is recomputed live from the bundled plant snapshots.")
    ev = run_eval(engine, "plant")
    lo, hi = E.clopper_pearson(ev["caught"], ev["n_events"])
    c = st.columns(5)
    c[0].metric("Events caught", f"{ev['caught']} of {ev['n_events']}",
                help=f"95 % Clopper-Pearson interval {lo * 100:.0f}–{hi * 100:.0f} %")
    c[1].metric("Median warning lead", f"{ev['lead_median']:.0f} h", help="Running hours before the inferred event")
    c[2].metric("AUC, fails within 10 days", f"{ev['auc']:.2f}")
    c[3].metric("False-alarm rate", f"{ev['false_alarm_pct']:.1f} %", help="Snapshots with ≥ 720 h left")
    c[4].metric("Alarms on healthy running", f"{ev['censored_alarms']} of {ev['censored_snaps']:,}",
                help="Snapshots with at least 1,500 running hours of record still ahead")
    st.info(f"With only nine events the uncertainty is wide: a detection rate of {ev['caught']}/{ev['n_events']} has "
            f"a 95 % interval of about {lo * 100:.0f}–{hi * 100:.0f} %. All four *clear* events were caught; the "
            "three misses are *moderate* events with peak V_H around 3–4 mm/s.", icon="ℹ️")

    t1, t2, t3, t4 = st.tabs(["Per event", "ROC", "Experiments", "Plant vs synthetic"])
    with t1:
        e = ev["events"]
        show_table(pd.DataFrame({
            "Fan": e["machine"].map(E.short_name), "Event": e["event_time"].dt.strftime("%Y-%m-%d"),
            "Grade": e["strength"], "Caught": np.where(e["caught"], "✅ Yes", "❌ No"),
            "Lead (running h)": e["lead_h"].map(lambda v: "–" if not np.isfinite(v) else f"{v:.0f}"),
            "Lead (running days)": e["lead_h"].map(lambda v: "–" if not np.isfinite(v) else f"{v / 24:.0f}"),
            "Peak V_H, last 10 days (mm/s)": e["peak_vh"].round(2), "Warnings > 30 days early": e["early"]}))
        pick = st.selectbox("Show event", range(len(e)),
                            format_func=lambda i: f"{E.short_name(e['machine'][i])}  {e['event_time'][i]:%d %b %Y}"
                                                  f"  ({e['strength'][i]})")
        Pp, _ = run(engine, load_plant(), "plant")
        gm = Pp[Pp["machine"] == e["machine"][pick]]
        show_fig(charts.event_chart(gm, e["event_time"][pick], e["strength"][pick], e["lead_h"][pick]))
    with t2:
        cc = st.columns([1, 1.3])
        with cc[0]:
            show_fig(charts.roc_chart(ev["fpr"], ev["tpr"], ev["auc"]))
        with cc[1]:
            st.markdown(f"The question asked of every labelled snapshot ({ev['labelled']:,} in total) is *will the "
                        "fan fail within the next 10 running days?* The predicted RUL ranks the snapshots; an AUC "
                        f"of **{ev['auc']:.2f}** means a snapshot 10 days or less before failure gets a lower "
                        "predicted RUL than a healthy one in that share of random pairs.\n\nR² on the plant data is "
                        "negative because plant fans stay at their normal level until about ten days before failure, "
                        "while the synthetic fleet degrades over the whole 1,500-hour window. The model is "
                        "therefore used as an **early-warning tool**, not as a long-range RUL clock.")
    with t3:
        R = stored_results()
        main = R["grid"]["v2|observable|cap1500"]
        names = {"E2": "E2 Synthetic only → plant (deployed)", "E3": "E3 Plant only (leave one fan out)",
                 "E4": "E4 Synthetic + plant (plant weight 5)"}
        rows = [["E1 Synthetic → synthetic test", f"{main['E1']:.2f}", "–", "–", "–", "–", "–"]]
        for k, n in names.items():
            v = main[k]
            rows.append([n, f"{v['R2']:.2f}", f"{v['MAE']:.0f}", f"{v['AUC_10d']:.2f}",
                         f"{v['caught']}/{v['events']}",
                         "–" if not (isinstance(v['lead_h_median'], (int, float)) and np.isfinite(v['lead_h_median']))
                         else f"{v['lead_h_median']:.0f}", f"{v['false_alarm_pct']:.1f}"])
        show_table(pd.DataFrame(rows, columns=["Experiment", "R²", "MAE (h)", "AUC 10 days", "Events caught",
                                               "Median lead (h)", "False alarms (%)"]))
        st.caption("Main configuration: trailing-reference features, observable failure modes, cap 1,500 h. Models "
                   "trained on plant data (E3, E4) never cross the 120-hour threshold, because only about 10 % of "
                   "plant snapshots are within 10 days of an event, so they caught no event.")
        st.markdown("**All configurations (deployed experiment E2)**")
        cfg = []
        for k, v in R["grid"].items():
            fs_, sub, cap = k.split("|")
            e2 = v["E2"]
            cfg.append(["Trailing reference (v2)" if fs_ == "v2" else "Cycle-start baseline (v1)",
                        "Observable modes" if sub == "observable" else "All modes", int(cap[3:]),
                        round(v["E1"], 2), round(e2["AUC_10d"], 2), f"{e2['caught']}/{e2['events']}",
                        round(e2["false_alarm_pct"], 1)])
        cfg = pd.DataFrame(cfg, columns=["Features", "Synthetic machines", "Cap (h)", "R² synthetic test",
                                         "AUC plant", "Events caught", "False alarms (%)"])
        show_table(cfg.sort_values(["Features", "Synthetic machines", "Cap (h)"], ascending=[False, False, True]))
    with t4:
        R = stored_results()
        cc = st.columns([1.2, 1])
        with cc[0]:
            show_fig(charts.ratio_chart(R["ratio"]))
        with cc[1]:
            st.markdown("Median V_H relative to the start of the life-cycle, at fixed times before failure. "
                        "Synthetic machines degrade steadily across the whole window. The plant fans stay within "
                        "20–30 % of their baseline until about ten running days before the event, then rise "
                        "sharply. This difference in dynamics is the main gap between the synthetic data and the "
                        f"plant, and the reason the deployed model uses only the {R['obs_machines']} synthetic "
                        "machines whose failure mode the three plant channels can see.")


def page_method():
    st.title("Method & data")
    R = stored_results()
    st.markdown(
        "This app implements Chapter 5 of the M.Tech dissertation *A Machine Learning Framework for Predictive "
        "Health Monitoring and Remaining Useful Life Estimation of Manufacturing Machinery Using Multi-Sensor Data* "
        "(Subham Mishra, BITS Pilani WILP; Tata Motors, Jamshedpur).")
    c = st.columns(4)
    c[0].metric("Plant export rows", f"{R['raw_rows']:,}")
    c[1].metric("Labelled snapshots", f"{R['labelled']:,}")
    c[2].metric("Inferred failure events", "9")
    c[3].metric("Synthetic training machines", f"{engine['train_machines']}")
    st.subheader("Pipeline")
    st.markdown(
        "1. **Common features.** The plant export holds only vibration velocity (one row per axis, H or V) and "
        "temperature; every other column is empty. The model uses those three channels.\n"
        f"2. **Cleaning.** Duplicate records removed ({R['dups']:,} in the plant file), H and V rows joined per "
        "timestamp, logger drop-outs (all zero) dropped, negative velocity and temperature ≤ 0 set to missing.\n"
        f"3. **Running filter.** A sample counts as running when max(V_H, V_V) > {E.RUN_THR} mm/s "
        f"({R['running_pct']:.0f} % of plant samples).\n"
        "4. **12-hour running snapshots.** Running samples are accumulated by running time (each counts for its "
        "interval, at most 1.1 h) and the median of every 12 running hours forms one snapshot, matching the "
        "synthetic fleet. Stops do not age the fan.\n"
        "5. **Features.** Gaps interpolated, 3-point rolling median, then ten trailing-reference features: the last "
        "2 days compared with the fan's own level 14–60 days earlier (table below).\n"
        f"6. **Model.** Extra Trees ({engine['n_trees']} trees, min. 5 samples per leaf) trained on "
        f"{engine['train_machines']} synthetic machines ({engine['train_rows']:,} snapshots) whose failure mode the "
        "three channels can see: motor drive-end bearing, misalignment and impeller faults. RUL capped at 1,500 h.\n"
        "7. **Early warning.** Predicted RUL below 120 h on two consecutive snapshots.\n"
        "8. **Life-cycles.** For the plant fans the record is split at the inferred events, so the reference level "
        "restarts after each repair.")
    st.subheader("Features")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        show_table(pd.DataFrame({"Feature": E.FEATURES, "Meaning": [E.FEATURE_TEXT[f] for f in E.FEATURES]}))
    with c2:
        show_fig(charts.importance_chart(engine["importance"]))
    st.subheader("Inferred events")
    show_table(pd.DataFrame([{"Fan": E.short_name(m), "Last snapshot before the drop": d[:16], "Grade": s}
                             for m, d, s in E.EVENTS]))
    st.subheader("Limitations")
    st.markdown(
        "- **Inferred labels.** The export has no maintenance records; the nine events were inferred from the "
        "vibration signal and reviewed visually. Confirm them against work orders.\n"
        "- **Three channels.** Faults that do not raise motor DE vibration (stator winding, fan-side bearing "
        "without transmission) cannot be seen. The ten-sensor model in the main framework covers those.\n"
        "- **Short warning horizon.** Plant fans degrade over about ten running days, so the RUL is reliable as an "
        "early warning (below about 240 h), not as a long-range estimate.\n"
        "- **Few events.** Nine events give a wide confidence interval on the detection rate. Re-evaluate as more "
        "failures are recorded.")


# ------------------------------------------------------------------------------------------ render
def render(source):
    """Sidebar controls and page for the plant fans: source is PLANT (stored record) or UPLOAD (new export)."""
    global page, engine, S, key, has_labels, audit, P, fs, ev_all
    _check_files()
    page = st.sidebar.radio("Page", PAGES, key="plant_page")
    st.sidebar.divider()
    engine = get_engine()
    S, key, has_labels, audit = None, "", False, None
    
    if source == PLANT:
        S, key, has_labels = load_plant(), "plant", True
    else:
        up = st.sidebar.file_uploader("Plant export or readings (CSV)", type=["csv"])
        include_other = st.sidebar.checkbox("Include machines that are not RC fans", value=False,
                                            help="The model was built for the RC fan motors. Other machines (e.g. the "
                                                 "cooling-tower pump) are left out unless you tick this.")
        with st.sidebar.expander("Accepted formats"):
            st.markdown(E.format_help())
            st.download_button("Download a sample export (3 fans, Jun–Oct 2026)",
                               find_file("sample_plant_export.csv").read_bytes(), "sample_plant_export.csv", "text/csv")
        if up is not None:
            data = up.getvalue()
            S, audit, fmt = process_upload(data, up.name, include_other)
            if S is None:
                st.sidebar.error(audit["error"])
            else:
                key = f"upload-{up.name}-{len(data)}-{include_other}"
                st.sidebar.success(f"{S['machine'].nunique()} machine(s), {len(S):,} snapshots "
                                   f"({'plant export' if fmt == 'export' else 'one row per reading'}).")
    
    st.sidebar.divider()
    st.sidebar.caption("RUL is in **running hours** (stops do not count) and capped at 1,500 h. The failure events "
                       "used for validation were inferred from the vibration record, not from work orders.")
    
    if page in ("Fleet status", "Fan detail") and S is None:
        st.title(page)
        st.info("Upload a CSV in the sidebar (a sample file is under *Accepted formats*), or choose *Plant fans* as the "
                "data source.")
        st.stop()
    if S is not None:
        P, fs = run(engine, S, key)
        ev_all = run_eval(engine, "plant") if has_labels else None
    {"Fleet status": page_fleet, "Fan detail": page_fan, "Validation on plant data": page_validation,
     "Method & data": page_method}[page]()
