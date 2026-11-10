"""
RC Fan RUL Monitor: Streamlit front end for the M.Tech dissertation
"A Machine Learning Framework for Predictive Health Monitoring and Remaining Useful Life Estimation
of Manufacturing Machinery Using Multi-Sensor Data" (Subham Mishra, BITS Pilani WILP, 2026).

One application, three data sources chosen in the sidebar. Each source keeps its own model:
  * Synthetic fleet  - ten-feature Extra Trees model of Chapter 5 on the 50 synthetic test blowers (true RUL known)
  * Plant fans       - three-channel early-warning model of Chapter 6 on the stored record of SQF3-SQF10
  * Upload           - the same three-channel model on a newer plant monitoring export

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="RC Fan RUL Monitor", page_icon="⚙️", layout="wide")

import fleet_view  # noqa: E402  (imported after set_page_config)
import plant_view  # noqa: E402

SYNTH = "Synthetic fleet (10 features)"
PLANT = "Plant fans (SQF3–SQF10)"
UPLOAD = "Upload a new plant export"
HELP = {SYNTH: "Ten-feature RUL model with 90 % intervals and health index, on 50 synthetic test blowers whose "
               "true RUL is known.",
        PLANT: "Three-channel early-warning model (V_H, V_V, temperature) on the stored plant record, validated "
               "against nine inferred failure events.",
        UPLOAD: "The same three-channel model applied to a newer monitoring export of the plant fans."}

st.sidebar.title("⚙️ RC Fan RUL Monitor")
st.sidebar.caption("Predictive health monitoring and remaining-useful-life estimation for motor-driven RC fans.")
source = st.sidebar.radio("Data source", [SYNTH, PLANT, UPLOAD], help="Each data source uses its own model.")
st.sidebar.caption(HELP[source])
st.sidebar.divider()

if source == SYNTH:
    fleet_view.render()
else:
    plant_view.render(plant_view.PLANT if source == PLANT else plant_view.UPLOAD)
