# Blower RUL Monitor (Streamlit)

Web front end for the M.Tech dissertation *A Machine Learning Framework for Predictive Health Monitoring and
Remaining Useful Life Estimation of Manufacturing Machinery Using Multi-Sensor Data* (Subham Mishra, BITS Pilani
WILP, Tata Motors Jamshedpur).

It predicts the remaining useful life (RUL) of motor-driven centrifugal blowers from ten real-time sensor
features. Each RUL comes with a 90 % interval, a health-index alarm, an ISO 10816-3 zone and a hint of which
fault to check.

> The model is trained on a **simulated** run-to-failure fleet calibrated to the plant RC fan. Validate it on
> real machines before using it for maintenance decisions.

## Pages

| Page | What it shows |
|---|---|
| Fleet overview | Status of every machine (Critical / Plan maintenance / Watch / Healthy), predicted RUL with interval, ISO zone, health alarm and fault hint; the 20 most urgent machines on one chart; CSV download |
| Machine detail | RUL trend with 90 % band (and true RUL for demo data), health index with alarm periods, all sensor trends with ISO limits, latest readings against the commissioning baseline |
| Quick estimate | Type in a machine's commissioning readings and current readings to get a one-off estimate (with examples for each failure mode) |
| Model & results | Accuracy of the deployed model on 50 unseen machines, the 12-model benchmark, feature importance, early-warning results and the method |

## Run on your computer

```bash
cd rul_streamlit_app
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at http://localhost:8501. On first start the app trains the model on `data/train.csv`
(about 10–30 seconds). After that it is cached.

## Put it online (Streamlit Community Cloud, free)

1. Create a GitHub repository and upload the contents of this folder, including the `data/`, `assets/` and
   `.streamlit/` folders.
2. Sign in at https://share.streamlit.io with GitHub, click **Create app**, choose the repository and set the main
   file to `app.py`.
3. Click **Deploy**. You will get a public link you can share with your supervisor and examiner.

## Using your own data

Upload a CSV in the sidebar with the following columns:

- `machine_id`
- `cycle` (snapshot number) **or** `timestamp`
- the ten features: `load_pct`, `motor_de_vel_h_mms`, `motor_de_vel_v_mms`, `motor_de_vel_a_mms`,
  `motor_de_acc_g`, `motor_de_temp_c`, `fan_de_vel_h_mms`, `fan_de_temp_c`, `winding_temp_c`,
  `current_unbalance_pct`

The model expects one snapshot every 12 operating hours. If your timestamps are denser, for example every 10
minutes, the app offers to average them into 12-hour snapshots. Each machine's first 20 snapshots (10 days) are
treated as its healthy baseline, so include its history from commissioning or from just after a repair.
`data/sample_upload.csv` is a working example.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit pages |
| `rul_engine.py` | Pre-processing, Extra Trees model, conformal intervals, health index and status rules (no Streamlit code; reusable on its own) |
| `charts.py` | Charts (matplotlib) |
| `data/train.csv` | 150 simulated run-to-failure histories used for training |
| `data/test.csv`, `data/test_full_labels.csv`, `data/machines_metadata.csv` | Demo fleet of 50 unseen machines, with true RUL |
| `data/sample_upload.csv` | Example upload (2 machines) |
| `assets/report_results.json`, `assets/framework.png` | Results and framework diagram from the final report |

## Model in one paragraph

Sensors are cleaned (gap filling and despiking), corrected for load, expressed as the change since the machine's
first 10 days, and smoothed over 7 days. An Extra Trees model (150 trees) maps these ten features to RUL, capped
at 1,500 h. On the 50 test machines it reaches R² 0.88 over all snapshots and 0.95 at the last snapshot. A
split-conformal 90 % interval is added, which covers 94 % of test snapshots. A degradation-directed Mahalanobis
health index raises an alarm about 110 days before failure in median.
