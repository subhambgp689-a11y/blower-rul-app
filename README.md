# RC Fan RUL Monitor

Streamlit front end for the M.Tech dissertation *A Machine Learning Framework for Predictive Health Monitoring and
Remaining Useful Life Estimation of Manufacturing Machinery Using Multi-Sensor Data* (Chapter 7).

One application with three data sources, chosen in the sidebar. Each data source keeps its own model:

| Data source | Model | Pages |
|---|---|---|
| Synthetic fleet (10 features) | Ten-feature Extra Trees trained on 150 synthetic machines, 90 % conformal intervals, health index | Fleet overview, Machine detail, Quick estimate, Model & results |
| Plant fans (SQF3–SQF10) | Three-channel Extra Trees (V_H, V_V, temperature) trained on the 91 synthetic machines with observable failure modes | Fleet status, Fan detail, Validation on plant data, Method & data |
| Upload a new plant export | Same three-channel model | Fleet status, Fan detail, Method & data |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first start both models are trained from the files in `data/` (about a minute) and cached.

## Files

| File | Purpose |
|---|---|
| `app.py` | Entry point: page settings, sidebar data-source choice, hands over to a view |
| `fleet_view.py`, `fleet_engine.py`, `fleet_charts.py` | Synthetic-fleet view, ten-feature model and its charts |
| `plant_view.py`, `plant_engine.py`, `plant_charts.py` | Plant view (stored record and uploads), three-channel model and its charts |
| `data/` | Synthetic training and test data, plant snapshots, a sample plant export |
| `assets/` | Framework figure and stored validation results |

## Deploy on Streamlit Community Cloud

Push this folder to a GitHub repository, create a new app and select `app.py` as the entry point.

## Upload format (plant)

Either the plant export layout (`DATE_TIME`, `Equipment_Name`, `Velocity`, `Temperature`, `Axis` = H or V, one row per
axis) or one row per reading (`machine`, `timestamp`, `vel_h`, `vel_v`, `temp`). A sample export is in `data/`.
