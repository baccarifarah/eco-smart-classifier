import json
import os

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]

reference_data = pd.read_csv("data/processed/df_clean.csv")[FEATURES]
current_data = pd.read_csv("logs/current_data.csv")[FEATURES]

reference_data["Source"] = reference_data["Source"].astype(str)
current_data["Source"] = current_data["Source"].astype(str)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

os.makedirs("reports", exist_ok=True)
report.save_html("reports/report.html")

result = report.as_dict()

try:
    drift_score = result["metrics"][0]["result"]["dataset_drift"]
except Exception:
    drift_score = 0

os.makedirs("reports/evidently", exist_ok=True)

with open("reports/evidently/drift_metrics.json", "w") as f:
    json.dump({"data_drift_score": float(drift_score)}, f)
