import pandas as pd
import json
import os

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print(" Numeric drift monitoring started")

# =====================
# FEATURES
# =====================

FEATURES = [
    "Poids",
    "Volume",
    "Conductivite",
    "Opacite",
    "Rigidite",
    "Source"
]

# =====================
# LOAD DATA
# =====================

reference_data = pd.read_csv(
    "data/processed/df_clean.csv"
)

current_data = pd.read_csv(
    "logs/current_data.csv"
)

reference_data = reference_data[FEATURES]
current_data = current_data[FEATURES]

# =====================
# FIX TYPES
# =====================

reference_data["Source"] = reference_data["Source"].astype(str)

current_data["Source"] = current_data["Source"].astype(str)

# =====================
# EVIDENTLY REPORT
# =====================

report = Report(
    metrics=[DataDriftPreset()]
)

report.run(
    reference_data=reference_data,
    current_data=current_data
)

# =====================
# SAVE HTML REPORT
# =====================

os.makedirs(
    "reports",
    exist_ok=True
)

report.save_html(
    "reports/report.html"
)

# =====================
# EXTRACT DRIFT SCORE
# =====================

result = report.as_dict()

try:

    drift_score = result["metrics"][0]["result"]["dataset_drift"]

except:

    drift_score = 0

# =====================
# SAVE JSON
# =====================

os.makedirs(
    "reports/evidently",
    exist_ok=True
)

with open(
    "reports/evidently/drift_metrics.json",
    "w"
) as f:

    json.dump({
        "data_drift_score": float(drift_score)
    }, f)

print(" Numeric Drift:", drift_score)

print(" Monitoring finished")