from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
import os
import json
from datetime import datetime

#  PROMETHEUS IMPORT UNIQUE
from src.monitoring.metrics import PREDICTIONS

router = APIRouter()

model = joblib.load("models/numeric_model.pkl")

FEATURES = [
    "Poids",
    "Volume",
    "Conductivite",
    "Opacite",
    "Rigidite",
    "Source"
]

LOG_FILE_CSV = "logs/current_data.csv"
LOG_FILE_JSON = "logs/predictions.json"


@router.post("/")
def predict_numeric(data: dict):

    # ✅ VALIDATION SAFE
    for f in FEATURES:
        if f not in data:
            raise HTTPException(status_code=422, detail=f"Missing field: {f}")

    # DATAFRAME
    df = pd.DataFrame([data])[FEATURES]

    # PREDICTION
    prediction = model.predict(df)

    #  PROMETHEUS COUNTER INCREMENT
    PREDICTIONS.labels(model="numeric").inc()

    # LOG FOLDER
    os.makedirs("logs", exist_ok=True)

    # CSV LOG
    df_log = df.copy()
    df_log["prediction"] = prediction

    if os.path.exists(LOG_FILE_CSV):
        df_log.to_csv(LOG_FILE_CSV, mode="a", header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE_CSV, index=False)

    # JSON LOG
    log_entry = {
        "timestamp": str(datetime.now()),
        "features": data,
        "prediction": int(prediction[0])
    }

    with open(LOG_FILE_JSON, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "prediction": int(prediction[0])
    }