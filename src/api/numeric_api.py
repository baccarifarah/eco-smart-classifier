import json
import os
from datetime import datetime

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.monitoring.metrics import PREDICTIONS

router = APIRouter()

MODEL_PATH = "models/numeric_model.pkl"


def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]

LOG_FILE_CSV = "logs/current_data.csv"
LOG_FILE_JSON = "logs/predictions.json"


@router.post("/")
def predict_numeric(data: dict):

    model = get_model()

    if model is None:
        return {"error": "Model not available"}

    for f in FEATURES:
        if f not in data:
            raise HTTPException(status_code=422, detail=f"Missing field: {f}")

    df = pd.DataFrame([data])[FEATURES]
    prediction = model.predict(df)

    PREDICTIONS.labels(model="numeric").inc()

    os.makedirs("logs", exist_ok=True)

    df_log = df.copy()
    df_log["prediction"] = prediction

    if os.path.exists(LOG_FILE_CSV):
        df_log.to_csv(LOG_FILE_CSV, mode="a", header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE_CSV, index=False)

    log_entry = {
        "timestamp": str(datetime.now()),
        "features": data,
        "prediction": int(prediction[0]),
    }

    with open(LOG_FILE_JSON, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": int(prediction[0])}
