import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.monitoring.metrics import PREDICTIONS

router = APIRouter()

# =========================
# BASE PATH (RENDER SAFE)
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "numeric_model.pkl"


# =========================
# LOAD MODEL
# =========================
def get_model():
    if not MODEL_PATH.exists():
        print("MODEL NOT FOUND:", MODEL_PATH)
        return None
    return joblib.load(str(MODEL_PATH))


# =========================
# FEATURES
# =========================
FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]

# =========================
# LOG PATHS (ABSOLUTE)
# =========================
LOG_DIR = BASE_DIR / "logs"
LOG_FILE_CSV = LOG_DIR / "current_data.csv"
LOG_FILE_JSON = LOG_DIR / "predictions.jsonl"


@router.post("/")
def predict_numeric(data: dict):

    model = get_model()

    if model is None:
        return {"error": "Model not available"}

    # =========================
    # VALIDATION INPUT
    # =========================
    for f in FEATURES:
        if f not in data:
            raise HTTPException(status_code=422, detail=f"Missing field: {f}")

    df = pd.DataFrame([data])[FEATURES]

    prediction = model.predict(df)

    # metric monitoring
    PREDICTIONS.labels(model="numeric").inc()

    # =========================
    # CREATE LOG DIR SAFELY
    # =========================
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # CSV LOG
    # =========================
    df_log = df.copy()
    df_log["prediction"] = prediction[0]

    if LOG_FILE_CSV.exists():
        df_log.to_csv(LOG_FILE_CSV, mode="a", header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE_CSV, index=False)

    # =========================
    # JSONL LOG (BEST PRACTICE)
    # =========================
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "features": data,
        "prediction": int(prediction[0]),
    }

    with open(LOG_FILE_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": int(prediction[0])}
