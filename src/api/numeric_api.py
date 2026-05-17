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
# DEBUG TEMPORAIRE
# =========================
print("\n========== DEBUG RENDER ==========")
print("BASE_DIR =", BASE_DIR)
print("MODEL_PATH =", MODEL_PATH)
print("MODEL EXISTS =", MODEL_PATH.exists())

models_dir = BASE_DIR / "models"

if models_dir.exists():
    print("MODELS DIRECTORY EXISTS")
    print("MODELS CONTENT =")

    for file in models_dir.iterdir():
        print(" -", file.name)

else:
    print("MODELS DIRECTORY DOES NOT EXIST")

print("==================================\n")


# =========================
# LOAD MODEL
# =========================
def get_model():

    print("\n--- LOADING MODEL ---")
    print("Trying to load:", MODEL_PATH)

    if not MODEL_PATH.exists():
        print("MODEL NOT FOUND:", MODEL_PATH)
        return None

    print("MODEL FOUND SUCCESSFULLY")

    return joblib.load(str(MODEL_PATH))


# =========================
# FEATURES
# =========================
FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]

# =========================
# LOG PATHS
# =========================
LOG_DIR = BASE_DIR / "logs"

LOG_FILE_CSV = LOG_DIR / "current_data.csv"
LOG_FILE_JSON = LOG_DIR / "predictions.jsonl"


@router.post("/")
def predict_numeric(data: dict):

    # =========================
    # LOAD MODEL
    # =========================
    model = get_model()

    if model is None:
        return {"error": "Model not available", "model_path": str(MODEL_PATH)}

    # =========================
    # VALIDATION INPUT
    # =========================
    for f in FEATURES:
        if f not in data:
            raise HTTPException(status_code=422, detail=f"Missing field: {f}")

    # =========================
    # DATAFRAME
    # =========================
    df = pd.DataFrame([data])[FEATURES]

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(df)

    # =========================
    # METRICS
    # =========================
    PREDICTIONS.labels(model="numeric").inc()

    # =========================
    # CREATE LOG DIR
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
    # JSON LOG
    # =========================
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "features": data,
        "prediction": int(prediction[0]),
    }

    with open(LOG_FILE_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    # =========================
    # RESPONSE
    # =========================
    return {"prediction": int(prediction[0])}
