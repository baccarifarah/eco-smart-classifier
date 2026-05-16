from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter

router = APIRouter()

# ========================
# BASE DIRECTORY (robuste)
# ========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "models" / "nlp_model.pkl"


# ========================
# LOAD MODEL
# ========================
def get_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(str(MODEL_PATH))


# ========================
# LOG FILE (ABSOLUTE PATH)
# ========================
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "current_nlp.csv"


@router.post("/")
def predict_nlp(data: dict):

    model = get_model()

    if model is None:
        return {"error": "Model not available"}

    text = data["Rapport_Collecte"]

    prediction = model.predict([text])

    # ========================
    # CREATE LOG FOLDER SAFE
    # ========================
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df_log = pd.DataFrame(
        [{"Rapport_Collecte": text, "prediction": int(prediction[0])}]
    )

    if LOG_FILE.exists():
        df_log.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE, index=False)

    return {"prediction": int(prediction[0])}
