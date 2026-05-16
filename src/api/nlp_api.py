import os

import joblib
import pandas as pd
from fastapi import APIRouter, Path

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "nlp_model.pkl"


def get_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(str(MODEL_PATH))


LOG_FILE = "logs/current_nlp.csv"


@router.post("/")
def predict_nlp(data: dict):

    model = get_model()

    if model is None:
        return {"error": "Model not available"}

    text = data["Rapport_Collecte"]

    prediction = model.predict([text])

    os.makedirs("logs", exist_ok=True)

    df_log = pd.DataFrame(
        [{"Rapport_Collecte": text, "prediction": int(prediction[0])}]
    )

    if os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE, index=False)

    return {"prediction": int(prediction[0])}
