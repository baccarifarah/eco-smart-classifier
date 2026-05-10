import os

import joblib
import pandas as pd
from fastapi import APIRouter

router = APIRouter()

MODEL_PATH = "models/nlp_model.pkl"


def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


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
