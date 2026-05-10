import os

import joblib
import pandas as pd
from fastapi import APIRouter

router = APIRouter()

model = joblib.load("models/nlp_model.pkl")

LOG_FILE = "logs/current_nlp.csv"


@router.post("/")
def predict_nlp(data: dict):

    text = data["Rapport_Collecte"]

    prediction = model.predict([text])

    # =====================
    # LOG NLP (IMPORTANT)
    # =====================
    os.makedirs("logs", exist_ok=True)

    df_log = pd.DataFrame(
        [{"Rapport_Collecte": text, "prediction": int(prediction[0])}]
    )

    if os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE, index=False)

    return {"prediction": int(prediction[0])}
