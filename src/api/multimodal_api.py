import os

import joblib
import pandas as pd
from fastapi import APIRouter

router = APIRouter()

MODEL_PATH = "models/multimodal_model.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


@router.post("/")
def predict_multimodal(data: dict):

    if model is None:
        return {"error": "Model not available (CI or missing file)"}

    df = pd.DataFrame([data])
    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}
