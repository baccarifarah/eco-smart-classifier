import os

import joblib
import pandas as pd
from fastapi import APIRouter, Path

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "multimodal_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(str(MODEL_PATH))


model = load_model()


@router.post("/")
def predict_multimodal(data: dict):

    if model is None:
        return {"error": "Model not available (CI or missing file)"}

    df = pd.DataFrame([data])
    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}
