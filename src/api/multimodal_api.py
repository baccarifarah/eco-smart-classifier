from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "multimodal_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        print("MODEL NOT FOUND:", MODEL_PATH)
        return None
    return joblib.load(str(MODEL_PATH))


def get_model():
    return load_model()


@router.post("/")
def predict_multimodal(data: dict):

    model = get_model()

    if model is None:
        return {"error": "Model not available (CI or missing file)"}

    df = pd.DataFrame([data])
    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}
