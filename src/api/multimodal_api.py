from fastapi import APIRouter
import joblib
import pandas as pd

router = APIRouter()

model = joblib.load("models/multimodal_model.pkl")


@router.post("/")
def predict_multimodal(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }