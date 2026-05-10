import json
import os

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.multimodal_api import router as multi_router
from src.api.nlp_api import router as nlp_router
from src.api.numeric_api import router as numeric_router
from src.monitoring.metrics import (
    API_STATUS,
    DATA_DRIFT_SCORE,
    PREDICTIONS,
    TEXT_DRIFT_SCORE,
)

app = FastAPI(title="Eco-Smart Classifier API", version="1.0")

# =====================
# API STATUS
# =====================

API_STATUS.set(1)

# =====================
# ROUTES
# =====================

app.include_router(numeric_router, prefix="/predict/numeric")

app.include_router(nlp_router, prefix="/predict/nlp")

app.include_router(multi_router, prefix="/predict/multimodal")

# =====================
# HOME
# =====================


@app.get("/")
def home():

    return {"message": "Eco-Smart API running "}


# =====================
# METRICS
# =====================


@app.get("/metrics")
def metrics():

    # DATA DRIFT

    if os.path.exists("reports/evidently/drift_metrics.json"):

        with open("reports/evidently/drift_metrics.json") as f:

            data = json.load(f)

            DATA_DRIFT_SCORE.set(data["data_drift_score"])

    # TEXT DRIFT

    if os.path.exists("reports/evidently/text_drift.json"):

        with open("reports/evidently/text_drift.json") as f:

            data = json.load(f)

            TEXT_DRIFT_SCORE.set(data["text_drift_score"])

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
