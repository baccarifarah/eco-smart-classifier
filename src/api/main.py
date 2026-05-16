import json
import os

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.multimodal_api import router as multi_router
from src.api.nlp_api import router as nlp_router
from src.api.numeric_api import router as numeric_router


from src.monitoring.metrics import (
    API_STATUS,
    DATA_DRIFT_SCORE,
    TEXT_DRIFT_SCORE,
)

# APP INIT

app = FastAPI(
    title="Eco-Smart Classifier API",
    version="1.0",
)

# CORS CONFIG (IMPORTANT FIX)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROUTES

app.include_router(numeric_router, prefix="/predict/numeric")
app.include_router(nlp_router, prefix="/predict/nlp")
app.include_router(multi_router, prefix="/predict/multimodal")



# API STATUS

API_STATUS.set(1)

# =====================
# HOME
# =====================

@app.get("/")
def home():
    return {
        "message": "Eco-Smart API running "
    }

# =====================
# DEBUG ROUTES (OPTIONAL)
# =====================

@app.get("/routes")
def routes():
    return [route.path for route in app.routes]

# =====================
# METRICS (PROMETHEUS + DRIFT)
# =====================

@app.get("/metrics")
def metrics():

    # =====================
    # DATA DRIFT
    # =====================

    if os.path.exists("reports/evidently/drift_metrics.json"):
        with open("reports/evidently/drift_metrics.json") as f:
            data = json.load(f)
            DATA_DRIFT_SCORE.set(data.get("data_drift_score", 0))

    # =====================
    # TEXT DRIFT
    # =====================

    if os.path.exists("reports/evidently/text_drift.json"):
        with open("reports/evidently/text_drift.json") as f:
            data = json.load(f)
            TEXT_DRIFT_SCORE.set(data.get("text_drift_score", 0))

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )