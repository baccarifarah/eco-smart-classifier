import json
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.multimodal_api import router as multi_router
from src.api.nlp_api import router as nlp_router
from src.api.numeric_api import router as numeric_router
from src.monitoring.metrics import API_STATUS, DATA_DRIFT_SCORE, TEXT_DRIFT_SCORE

app = FastAPI(
    title="Eco-Smart Classifier API",
    version="1.0",
)

# =====================
# CORS
# =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# ROUTES
# =====================
app.include_router(numeric_router, prefix="/predict/numeric")
app.include_router(nlp_router, prefix="/predict/nlp")
app.include_router(multi_router, prefix="/predict/multimodal")

# =====================
# BASE DIR (SAFE PATHS)
# =====================
BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports" / "evidently"


API_STATUS.set(1)


@app.get("/")
def home():
    return {"message": "Eco-Smart API running"}


@app.get("/routes")
def routes():
    return [route.path for route in app.routes]


@app.get("/metrics")
def metrics():

    drift_file = REPORT_DIR / "drift_metrics.json"
    text_drift_file = REPORT_DIR / "text_drift.json"

    # =====================
    # DATA DRIFT
    # =====================
    if drift_file.exists():
        with open(drift_file, encoding="utf-8") as f:
            data = json.load(f)
            DATA_DRIFT_SCORE.set(data.get("data_drift_score", 0))

    # =====================
    # TEXT DRIFT
    # =====================
    if text_drift_file.exists():
        with open(text_drift_file, encoding="utf-8") as f:
            data = json.load(f)
            TEXT_DRIFT_SCORE.set(data.get("text_drift_score", 0))

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
