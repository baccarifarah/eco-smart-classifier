from prometheus_client import Counter, Gauge

# =====================
# COUNTER
# =====================
PREDICTIONS = Counter("predictions_total", "Total predictions", ["model"])

# =====================
# API STATUS
# =====================
API_STATUS = Gauge("api_status", "API status (1=up, 0=down)")

# =====================
# DRIFT NUMERIC (EVIDENTLY)
# =====================
DATA_DRIFT_SCORE = Gauge("data_drift_score", "Numeric data drift score")

# =====================
# TEXT DRIFT (JENSEN-SHANNON)
# =====================
TEXT_DRIFT_SCORE = Gauge("text_drift_score", "Text drift score (Jensen-Shannon)")
