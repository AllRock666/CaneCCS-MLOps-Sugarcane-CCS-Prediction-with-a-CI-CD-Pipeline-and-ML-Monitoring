import json
import time
from collections import deque
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# ---- Prometheus metrics -------------------------------------------------
REQUESTS = Counter("cane_http_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("cane_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
PRED_HIST = Histogram("cane_predicted_ccs", "Predicted CCS %", buckets=(6, 7, 8, 9, 10, 11, 12, 13))
PRED_TOTAL = Counter("cane_predictions_total", "Predictions served")
PRED_ERRORS = Counter("cane_prediction_errors_total", "Invalid or failed predictions")
BRIX_DRIFT = Gauge("cane_brix_drift_zscore", "Rolling mean brix vs training mean, in std units")
MODEL_INFO = Gauge("cane_model_info", "Model metadata", ["version", "algorithm"])
LOW_CCS = Counter("cane_low_ccs_batches_total", "Batches predicted below 9 CCS")


def create_app():
    app = Flask(__name__)
    model = joblib.load(MODEL_DIR / "ccs_model.joblib")
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    MODEL_INFO.labels(meta["model_version"], meta["algorithm"]).set(1)
    window = deque(maxlen=50)

    def timed(endpoint):
        def deco(fn):
            def wrapper(*a, **kw):
                t0 = time.perf_counter()
                resp = fn(*a, **kw)
                status = resp[1] if isinstance(resp, tuple) else 200
                LATENCY.labels(endpoint).observe(time.perf_counter() - t0)
                REQUESTS.labels(endpoint, str(status)).inc()
                return resp
            wrapper.__name__ = fn.__name__
            return wrapper
        return deco

    @app.get("/")
    @timed("index")
    def index():
        return render_template("index.html", varieties=meta["varieties"], meta=meta)

    @app.get("/health")
    @timed("health")
    def health():
        return jsonify(status="ok", model_version=meta["model_version"])

    @app.get("/model-info")
    @timed("model_info")
    def model_info():
        return jsonify(meta)

    @app.post("/predict")
    @timed("predict")
    def predict():
        try:
            d = request.get_json(force=True)
            variety = d["variety"]
            row = {
                "brix": float(d["brix"]), "pol": float(d["pol"]),
                "fibre_pct": float(d["fibre_pct"]), "trash_pct": float(d["trash_pct"]),
                "days_since_harvest": int(d["days_since_harvest"]),
                "variety_code": meta["varieties"].index(variety),
            }
            if not (10 <= row["brix"] <= 30) or row["pol"] > row["brix"]:
                raise ValueError("brix must be 10-30 and pol cannot exceed brix")
            ccs = float(model.predict(pd.DataFrame([row])[meta["features"]])[0])
        except (KeyError, ValueError, TypeError) as e:
            PRED_ERRORS.inc()
            return jsonify(error=str(e)), 400
        PRED_TOTAL.inc()
        PRED_HIST.observe(ccs)
        if ccs < 9:
            LOW_CCS.inc()
        window.append(row["brix"])
        BRIX_DRIFT.set(((sum(window) / len(window)) - meta["train_brix_mean"]) / meta["train_brix_std"])
        grade = "A" if ccs >= 10.5 else "B" if ccs >= 9.5 else "C"
        return jsonify(ccs=round(ccs, 3), grade=grade, model_version=meta["model_version"])

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    return app


app = create_app() if (MODEL_DIR / "ccs_model.joblib").exists() else None

if __name__ == "__main__":
    (app or create_app()).run(host="0.0.0.0", port=5000)
