# CaneCCS-MLOps
Predicts Commercial Cane Sugar (CCS %) of sugarcane batches at a sugar-mill weighbridge,
served by Flask, shipped via Jenkins + Docker, monitored with Prometheus + Grafana.

## Run locally
    pip install -r requirements.txt
    python train.py
    python -m app.main            # http://localhost:5000

## Full stack
    docker compose up -d --build
    App http://localhost:5000 | Prometheus http://localhost:9090 | Grafana http://localhost:3000 (admin/admin)

## Endpoints
GET / , GET /health , GET /model-info , POST /predict , GET /metrics

Sample:
    curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
     -d '{"variety":"Co-86032","brix":19.5,"pol":17.2,"fibre_pct":13.5,"trash_pct":2.5,"days_since_harvest":1}'
