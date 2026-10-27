import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOOD = {"variety": "Co-86032", "brix": 19.5, "pol": 17.2, "fibre_pct": 13.5,
        "trash_pct": 2.5, "days_since_harvest": 1}


@pytest.fixture(scope="session")
def client():
    if not (ROOT / "model" / "ccs_model.joblib").exists():
        subprocess.check_call([sys.executable, str(ROOT / "train.py")])
    from app.main import create_app
    return create_app().test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert 8 < r.get_json()["ccs"] < 15


def test_staleness_lowers_ccs(client):
    fresh = client.post("/predict", json=GOOD).get_json()["ccs"]
    stale = client.post("/predict", json={**GOOD, "days_since_harvest": 5}).get_json()["ccs"]
    assert stale < fresh


def test_rejects_pol_above_brix(client):
    assert client.post("/predict", json={**GOOD, "pol": 25}).status_code == 400


def test_rejects_missing_field(client):
    bad = {k: v for k, v in GOOD.items() if k != "brix"}
    assert client.post("/predict", json=bad).status_code == 400


def test_metrics_exposed(client):
    client.post("/predict", json=GOOD)
    body = client.get("/metrics").get_data(as_text=True)
    assert "cane_predictions_total" in body and "cane_brix_drift_zscore" in body
