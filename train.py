"""Train the Cane CCS regressor on synthetic weighbridge-lab data.

CCS = Commercial Cane Sugar %, the quantity Indian sugar mills (e.g. around
Kolhapur) use to price cane and judge recovery. We simulate lab readings and
learn CCS from fast-to-measure features.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

VARIETIES = ["Co-86032", "CoM-0265", "Co-0238"]
VARIETY_BIAS = {"Co-86032": 0.0, "CoM-0265": -0.25, "Co-0238": 0.35}
FEATURES = ["brix", "pol", "fibre_pct", "trash_pct", "days_since_harvest", "variety_code"]
MODEL_DIR = Path(__file__).parent / "model"


def make_data(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    brix = rng.normal(19.5, 1.6, n).clip(14, 24)
    purity = rng.normal(88, 3.5, n).clip(75, 96)
    pol = brix * purity / 100
    fibre = rng.normal(13.5, 1.2, n).clip(10, 18)
    trash = rng.gamma(2.0, 1.2, n).clip(0, 12)
    days = rng.integers(0, 6, n)
    variety = rng.integers(0, len(VARIETIES), n)
    bias = np.array([VARIETY_BIAS[VARIETIES[v]] for v in variety])
    # Indian formula for CCS of juice, then correct for fibre, trash, field-to-mill delay
    ccs_juice = 0.73 * (pol - 0.4 * (brix - pol))
    ccs = ccs_juice * (1 - fibre / 100) * (1 - trash / 100)
    ccs = ccs - 0.28 * days - 0.0125 * days ** 2 + bias + rng.normal(0, 0.12, n)
    return pd.DataFrame({
        "brix": brix, "pol": pol, "fibre_pct": fibre, "trash_pct": trash,
        "days_since_harvest": days, "variety_code": variety, "ccs": ccs,
    })


def main():
    df = make_data()
    X, y = df[FEATURES], df["ccs"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7)
    model = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.08, random_state=7)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    meta = {
        "model_version": time.strftime("%Y.%m.%d"),
        "algorithm": "GradientBoostingRegressor",
        "features": FEATURES,
        "varieties": VARIETIES,
        "mae": round(float(mean_absolute_error(yte, pred)), 4),
        "r2": round(float(r2_score(yte, pred)), 4),
        "train_brix_mean": round(float(Xtr["brix"].mean()), 3),
        "train_brix_std": round(float(Xtr["brix"].std()), 3),
        "n_train": int(len(Xtr)),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "ccs_model.joblib")
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
