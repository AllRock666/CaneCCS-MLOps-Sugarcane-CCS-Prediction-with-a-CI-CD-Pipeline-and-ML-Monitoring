"""MLflow Experiment Tracking and Drift Detection for CaneCCS-MLOps.

Logs model parameters, metrics, artifacts to MLflow, and executes statistical
data drift detection (Kolmogorov-Smirnov test and Z-score tracking) comparing
reference training distribution with current production/simulated incoming batches.
"""
import json
import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from train import make_data, FEATURES, VARIETIES, MODEL_DIR

EXPERIMENT_NAME = "CaneCCS-Quality-Prediction"

def run_experiment_and_drift():
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print("=" * 60)
    print("1. RUNNING MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)
    
    with mlflow.start_run(run_name="GradientBoosting_Baseline") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # 1. Generate & prepare reference dataset
        df = make_data(n=6000, seed=42)
        X = df[FEATURES]
        y = df["ccs"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        
        # Hyperparameters
        params = {
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.08,
            "random_state": 7,
            "features": FEATURES,
            "varieties": VARIETIES,
            "n_train_samples": len(X_train)
        }
        mlflow.log_params({
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.08,
            "random_state": 7,
            "n_train_samples": len(X_train)
        })
        
        # 2. Train model
        model = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"]
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # 3. Log metrics
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        train_brix_mean = float(X_train["brix"].mean())
        train_brix_std = float(X_train["brix"].std())
        
        mlflow.log_metric("mae", round(mae, 4))
        mlflow.log_metric("r2_score", round(r2, 4))
        mlflow.log_metric("train_brix_mean", round(train_brix_mean, 3))
        mlflow.log_metric("train_brix_std", round(train_brix_std, 3))
        
        # 4. Save & Log model and metadata artifact
        meta_dict = {
            "model_version": "2026.10.28",
            "algorithm": "GradientBoostingRegressor",
            "features": FEATURES,
            "varieties": VARIETIES,
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "train_brix_mean": round(train_brix_mean, 3),
            "train_brix_std": round(train_brix_std, 3),
            "n_train": len(X_train)
        }
        
        reports_dir = Path("docs/evidence")
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "mlflow_summary.json"
        report_path.write_text(json.dumps(meta_dict, indent=2))
        mlflow.log_artifact(str(report_path))
        
        print(f"Metrics logged: MAE={mae:.4f}, R2={r2:.4f}, Brix Mean={train_brix_mean:.3f}")
        
        # ==========================================================
        # 2. DRIFT SIMULATION AND STATISTICAL DETECTION
        # ==========================================================
        print("\n" + "=" * 60)
        print("2. DRIFT SIMULATION & STATISTICAL DETECTION")
        print("=" * 60)
        
        # Reference distribution (Brix feature in training data)
        ref_brix = X_train["brix"].values
        
        # Scenario A: Normal production incoming batches (No drift)
        rng = np.random.default_rng(101)
        current_normal_brix = rng.normal(19.5, 1.6, 50).clip(14, 24)
        
        # Scenario B: Monsoon-delayed / High-trash abnormal harvest (Simulated Data Drift)
        # Higher moisture/stale brix shifts or severe weather shock (e.g. mean 23.5)
        current_drifted_brix = rng.normal(23.5, 1.4, 50).clip(18, 27)
        
        # Statistical test: Kolmogorov-Smirnov (KS) test & Z-Score drift
        # Test 1: Normal batch
        ks_stat_norm, p_val_norm = ks_2samp(ref_brix, current_normal_brix)
        zscore_norm = (np.mean(current_normal_brix) - train_brix_mean) / train_brix_std
        
        # Test 2: Drifted batch
        ks_stat_drift, p_val_drift = ks_2samp(ref_brix, current_drifted_brix)
        zscore_drift = (np.mean(current_drifted_brix) - train_brix_mean) / train_brix_std
        
        # Log drift metrics
        mlflow.log_metric("drift_ks_stat_normal", round(float(ks_stat_norm), 4))
        mlflow.log_metric("drift_p_value_normal", round(float(p_val_norm), 4))
        mlflow.log_metric("drift_zscore_normal", round(float(zscore_norm), 4))
        
        mlflow.log_metric("drift_ks_stat_drifted", round(float(ks_stat_drift), 4))
        mlflow.log_metric("drift_p_value_drifted", round(float(p_val_drift), 4))
        mlflow.log_metric("drift_zscore_drifted", round(float(zscore_drift), 4))
        
        drift_results = {
            "drift_type": "Data Drift (Covariate Shift on Brix / Sugar Quality Index)",
            "significance_threshold_alpha": 0.05,
            "zscore_alert_threshold": 1.5,
            "normal_scenario": {
                "mean_brix": round(float(np.mean(current_normal_brix)), 3),
                "zscore": round(float(zscore_norm), 3),
                "ks_statistic": round(float(ks_stat_norm), 4),
                "p_value": round(float(p_val_norm), 4),
                "drift_detected": bool(p_val_norm < 0.05 or abs(zscore_norm) > 1.5)
            },
            "drifted_scenario": {
                "mean_brix": round(float(np.mean(current_drifted_brix)), 3),
                "zscore": round(float(zscore_drift), 3),
                "ks_statistic": round(float(ks_stat_drift), 4),
                "p_value": round(float(p_val_drift), 6),
                "drift_detected": bool(p_val_drift < 0.05 or abs(zscore_drift) > 1.5)
            }
        }
        
        drift_report_path = reports_dir / "drift_report.json"
        drift_report_path.write_text(json.dumps(drift_results, indent=2))
        mlflow.log_artifact(str(drift_report_path))
        
        print("\nDrift Detection Summary:")
        print(json.dumps(drift_results, indent=2))
        print("\nMLflow run completed successfully!")

if __name__ == "__main__":
    run_experiment_and_drift()
