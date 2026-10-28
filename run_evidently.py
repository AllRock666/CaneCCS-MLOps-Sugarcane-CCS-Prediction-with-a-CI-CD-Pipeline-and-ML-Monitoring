"""Evidently Data Drift and Target Drift Monitoring for CaneCCS MLOps."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric
)
from train import make_data, FEATURES

def main():
    print("Generating reference and production datasets...")
    # Reference training dataset
    df_ref = make_data(n=2000, seed=42)
    
    # Current production dataset: Simulated drift with higher brix, delayed crushing, and elevated trash
    rng = np.random.default_rng(2026)
    n_curr = 200
    brix_curr = rng.normal(23.2, 1.5, n_curr).clip(16, 26)
    purity_curr = rng.normal(82, 4.0, n_curr).clip(70, 92)
    pol_curr = brix_curr * purity_curr / 100
    fibre_curr = rng.normal(15.0, 1.3, n_curr).clip(11, 20)
    trash_curr = rng.gamma(3.5, 1.5, n_curr).clip(2, 16)
    days_curr = rng.integers(2, 7, n_curr)
    variety_curr = rng.integers(0, 3, n_curr)
    
    # CCS calculation
    ccs_juice_curr = 0.73 * (pol_curr - 0.4 * (brix_curr - pol_curr))
    ccs_curr = ccs_juice_curr * (1 - fibre_curr / 100) * (1 - trash_curr / 100) - 0.28 * days_curr + rng.normal(0, 0.12, n_curr)
    
    df_curr = pd.DataFrame({
        "brix": brix_curr,
        "pol": pol_curr,
        "fibre_pct": fibre_curr,
        "trash_pct": trash_curr,
        "days_since_harvest": days_curr,
        "variety_code": variety_curr,
        "ccs": ccs_curr
    })
    
    print("Computing Evidently Data Drift & Target Drift Report...")
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
        ColumnDriftMetric(column_name="brix"),
        ColumnDriftMetric(column_name="ccs"),
        DataDriftPreset()
    ])
    
    report.run(reference_data=df_ref, current_data=df_curr)
    
    reports_dir = Path("docs/evidence")
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "evidently_drift_report.html"
    json_path = reports_dir / "evidently_drift_report.json"
    
    report.save_html(str(html_path))
    
    summary = report.as_dict()
    json_path.write_text(json.dumps(summary, indent=2))
    
    print("\n" + "=" * 60)
    print("EVIDENTLY MONITORING REPORT SUMMARY")
    print("=" * 60)
    metrics_list = summary.get("metrics", [])
    if metrics_list:
        drift_res = metrics_list[0].get("result", {})
        print(f"Dataset Drift Detected: {drift_res.get('dataset_drift')}")
        print(f"Drift Share: {drift_res.get('drift_share', 0):.2%}")
        print(f"Number of Drifted Columns: {drift_res.get('number_of_drifted_columns')} / {drift_res.get('number_of_columns')}")
    print(f"\nHTML Report saved to: {html_path.resolve()}")
    print(f"JSON Report saved to: {json_path.resolve()}")

if __name__ == "__main__":
    main()
