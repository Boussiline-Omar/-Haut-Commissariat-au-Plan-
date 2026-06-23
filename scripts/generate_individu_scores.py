import pandas as pd
import numpy as np
import os
import joblib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rgph_xgboost import load_and_etl, run_xgb_pipeline

DATA_INDIVIDU = PROJECT_ROOT / "Individu.csv"
SCORES_PATH = PROJECT_ROOT / "outputs" / "individu_scores.csv"

def generate_scores():
    print("Generating individual vulnerability scores via XGBoost...")

    # 1. Load and ETL
    # Note: load_and_etl internally handles the synthetic data if file is missing
    # For a real production run, we should ensure the file exists.
    if not DATA_INDIVIDU.exists():
        raise FileNotFoundError(f"{DATA_INDIVIDU} not found. Cannot generate real scores.")

    df = load_and_etl(str(DATA_INDIVIDU))

    # 2. Run XGBoost pipeline
    # This function handles target construction, training and scoring
    df_scored = run_xgb_pipeline(df, tune=False)

    # 3. Save output
    export_cols = [
        "reg", "region_label", "MEN_PRO", "NOR_MEN",
        "sexe", "sex_label", "AGE5", "age_group",
        "mil", "milieu_label",
        "score_vuln_indiv", "classe_vuln_indiv",
        "pred_xgb", "risk_level", "score_continu",
        "proba_xgb_0", "proba_xgb_1", "proba_xgb_2",
    ]
    export_cols = [c for c in export_cols if c in df_scored.columns]
    os.makedirs(SCORES_PATH.parent, exist_ok=True)
    df_scored[export_cols].to_csv(SCORES_PATH, index=False)

    print(f"Scores successfully saved to {SCORES_PATH}")

if __name__ == "__main__":
    generate_scores()
