
import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# Import necessary functions from rgph_pipeline
from rgph_pipeline import (
    load_individu, run_etl, feature_engineering_individu, RANDOM_STATE
)

warnings.filterwarnings("ignore")

# Define paths for model and report as requested
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_individu_scorer.pkl")
REPORT_PATH = os.path.join(MODELS_DIR, "xgb_report.txt")

os.makedirs(MODELS_DIR, exist_ok=True)

# These features must match exactly what main.py expects and constructs
FEATURE_COLS = [
    "reg", "mil", "sexe", "AGE5",
    "LIEN_CM",
    "HANDI_VIS", "HANDI_AUD", "HANDI_MOB",
    "HANDI_MEM", "HANDI_ENTR", "HANDI_COM",
    "SIT_HANDICAP",
    "NIV_ET_AGR",
    "LIR_ECR",
    "a_diplome_eg",
    "a_formation_pro",
    "scolarise_actuel",
    "TY_ACT",
    "STAT_PROF",
    "ACT_SECTEUR",
    "indice_scolarisation",
    "est_occupe",
    "est_chomeur",
    "a_handicap",
    "ENF_VIV",
    "ENF_DEC"
]

def train_xgboost_individu():
    print("=============================================================")
    print("  RGPH 2014 — XGBoost Individual Vulnerability Training")
    print("=============================================================")

    # 1. Load Data
    data_individu_path = os.path.join("RGPH_projet", "Individu.csv")
    if not os.path.exists(data_individu_path):
        # Fallback to current dir if not in RGPH_projet (e.g. inside Docker)
        data_individu_path = "Individu.csv"
        if not os.path.exists(data_individu_path):
            print(f"[ERROR] Required file 'Individu.csv' not found. Please ensure it exists.")
            return

    df = load_individu(data_individu_path)

    # 2. ETL & Feature Engineering
    df = run_etl(df)
    df = feature_engineering_individu(df)

    # 3. Define Target Variable
    # Map SIT_HANDICAP (1-4) to 3 classes (0-2) to match VULN_LABELS in main.py
    # 1 (Pas de difficulté) -> 0 (Non vulnérable)
    # 2 (Peu de difficulté) -> 1 (Vulnérable)
    # 3 (Beaucoup de difficulté) & 4 (Incapable) -> 2 (Très vulnérable)
    
    if "SIT_HANDICAP" not in df.columns:
        print("[ERROR] SIT_HANDICAP column not found in data. Cannot train.")
        return

    print("[XGB] Mapping SIT_HANDICAP to 3 vulnerability classes...")
    df['target'] = df['SIT_HANDICAP'].map({
        1: 0,
        2: 1,
        3: 2,
        4: 2
    })
    
    # Drop rows where the target is NaN (shouldn't happen with SIT_HANDICAP usually)
    df_cleaned = df.dropna(subset=['target']).copy()
    
    available_features = [col for col in FEATURE_COLS if col in df_cleaned.columns]
    missing_features = set(FEATURE_COLS) - set(available_features)
    if missing_features:
        print(f"[WARN] Missing features from dataset: {missing_features}")
        # We'll fill them with NaN so XGBoost can handle them or we can impute
        for col in missing_features:
            df_cleaned[col] = np.nan
    
    X = df_cleaned[FEATURE_COLS].astype(float)
    y = df_cleaned['target'].astype(int)

    print(f"[XGB] Target classes distribution:\n{y.value_counts(normalize=True)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[XGB] Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # 4. Train XGBoost Model
    print("\n[XGB] Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[XGB] Training complete.")

    # 5. Evaluate and Report
    print("\n[XGB] Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred,
                                        target_names=["Non vulnérable", "Vulnérable", "Très vulnérable"],
                                        zero_division=0)

    # Feature Importance
    # XGBoost importance types: 'weight', 'gain', 'cover'
    # main.py expects 'gain' and 'cover'
    booster = model.get_booster()
    gains = booster.get_score(importance_type='gain')
    covers = booster.get_score(importance_type='cover')
    
    importances = []
    for feat in FEATURE_COLS:
        importances.append({
            'feature': feat,
            'gain': gains.get(feat, 0),
            'cover': covers.get(feat, 0)
        })
    
    importances_df = pd.DataFrame(importances).sort_values(by='gain', ascending=False)

    report_content = [
        "=" * 60,
        "  XGBoost Individual Vulnerability Prediction Report",
        "=" * 60,
        f"  Accuracy       : {accuracy:.4f}",
        "\n  Classification Report :",
        report_text,
        "\n  IMPORTANCE DES FEATURES (Top 15 by Gain) :",
        f"  {'Feature':<30} {'Gain':<10} {'Cover':<10}",
        "-" * 60
    ]

    for _, row in importances_df.head(15).iterrows():
        feat = row['feature']
        gain = row['gain']
        cover = row['cover']
        bar = "█" * int((gain / importances_df['gain'].max()) * 30) if importances_df['gain'].max() > 0 else ""
        report_content.append(f"  {feat:<30} {gain:<10.4f} {cover:<10.4f} {bar}")

    final_report = "\n".join(report_content)
    print(final_report)

    # Save Report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"\n[SAVE] Report saved to → {REPORT_PATH}")

    # Save Model Bundle (matches main.py loading logic)
    bundle = {
        "model": model,
        "features": FEATURE_COLS
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"[SAVE] Model bundle saved to → {MODEL_PATH}")

if __name__ == "__main__":
    train_xgboost_individu()
