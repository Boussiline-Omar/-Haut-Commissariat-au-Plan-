"""
=============================================================
  RGPH 2014 — Score de Vulnérabilité Individuelle (XGBoost)
  Projet : Système Analytique Intelligent (Soutenance)
=============================================================
Ce module s'exécute APRÈS le pipeline ETL (rgph_pipeline.py).
Il opère au niveau INDIVIDU (pas ménage), en exploitant
l'avantage natif de XGBoost : gestion des NaN sans imputation.

Structure :
  1. Configuration
  2. Chargement post-ETL
  3. Feature engineering individuel
  4. Construction de la cible individuelle
  5. Entraînement XGBoost + tuning
  6. Interprétabilité SHAP
  7. Export scores + modèle
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ──────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────

DATA_INDIVIDU   = "Individu.csv"
OUTPUT_DIR      = "outputs/"
MODEL_PATH      = os.path.join(OUTPUT_DIR, "xgb_individu_scorer.pkl")
SCORES_PATH     = os.path.join(OUTPUT_DIR, "individu_scores.csv")
REPORT_PATH     = os.path.join(OUTPUT_DIR, "xgb_report.txt")
SHAP_PATH       = os.path.join(OUTPUT_DIR, "shap_summary.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── Features individuelles pour XGBoost ──────────────────────
# XGBoost gère les NaN nativement → on NE remplace PAS
# les variables conditionnelles par des sentinels ici.
# On les passe telles quelles : le modèle apprend lui-même
# la structure conditionnelle de la donnée manquante.

XGB_FEATURES = [
    # Géographie & milieu
    "reg", "mil",
    # Démographie
    "sexe", "AGE5", "LIEN_CM",
    # Handicap (100% complet — très informatif)
    "HANDI_VIS", "HANDI_AUD", "HANDI_MOB",
    "HANDI_MEM", "HANDI_ENTR", "HANDI_COM",
    "SIT_HANDICAP",
    # Éducation
    "NIV_ET_AGR", "LIR_ECR",
    "a_diplome_eg", "a_formation_pro", "scolarise_actuel",
    # Emploi (NaN = inactif → XGBoost le gère)
    "TY_ACT", "STAT_PROF", "ACT_SECTEUR",
    # Features synthétiques calculées dans l'ETL
    "indice_scolarisation", "est_occupe", "est_chomeur", "a_handicap",
    # Fécondité (NaN si homme/hors-âge → XGBoost le gère)
    "ENF_VIV", "ENF_DEC",
]


# ──────────────────────────────────────────────────────────────
# 2. CHARGEMENT & REPRISE POST-ETL
# ──────────────────────────────────────────────────────────────

def load_and_etl(path: str) -> pd.DataFrame:
    """Charge le CSV individus et applique l'ETL minimal.

    Le module est conçu pour être exécuté directement depuis le dossier du projet.
    On importe donc rgph_pipeline en local, avec un fallback si le projet est
    transformé plus tard en package Python.
    """
    try:
        from RGPH_projet.rgph_pipeline import (
            load_individu, run_etl,
            feature_engineering_individu, generate_synthetic_data
        )
    except ImportError:
        from RGPH_projet.rgph_pipeline import (
            load_individu, run_etl,
            feature_engineering_individu, generate_synthetic_data
        )

    if not os.path.exists(path):
        print(f"[WARN] '{path}' introuvable → données synthétiques")
        df = generate_synthetic_data(n=50_000)
    else:
        df = load_individu(path)
    df = run_etl(df)
    df = feature_engineering_individu(df)
    return df


# ──────────────────────────────────────────────────────────────
# 3. CONSTRUCTION DE LA CIBLE INDIVIDUELLE
# ──────────────────────────────────────────────────────────────

def build_individual_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score de vulnérabilité individuelle (proxy).
    
    Logique : un individu est vulnérable si plusieurs facteurs
    de risque sont présents simultanément.
    
    Composantes :
      - Milieu rural
      - Faible niveau d'éducation
      - Inactivité économique (hors étudiant)
      - Présence d'handicap significatif
      - Tranche d'âge (très jeune ou senior)
      - Analphabétisme
    
    Résultat : score continu → terciles 0/1/2
    (Non vulnérable / Vulnérable / Très vulnérable)
    """
    print("\n[TARGET] Construction du score de vulnérabilité individuelle")

    score = pd.Series(0.0, index=df.index)

    # Milieu rural
    if "mil" in df.columns:
        score += (df["mil"] == 2).astype(float) * 1.5

    # Niveau d'éducation faible
    if "NIV_ET_AGR" in df.columns:
        score += ((5 - df["NIV_ET_AGR"].clip(0, 5)) / 5) * 2.0

    # Analphabétisme
    if "LIR_ECR" in df.columns:
        lir_risk = df["LIR_ECR"].map({1: 0.0, 2: 0.5, 3: 1.0}).fillna(0.5)
        score += lir_risk * 1.5

    # Inactivité économique (adulte non étudiant)
    if "TY_ACT" in df.columns and "AGE5" in df.columns:
        adulte = df["AGE5"] >= 15
        inactif_non_etud = df["TY_ACT"].isin([3, 5])  # femme au foyer, autre inactif
        score += (adulte & inactif_non_etud).astype(float) * 1.5
        chomeur = df["TY_ACT"].isin([1, 2])
        score += (adulte & chomeur).astype(float) * 1.0

    # Handicap
    if "SIT_HANDICAP" in df.columns:
        # SIT_HANDICAP : 1=sans, 2=légère, 3=modérée, 4=sévère
        handi_risk = df["SIT_HANDICAP"].map(
            {1: 0.0, 2: 0.5, 3: 1.0, 4: 2.0}
        ).fillna(0.0)
        score += handi_risk

    # Tranche d'âge vulnérable (< 5 ans ou >= 65 ans)
    if "AGE5" in df.columns:
        age_risk = ((df["AGE5"] < 5) | (df["AGE5"] >= 65)).astype(float)
        score += age_risk * 0.5

    # Pas de diplôme
    if "a_diplome_eg" in df.columns:
        score += (1 - df["a_diplome_eg"]) * 0.5

    # Discrétisation en terciles
    df["score_vuln_indiv"] = score
    df["classe_vuln_indiv"] = pd.qcut(
        score, q=3, labels=[0, 1, 2], duplicates="drop"
    ).astype(int)

    labels = ["Non vulnérable", "Vulnérable", "Très vulnérable"]
    dist = df["classe_vuln_indiv"].value_counts().sort_index()
    for i, label in enumerate(labels):
        pct = dist.get(i, 0) / len(df) * 100
        print(f"       Classe {i} ({label}) : {dist.get(i,0):,} individus ({pct:.1f}%)")

    return df


# ──────────────────────────────────────────────────────────────
# 4. PRÉPARATION DES FEATURES
# ──────────────────────────────────────────────────────────────

def prepare_xgb_features(df: pd.DataFrame) -> tuple:
    """
    Sélectionne les features disponibles.
    Pas d'imputation : XGBoost gère les NaN nativement
    via son algorithme de split (learns the missing direction).
    """
    available = [c for c in XGB_FEATURES if c in df.columns]
    print(f"\n[XGB] Features disponibles : {len(available)}/{len(XGB_FEATURES)}")

    missing_feats = [c for c in XGB_FEATURES if c not in df.columns]
    if missing_feats:
        print(f"[XGB] Features absentes (ignorées) : {missing_feats}")

    X = df[available].copy()
    y = df["classe_vuln_indiv"].copy()

    # Statistiques NaN par feature (informatif)
    nan_pct = (X.isna().sum() / len(X) * 100).round(1)
    nan_feats = nan_pct[nan_pct > 0].sort_values(ascending=False)
    if not nan_feats.empty:
        print("\n[XGB] NaN transmis au modèle (gérés nativement) :")
        for feat, pct in nan_feats.items():
            print(f"       {feat:<25} {pct:.1f}% manquant")

    return X, y, available


# ──────────────────────────────────────────────────────────────
# 5. ENTRAÎNEMENT XGBOOST
# ──────────────────────────────────────────────────────────────

def get_scale_pos_weight(y: pd.Series) -> dict:
    """Calcule les poids par classe pour compenser le déséquilibre."""
    counts = y.value_counts().sort_index()
    max_count = counts.max()
    return {cls: max_count / cnt for cls, cnt in counts.items()}


def train_xgboost(X_train, y_train, n_classes: int = 3) -> XGBClassifier:
    """
    Entraîne XGBoost multiclasse avec hyperparamètres optimisés
    pour données sociodémographiques.
    """
    print("\n[XGB] Entraînement XGBoost (multiclasse softmax)...")

    # Calcul des poids de classe
    sample_weights = y_train.map(get_scale_pos_weight(y_train))

    xgb_model = XGBClassifier(
        # Architecture
        n_estimators=500,
        max_depth=6,
        min_child_weight=10,       # évite l'overfitting sur petits groupes

        # Learning rate + régularisation
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.1,             # L1 — induit la sparsité
        reg_lambda=1.0,            # L2

        # Multiclasse
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",

        # NaN handling : XGBoost choisit automatiquement
        # la direction de split pour les valeurs manquantes
        # → aucun paramètre à spécifier

        # Performance
        tree_method="hist",        # rapide sur CPU
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping_rounds=30,
        verbosity=0,
    )

    # Split interne pour early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1,
        stratify=y_train, random_state=RANDOM_STATE
    )
    sw_tr = y_tr.map(get_scale_pos_weight(y_train))

    xgb_model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best = xgb_model.best_iteration
    print(f"       Best iteration (early stopping) : {best}")
    return xgb_model


def tune_xgboost(X_train, y_train, n_iter: int = 20) -> XGBClassifier:
    """
    Recherche aléatoire d'hyperparamètres (optionnel — prend ~5 min).
    Appeler uniquement si on veut pousser la performance.
    """
    print("\n[XGB] Tuning hyperparamètres (RandomizedSearch)...")

    param_dist = {
        "max_depth":        [4, 5, 6, 7, 8],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "n_estimators":     [300, 400, 500],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [5, 10, 15, 20],
        "reg_alpha":        [0.0, 0.1, 0.5, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0],
    }

    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model, param_dist,
        n_iter=n_iter, cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"       Meilleur F1 : {search.best_score_:.4f}")
    print(f"       Meilleurs params : {search.best_params_}")
    return search.best_estimator_


# ──────────────────────────────────────────────────────────────
# 6. ÉVALUATION
# ──────────────────────────────────────────────────────────────

def evaluate_xgboost(model, X_test, y_test, feature_names: list) -> str:
    """Évaluation complète : métriques + importance + courbe ROC AUC."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC AUC multiclasse (One-vs-Rest)
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    try:
        auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    # Average Precision (PR AUC) — plus robuste si déséquilibre
    try:
        ap = average_precision_score(y_bin, y_proba, average="weighted")
    except Exception:
        ap = float("nan")

    labels_str = ["Non vulnérable (0)", "Vulnérable (1)", "Très vulnérable (2)"]

    # Importance des features (gain moyen)
    importance_gain   = model.get_booster().get_score(importance_type="gain")
    importance_cover  = model.get_booster().get_score(importance_type="cover")
    importance_weight = model.get_booster().get_score(importance_type="weight")

    imp_df = pd.DataFrame({
        "feature": list(importance_gain.keys()),
        "gain":    list(importance_gain.values()),
        "cover":   [importance_cover.get(f, 0) for f in importance_gain],
        "weight":  [importance_weight.get(f, 0) for f in importance_gain],
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    report_lines = [
        "=" * 65,
        "  RAPPORT D'ÉVALUATION — XGBoost",
        "  RGPH 2014 — Score Vulnérabilité Individuelle",
        "=" * 65,
        f"  Accuracy         : {acc:.4f}",
        f"  F1-Score         : {f1:.4f}   (weighted)",
        f"  ROC AUC (OvR)    : {auc:.4f}  (weighted)",
        f"  Avg Precision    : {ap:.4f}   (weighted)",
        f"  Best iteration   : {model.best_iteration}",
        "",
        "  RAPPORT DÉTAILLÉ PAR CLASSE :",
        classification_report(y_test, y_pred,
            target_names=labels_str, zero_division=0),
        "",
        "  IMPORTANCE DES FEATURES (par gain moyen) :",
        f"  {'Feature':<28} {'Gain':>10}  {'Cover':>10}  Barre",
    ]

    max_gain = imp_df["gain"].max() if not imp_df.empty else 1
    for _, row in imp_df.head(15).iterrows():
        bar = "█" * int(row["gain"] / max_gain * 30)
        report_lines.append(
            f"  {row['feature']:<28} {row['gain']:>10.1f}  "
            f"{row['cover']:>10.1f}  {bar}"
        )

    report = "\n".join(report_lines)
    print(report)
    return report, imp_df


# ──────────────────────────────────────────────────────────────
# 7. INTERPRÉTABILITÉ SHAP
# ──────────────────────────────────────────────────────────────

def compute_shap_summary(model, X_sample: pd.DataFrame,
                          feature_names: list) -> pd.DataFrame:
    """
    Calcule les valeurs SHAP pour interpréter le modèle.
    Retourne un DataFrame avec l'importance SHAP moyenne par feature.
    
    SHAP (SHapley Additive exPlanations) :
    - Explique la contribution de chaque feature à chaque prédiction
    - Utilisable pour justifier les scores devant le jury
    """
    try:
        import shap # type: ignore
        print("\n[SHAP] Calcul des valeurs SHAP (échantillon)...")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Pour multiclasse : shap_values est une liste de 3 matrices
        # On prend la valeur absolue moyenne sur toutes les classes
        if isinstance(shap_values, list):
            shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_abs = np.abs(shap_values)

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_mean_abs": shap_abs.mean(axis=0),
        }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)

        print("\n  SHAP — Top 10 features les plus influentes :")
        print(f"  {'Feature':<28} {'SHAP |mean|':>12}  Barre")
        max_shap = shap_df["shap_mean_abs"].max()
        for _, row in shap_df.head(10).iterrows():
            bar = "█" * int(row["shap_mean_abs"] / max_shap * 30)
            print(f"  {row['feature']:<28} {row['shap_mean_abs']:>12.4f}  {bar}")

        shap_df.to_csv(SHAP_PATH, index=False)
        print(f"\n[SAVE] Valeurs SHAP → {SHAP_PATH}")
        return shap_df

    except ImportError:
        print("[SHAP] Package 'shap' non installé — calcul ignoré.")
        print("       Installer avec : pip install shap")
        # Fallback : importance XGBoost native
        imp = model.get_booster().get_score(importance_type="gain")
        shap_df = pd.DataFrame(
            list(imp.items()), columns=["feature", "shap_mean_abs"]
        ).sort_values("shap_mean_abs", ascending=False)
        return shap_df


# ──────────────────────────────────────────────────────────────
# 8. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────

def run_xgb_pipeline(df: pd.DataFrame, tune: bool = False) -> pd.DataFrame:
    """
    Pipeline XGBoost complet.
    
    Args:
        df    : DataFrame post-ETL (depuis rgph_pipeline.py)
        tune  : Si True, lance la recherche d'hyperparamètres
    
    Returns:
        df avec colonnes score_vuln_indiv, classe_vuln_indiv,
        pred_xgb, proba_xgb_0/1/2 ajoutées
    """
    print("\n" + "="*65)
    print("  PHASE XGBoost — SCORING INDIVIDUEL")
    print("="*65)

    # Construction cible
    df = build_individual_target(df)

    # Préparation features
    X, y, features = prepare_xgb_features(df)

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n[XGB] Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")

    # Entraînement
    if tune:
        model = tune_xgboost(X_train, y_train)
    else:
        model = train_xgboost(X_train, y_train)

    # Évaluation
    report, imp_df = evaluate_xgboost(model, X_test, y_test, features)

    # SHAP sur un sous-échantillon (limiter le temps de calcul)
    n_shap = min(5_000, len(X_test))
    shap_df = compute_shap_summary(
        model,
        X_test.iloc[:n_shap],
        features
    )

    # Scores sur tout le dataset
    print("\n[XGB] Génération des scores sur l'ensemble du dataset...")
    y_all_pred  = model.predict(X)
    y_all_proba = model.predict_proba(X)

    df["pred_xgb"]      = y_all_pred
    df["proba_xgb_0"]   = y_all_proba[:, 0]  # P(Non vulnérable)
    df["proba_xgb_1"]   = y_all_proba[:, 1]  # P(Vulnérable)
    df["proba_xgb_2"]   = y_all_proba[:, 2]  # P(Très vulnérable)

    # Score continu normalisé [0,1] : utilité pour le dashboard
    df["score_continu"] = (
        df["proba_xgb_1"] * 0.5 + df["proba_xgb_2"] * 1.0
    ).clip(0, 1)

    # Sauvegarde rapport
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # Sauvegarde modèle
    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    print(f"[SAVE] Modèle + features → {MODEL_PATH}")

    # Export scores individuels
    export_cols = [
        "reg", "MEN_PRO", "NOR_MEN", "sexe", "AGE5", "mil",
        "score_vuln_indiv", "classe_vuln_indiv",
        "pred_xgb", "score_continu",
        "proba_xgb_0", "proba_xgb_1", "proba_xgb_2",
    ]
    export_cols = [c for c in export_cols if c in df.columns]
    df[export_cols].to_csv(SCORES_PATH, index=False)
    print(f"[SAVE] Scores individuels → {SCORES_PATH}")

    return df


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  RGPH 2014 — Score Vulnérabilité Individuelle XGBoost")
    print("="*65 + "\n")

    # Chargement + ETL (réutilise rgph_pipeline.py)
    df = load_and_etl(DATA_INDIVIDU)

    # Pipeline XGBoost
    df = run_xgb_pipeline(df, tune=False)

    print("\n" + "="*65)
    print("  XGBoost terminé ✓")
    print("="*65)
    return df


if __name__ == "__main__":
    df_result = main()
