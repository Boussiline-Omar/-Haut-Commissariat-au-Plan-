"""
=============================================================
  RGPH 2014 — Pipeline ETL + Classification Random Forest
  Projet : Système Analytique Intelligent (Soutenance)
=============================================================
Structure :
  1. Configuration & Constantes
  2. Chargement des données
  3. ETL — Nettoyage conditionnel (3 familles)
  4. Feature Engineering (indices synthétiques)
  5. Agrégation ménages
  6. Construction de la cible (proxy pauvreté)
  7. Modèle Random Forest — Entraînement & Évaluation
  8. Export des résultats
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ──────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────

# Chemins — adapter selon l'environnement
DATA_INDIVIDU  = "Individu.csv"
DATA_MENAGE    = "Menage.csv"        # fichier ménage si disponible
OUTPUT_DIR     = "outputs/"
MODEL_PATH     = os.path.join(OUTPUT_DIR, "rf_menage_classifier.pkl")
RESULTS_PATH   = os.path.join(OUTPUT_DIR, "menage_predictions.csv")
REPORT_PATH    = os.path.join(OUTPUT_DIR, "model_report.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── Taxonomie des variables conditionnelles ──────────────────

# Famille 1 : Fécondité (conditionnelle sexe/âge → jamais imputer)
FECONDITE_VARS = ["ENF_VIV", "ENF_DEC", "ENF_VIV_12M", "ENF_DEC_12M", "E_MAT"]

# Famille 2 : Emploi (conditionnel occupation → sentinel 0 = "non concerné")
EMPLOI_VARS = [
    "PROF_SGG", "PROF_GG", "STAT_PROF",
    "ACT_SECTION", "ACT_SECTEUR", "TRAV_LIEU", "TRAV_TRANS"
]

# Famille 3 : Scolarisation (conditionnel → indicateurs binaires synthétiques)
SCOLARITE_VARS = ["SEC_ENS", "ET_LIEU", "ET_TRANS"]

# Colonnes à exclure du pipeline ML (identifiants, poids, redondances)
DROP_COLS = [
    "MEN_PRO",       # identifiant ménage/province (clé de jointure)
    "NOR_MEN",       # numéro ordre dans ménage
    "pro",           # province → redondant avec reg pour ML
    "pds",           # poids de sondage → utilisé en inférence, pas en features
    "AGE1",          # doublon AGE5 (tranches larges vs exact)
    "EG_DIP_SGG",    # version détaillée du diplôme → garder EG_DIP_GG_DET
    "FP_DIP_SG",     # formation prof sous-groupe → garder FP_DIP_GG
    "FP_DIP_SGG",    # idem
    "LANG2", "LANG3",# langues secondaires/tertiaires très creuses
    "LANG_LOC2",     # langue locale 2 : 82% manquant
]

# ──────────────────────────────────────────────────────────────
# 2. CHARGEMENT
# ──────────────────────────────────────────────────────────────

def load_individu(path: str) -> pd.DataFrame:
    """Charge le fichier individus. Accepte CSV ou STATA."""
    print(f"[LOAD] Chargement : {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext in (".dta", ".sav"):
        import pyreadstat # type: ignore
        df, _ = pyreadstat.read_dta(path)
    else:
        raise ValueError(f"Format non supporté : {ext}")
    print(f"       {df.shape[0]:,} individus × {df.shape[1]} variables")
    return df


# ──────────────────────────────────────────────────────────────
# 3. ETL — NETTOYAGE CONDITIONNEL
# ──────────────────────────────────────────────────────────────

def etl_fecondite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Famille 1 — Fécondité.
    Ces variables sont conditionnelles au sexe (féminin) et à l'âge (15-49).
    Stratégie : NE PAS imputer individuellement.
    On crée un flag booléen de disponibilité + on agrègera au niveau ménage.
    """
    print("[ETL] Famille 1 : Fécondité → flags de disponibilité")
    for col in FECONDITE_VARS:
        if col in df.columns:
            df[f"has_{col}"] = df[col].notna().astype(int)
    return df


def etl_emploi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Famille 2 — Emploi.
    Conditionnel à TY_ACT (type d'activité).
    Sentinel 0 = « non concerné » (inactif, enfant, retraité…).
    XGBoost gère nativement les NaN, mais on garde le sentinel
    pour Random Forest et pour la logique métier.
    """
    print("[ETL] Famille 2 : Emploi → sentinel 0")
    # Personnes économiquement actives : TY_ACT in {0,1,2}
    # Inactives : TY_ACT in {3,4,5} ou NaN
    actif_mask = df["TY_ACT"].isin([0, 1, 2])
    for col in EMPLOI_VARS:
        if col in df.columns:
            df[col] = df[col].where(actif_mask, other=0).fillna(0)
    return df


def etl_scolarite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Famille 3 — Scolarisation.
    Conditionnel à scol (actuellement scolarisé) et NIV_ET.
    On dérive des indicateurs binaires plutôt que d'imputer les valeurs brutes.
    """
    print("[ETL] Famille 3 : Scolarisation → indicateurs binaires")

    # a_diplome : possède un diplôme d'enseignement général
    if "EG_DIP_GG_DET" in df.columns:
        df["a_diplome_eg"] = df["EG_DIP_GG_DET"].notna().astype(int)

    # a_formation_pro : possède une formation professionnelle
    if "FP_DIP_GG" in df.columns:
        df["a_formation_pro"] = df["FP_DIP_GG"].notna().astype(int)

    # scolarise_actuel : actuellement scolarisé
    if "scol" in df.columns:
        df["scolarise_actuel"] = (df["scol"] == 1).astype(int)

    # navette_scolaire : se déplace pour étudier
    if "ET_LIEU" in df.columns:
        df["navette_scolaire"] = df["ET_LIEU"].notna().astype(int)

    return df


def etl_autres(df: pd.DataFrame) -> pd.DataFrame:
    """Variables générales : imputation par mode groupé ou médiane."""
    print("[ETL] Variables générales → imputation résiduelle")

    # LIR_ECR : alphabétisme — mode par groupe (reg, mil, sexe)
    if "LIR_ECR" in df.columns:
        group_mode = (
            df.groupby(["reg", "mil", "sexe"])["LIR_ECR"]
            .transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        )
        df["LIR_ECR"] = df["LIR_ECR"].fillna(group_mode)
        # Fallback global si le groupe est entièrement NaN
        df["LIR_ECR"] = df["LIR_ECR"].fillna(df["LIR_ECR"].mode()[0])

    # AGE5 : âge en tranches de 5 ans — médiane par région/milieu
    if "AGE5" in df.columns:
        df["AGE5"] = df["AGE5"].fillna(
            df.groupby(["reg", "mil"])["AGE5"].transform("median")
        )

    # Variables géographiques résiduelles
    for col in ["reg", "mil", "LIEN_CM", "natio"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def run_etl(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline ETL complet."""
    print("\n" + "="*60)
    print("  PHASE ETL")
    print("="*60)
    df = etl_fecondite(df)
    df = etl_emploi(df)
    df = etl_scolarite(df)
    df = etl_autres(df)
    print(f"[ETL] Terminé. Shape : {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING — INDICES SYNTHÉTIQUES
# ──────────────────────────────────────────────────────────────

def feature_engineering_individu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule des indices composites au niveau individu,
    avant agrégation ménage.
    """
    print("\n[FE] Feature engineering individu")

    # ── Indice de scolarisation (0–1) ────────────────────────
    # Composantes : niveau d'éducation, alphabétisme, diplôme
    score_edu = pd.Series(0.0, index=df.index)

    if "NIV_ET_AGR" in df.columns:
        # NIV_ET_AGR : 0=aucun, 1=préscolaire, 2=primaire,
        #              3=secondaire collège, 4=secondaire lycée,
        #              5=supérieur, 6=formation pro
        score_edu += df["NIV_ET_AGR"].clip(0, 5) / 5 * 0.5

    if "LIR_ECR" in df.columns:
        # LIR_ECR : 1=lit et écrit, 2=lit seulement, 3=ne sait pas
        lir_score = df["LIR_ECR"].map({1: 1.0, 2: 0.5, 3: 0.0}).fillna(0)
        score_edu += lir_score * 0.3

    if "a_diplome_eg" in df.columns:
        score_edu += df["a_diplome_eg"] * 0.2

    df["indice_scolarisation"] = score_edu.clip(0, 1)

    # ── Score d'activité économique ──────────────────────────
    # TY_ACT : 0=occupé, 1=chômeur cherchant 1er emploi,
    #          2=chômeur ayant travaillé, 3=femme au foyer,
    #          4=étudiant, 5=autre inactif
    if "TY_ACT" in df.columns:
        df["est_occupe"] = (df["TY_ACT"] == 0).astype(int)
        df["est_chomeur"] = df["TY_ACT"].isin([1, 2]).astype(int)

    # ── Indicateur de handicap global ────────────────────────
    handi_cols = [c for c in df.columns if c.startswith("HANDI_") and c != "HANDI_ENTR"]
    if handi_cols:
        # Au moins un handicap sévère (valeur > 1 dans les colonnes HANDI)
        df["a_handicap"] = (df[handi_cols].max(axis=1) > 1).astype(int)

    return df


# ──────────────────────────────────────────────────────────────
# 5. AGRÉGATION MÉNAGES
# ──────────────────────────────────────────────────────────────

def aggregate_menage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège le fichier individus au niveau ménage.
    Clé de ménage : (reg, MEN_PRO) — identifie un ménage dans sa province.
    
    Retourne un DataFrame à grain ménage.
    """
    print("\n" + "="*60)
    print("  PHASE AGRÉGATION MÉNAGES")
    print("="*60)

    men_key = ["reg", "MEN_PRO"]

    agg_dict = {}

    # ── Taille & composition du ménage ───────────────────────
    agg_dict["taille_menage"]     = ("NOR_MEN", "max")
    agg_dict["nb_femmes"]         = ("sexe", lambda x: (x == 2).sum())
    agg_dict["nb_mineurs"]        = ("AGE5", lambda x: (x < 15).sum())
    agg_dict["nb_seniors"]        = ("AGE5", lambda x: (x >= 60).sum())
    agg_dict["age_moyen"]         = ("AGE5", "mean")

    # ── Géographie ───────────────────────────────────────────
    agg_dict["milieu"]            = ("mil", "first")   # 1=urbain, 2=rural
    agg_dict["province"]          = ("pro", "first")

    # ── Éducation ────────────────────────────────────────────
    agg_dict["indice_scol_moy"]   = ("indice_scolarisation", "mean")
    agg_dict["nb_diplomes"]       = ("a_diplome_eg", "sum")
    agg_dict["nb_scolarises"]     = ("scolarise_actuel", "sum")
    if "a_formation_pro" in df.columns:
        agg_dict["nb_formes_pro"] = ("a_formation_pro", "sum")

    # ── Emploi ───────────────────────────────────────────────
    agg_dict["nb_occupes"]        = ("est_occupe", "sum")
    agg_dict["nb_chomeurs"]       = ("est_chomeur", "sum")
    agg_dict["taux_activite"]     = ("est_occupe",
        lambda x: x.sum() / len(x) if len(x) > 0 else 0)

    # ── Fécondité (agrégé proprement) ────────────────────────
    if "ENF_VIV" in df.columns:
        agg_dict["total_enf_viv"] = ("ENF_VIV", "sum")
    if "ENF_DEC" in df.columns:
        agg_dict["total_enf_dec"] = ("ENF_DEC", "sum")

    # ── Handicap ─────────────────────────────────────────────
    agg_dict["nb_handicapes"]     = ("a_handicap", "sum")

    # ── Nationalité (proportion marocains) ───────────────────
    agg_dict["prop_marocains"]    = ("natio", lambda x: (x == 1).mean())

    # Construction du DataFrame ménage
    print(f"[AGG] Agrégation de {df.shape[0]:,} individus...")
    df_agg = df.groupby(men_key).agg(**agg_dict).reset_index()

    # ── Features dérivées post-agrégation ────────────────────
    df_agg["ratio_dependance"] = (
        (df_agg["nb_mineurs"] + df_agg["nb_seniors"])
        / df_agg["taille_menage"].replace(0, np.nan)
    ).fillna(0)

    df_agg["ratio_femmes"] = (
        df_agg["nb_femmes"] / df_agg["taille_menage"].replace(0, np.nan)
    ).fillna(0)

    df_agg["ratio_emploi"] = (
        df_agg["nb_occupes"] / df_agg["taille_menage"].replace(0, np.nan)
    ).fillna(0)

    if "total_enf_dec" in df_agg.columns and "total_enf_viv" in df_agg.columns:
        total = df_agg["total_enf_viv"] + df_agg["total_enf_dec"]
        df_agg["taux_mortalite_infantile"] = (
            df_agg["total_enf_dec"] / total.replace(0, np.nan)
        ).fillna(0)

    print(f"[AGG] {df_agg.shape[0]:,} ménages × {df_agg.shape[1]} features")
    return df_agg


# ──────────────────────────────────────────────────────────────
# 6. CONSTRUCTION DE LA CIBLE (proxy vulnérabilité)
# ──────────────────────────────────────────────────────────────

def build_target(df_men: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un score de vulnérabilité proxy en l'absence de revenus déclarés.
    
    Score composite basé sur :
      - Milieu (rural = plus vulnérable)
      - Ratio de dépendance démographique
      - Niveau d'éducation moyen
      - Taux d'emploi
      - Mortalité infantile (proxy conditions sanitaires)
      - Présence de handicap
    
    Seuil : terciles → 0=Non vulnérable, 1=Vulnérable, 2=Très vulnérable
    
    ⚠️  NOTE ACADÉMIQUE : Ce score est un proxy construit.
    En production, il serait remplacé par des données de revenus
    ou un indice officiel (HCP/Banque Mondiale).
    """
    print("\n[TARGET] Construction du score de vulnérabilité proxy")

    score = pd.Series(0.0, index=df_men.index)

    # Milieu rural (poids fort)
    if "milieu" in df_men.columns:
        score += (df_men["milieu"] == 2).astype(float) * 2.0

    # Ratio de dépendance (haut = vulnérable)
    if "ratio_dependance" in df_men.columns:
        score += df_men["ratio_dependance"].clip(0, 1) * 2.0

    # Éducation (inverse : faible éducation = vulnérable)
    if "indice_scol_moy" in df_men.columns:
        score += (1 - df_men["indice_scol_moy"].clip(0, 1)) * 2.0

    # Chômage
    if "nb_chomeurs" in df_men.columns and "taille_menage" in df_men.columns:
        ratio_chom = df_men["nb_chomeurs"] / df_men["taille_menage"].replace(0, np.nan)
        score += ratio_chom.fillna(0).clip(0, 1) * 1.5

    # Absence d'emploi
    if "ratio_emploi" in df_men.columns:
        score += (1 - df_men["ratio_emploi"].clip(0, 1)) * 1.5

    # Mortalité infantile
    if "taux_mortalite_infantile" in df_men.columns:
        score += df_men["taux_mortalite_infantile"].clip(0, 1) * 1.0

    # Handicap
    if "nb_handicapes" in df_men.columns and "taille_menage" in df_men.columns:
        ratio_handi = df_men["nb_handicapes"] / df_men["taille_menage"].replace(0, np.nan)
        score += ratio_handi.fillna(0).clip(0, 1) * 0.5

    # Discrétisation en terciles
    labels = ["Non vulnérable", "Vulnérable", "Très vulnérable"]
    df_men["score_vulnerabilite"] = score
    df_men["classe_vulnerabilite"] = pd.qcut(
        score, q=3, labels=[0, 1, 2], duplicates="drop"
    ).astype(int)

    dist = df_men["classe_vulnerabilite"].value_counts().sort_index()
    for i, label in enumerate(labels):
        pct = dist.get(i, 0) / len(df_men) * 100
        print(f"       Classe {i} ({label}) : {dist.get(i, 0):,} ménages ({pct:.1f}%)")

    return df_men


# ──────────────────────────────────────────────────────────────
# 7. MODÈLE RANDOM FOREST
# ──────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "milieu", "province",
    "taille_menage", "nb_femmes", "nb_mineurs", "nb_seniors",
    "age_moyen", "ratio_dependance", "ratio_femmes",
    "indice_scol_moy", "nb_diplomes", "nb_scolarises",
    "nb_occupes", "nb_chomeurs", "taux_activite", "ratio_emploi",
    "nb_handicapes",
    "prop_marocains",
]


def prepare_features(df_men: pd.DataFrame) -> tuple:
    """Sélectionne et prépare les features disponibles."""
    available = [c for c in FEATURE_COLS if c in df_men.columns]
    print(f"\n[RF] Features utilisées ({len(available)}) : {available}")

    X = df_men[available].copy()
    y = df_men["classe_vulnerabilite"].copy()

    # Imputation résiduelle (ne devrait pas y en avoir à ce stade)
    X = X.fillna(X.median(numeric_only=True))

    return X, y, available


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Entraîne le Random Forest avec les hyperparamètres recommandés."""
    print("\n[RF] Entraînement du Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",   # compense les déséquilibres de classes
        n_jobs=-1,
        random_state=RANDOM_STATE,
        oob_score=True,            # Out-of-Bag estimate gratuit
    )
    rf.fit(X_train, y_train)
    print(f"       OOB Score : {rf.oob_score_:.4f}")
    return rf


def evaluate_model(rf, X_test, y_test, feature_names: list) -> str:
    """Évalue le modèle et retourne un rapport texte."""
    y_pred = rf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    labels_str = ["Non vulnérable (0)", "Vulnérable (1)", "Très vulnérable (2)"]

    report_lines = [
        "=" * 60,
        "  RAPPORT D'ÉVALUATION — Random Forest",
        "  RGPH 2014 — Classification Ménages",
        "=" * 60,
        f"  Accuracy   : {acc:.4f}",
        f"  Precision  : {prec:.4f}  (weighted)",
        f"  Recall     : {rec:.4f}  (weighted)",
        f"  F1-Score   : {f1:.4f}  (weighted)",
        f"  OOB Score  : {rf.oob_score_:.4f}",
        "",
        "  RAPPORT DÉTAILLÉ PAR CLASSE :",
        classification_report(y_test, y_pred,
            target_names=labels_str, zero_division=0),
        "",
        "  IMPORTANCE DES FEATURES (Top 10) :",
    ]

    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top10 = importances.nlargest(10)
    for feat, imp in top10.items():
        bar = "█" * int(imp * 50)
        report_lines.append(f"  {feat:<30} {imp:.4f}  {bar}")

    report = "\n".join(report_lines)
    print(report)
    return report


def cross_validate_rf(X, y) -> None:
    """Validation croisée stratifiée 5-fold."""
    print("\n[RF] Validation croisée 5-fold...")
    rf_cv = RandomForestClassifier(
        n_estimators=100, max_depth=12,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(rf_cv, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"       F1 weighted — mean: {scores.mean():.4f} | std: {scores.std():.4f}")
    print(f"       Scores par fold : {np.round(scores, 4)}")


def run_ml_pipeline(df_men: pd.DataFrame) -> None:
    """Pipeline ML complet : split → train → evaluate → save."""
    print("\n" + "="*60)
    print("  PHASE MODÉLISATION — RANDOM FOREST")
    print("="*60)

    X, y, features = prepare_features(df_men)

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )
    print(f"[RF] Train : {X_train.shape[0]:,} ménages | Test : {X_test.shape[0]:,} ménages")

    # Entraînement
    rf = train_random_forest(X_train, y_train)

    # Validation croisée
    cross_validate_rf(X, y)

    # Évaluation sur le test set
    report = evaluate_model(rf, X_test, y_test, features)

    # Sauvegarde du rapport
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[SAVE] Rapport sauvegardé → {REPORT_PATH}")

    # Sauvegarde du modèle
    joblib.dump(rf, MODEL_PATH)
    print(f"[SAVE] Modèle sauvegardé → {MODEL_PATH}")

    # Prédictions sur tout le dataset ménage
    y_all_pred  = rf.predict(X)
    y_all_proba = rf.predict_proba(X)

    df_men["pred_classe"]       = y_all_pred
    df_men["proba_non_vuln"]    = y_all_proba[:, 0]
    df_men["proba_vuln"]        = y_all_proba[:, 1]
    df_men["proba_tres_vuln"]   = y_all_proba[:, 2]

    # Export résultats
    export_cols = ["reg", "MEN_PRO", "milieu", "province",
                   "taille_menage", "score_vulnerabilite",
                   "classe_vulnerabilite", "pred_classe",
                   "proba_non_vuln", "proba_vuln", "proba_tres_vuln"]
    export_cols = [c for c in export_cols if c in df_men.columns]
    df_men[export_cols].to_csv(RESULTS_PATH, index=False)
    print(f"[SAVE] Prédictions sauvegardées → {RESULTS_PATH}")


# ──────────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  RGPH 2014 — Système Analytique Intelligent")
    print("  Pipeline ETL + Classification Random Forest")
    print("="*60 + "\n")

    # ── Chargement ───────────────────────────────────────────
    if not os.path.exists(DATA_INDIVIDU):
        print(f"[WARN] Fichier '{DATA_INDIVIDU}' introuvable.")
        print("       Génération d'un dataset synthétique pour tests...\n")
        df = generate_synthetic_data(n=50_000)
    else:
        df = load_individu(DATA_INDIVIDU)

    # ── ETL ──────────────────────────────────────────────────
    df = run_etl(df)

    # ── Feature engineering individu ─────────────────────────
    df = feature_engineering_individu(df)

    # ── Agrégation ménages ───────────────────────────────────
    df_men = aggregate_menage(df)

    # ── Construction cible ───────────────────────────────────
    df_men = build_target(df_men)

    # ── Modélisation ─────────────────────────────────────────
    run_ml_pipeline(df_men)

    print("\n" + "="*60)
    print("  Pipeline terminé avec succès ✓")
    print("="*60)


# ──────────────────────────────────────────────────────────────
# UTILITAIRE : Données synthétiques pour tests sans fichier
# ──────────────────────────────────────────────────────────────

def generate_synthetic_data(n: int = 50_000) -> pd.DataFrame:
    """Génère un dataset synthétique qui mime la structure RGPH."""
    np.random.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    # On crée ~10 000 ménages de taille variable
    n_menages = n // 5
    menage_ids = np.arange(1, n_menages + 1)

    menage_reg = rng.integers(1, 13, size=n_menages)
    menage_pro = menage_reg * 10 + rng.integers(1, 8, size=n_menages)
    menage_mil = rng.choice([1, 2], size=n_menages, p=[0.6, 0.4])
    menage_size = rng.integers(1, 10, size=n_menages)

    rows = []
    for i in range(n_menages):
        sz = menage_size[i]
        for j in range(sz):
            age5 = int(rng.integers(0, 16) * 5)
            sexe = rng.choice([1, 2])
            ty_act = rng.choice([0,1,2,3,4,5], p=[0.35,0.05,0.05,0.2,0.15,0.2])

            row = {
                "reg": menage_reg[i], "pro": menage_pro[i],
                "mil": menage_mil[i], "MEN_PRO": menage_ids[i],
                "NOR_MEN": j + 1, "LIEN_CM": rng.choice([0,1,2,3,4,9]),
                "natio": rng.choice([1,2,3,4], p=[0.96,0.02,0.01,0.01]),
                "sexe": sexe, "AGE1": None if age5 > 14 else int(age5/5),
                "AGE5": age5,
                "E_MAT": rng.choice([1,2,3,4,5,np.nan], p=[0.4,0.3,0.1,0.1,0.05,0.05]) if sexe==2 and 15<=age5<50 else np.nan,
                "ENF_VIV": rng.integers(0,6) if sexe==2 and 15<=age5<50 else np.nan,
                "ENF_DEC": rng.integers(0,3) if sexe==2 and 15<=age5<50 else np.nan,
                "ENF_VIV_12M": rng.integers(0,2) if sexe==2 and 15<=age5<50 else np.nan,
                "ENF_DEC_12M": rng.integers(0,2) if sexe==2 and 15<=age5<50 else np.nan,
                "HANDI_VIS": rng.choice([1,2,3,4], p=[0.85,0.08,0.04,0.03]),
                "HANDI_AUD": rng.choice([1,2,3,4], p=[0.88,0.07,0.03,0.02]),
                "HANDI_MOB": rng.choice([1,2,3,4], p=[0.87,0.08,0.03,0.02]),
                "HANDI_MEM": rng.choice([1,2,3,4], p=[0.9,0.06,0.03,0.01]),
                "HANDI_ENTR": rng.choice([1,2,3,4], p=[0.9,0.06,0.03,0.01]),
                "HANDI_COM": rng.choice([1,2,3,4], p=[0.91,0.05,0.03,0.01]),
                "SIT_HANDICAP": rng.integers(1,5),
                "NIV_ET": rng.integers(0,7),
                "NIV_ET_AGR": rng.integers(0,6),
                "SEC_ENS": rng.choice([1,2,3,np.nan], p=[0.3,0.3,0.1,0.3]) if age5>=6 else np.nan,
                "scol": rng.choice([1,2,np.nan], p=[0.25,0.25,0.5]),
                "ET_LIEU": rng.choice([1,2,3,np.nan], p=[0.1,0.1,0.05,0.75]),
                "ET_TRANS": rng.choice([0,1,2,np.nan], p=[0.1,0.1,0.05,0.75]),
                "LIR_ECR": rng.choice([1,2,3,np.nan], p=[0.5,0.1,0.2,0.2]) if age5>=10 else np.nan,
                "LANG1": rng.choice([1,2,3,4,5,np.nan], p=[0.5,0.2,0.15,0.05,0.05,0.05]),
                "LANG_LOC1": rng.choice([1,2,3,np.nan], p=[0.3,0.2,0.1,0.4]),
                "LANG_LOC2": rng.choice([1,2,np.nan], p=[0.1,0.05,0.85]),
                "LANG2": rng.choice([1,2,3,np.nan], p=[0.1,0.1,0.1,0.7]),
                "LANG3": rng.choice([1,2,np.nan], p=[0.05,0.05,0.9]),
                "EG_DIP_GG_DET": rng.choice([1,2,3,4,5,np.nan], p=[0.1,0.1,0.1,0.05,0.05,0.6]),
                "FP_DIP_GG": rng.choice([1,2,np.nan], p=[0.05,0.05,0.9]),
                "TY_ACT": ty_act if age5>=15 else np.nan,
                "PROF_SGG": rng.integers(11,100) if ty_act==0 else np.nan,
                "PROF_GG": rng.integers(1,10) if ty_act==0 else np.nan,
                "STAT_PROF": rng.integers(1,7) if ty_act==0 else np.nan,
                "ACT_SECTION": rng.integers(1,22) if ty_act==0 else np.nan,
                "ACT_SECTEUR": rng.integers(1,9) if ty_act==0 else np.nan,
                "TRAV_LIEU": rng.integers(0,7) if ty_act==0 else np.nan,
                "TRAV_TRANS": rng.integers(0,9) if ty_act==0 else np.nan,
                "pds": 9.999995,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[SYNTH] Dataset synthétique : {df.shape[0]:,} individus × {df.shape[1]} colonnes")
    return df


if __name__ == "__main__":
    main()
