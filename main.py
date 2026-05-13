"""
=============================================================
  RGPH 2014 — Backend FastAPI
  Projet : Système Analytique Intelligent (Soutenance)
=============================================================
Endpoints :
  GET  /                          → health check
  GET  /regions                   → liste des 12 régions
  POST /predict/menage            → Random Forest (classification ménage)
  POST /predict/individu          → XGBoost (score vulnérabilité individuelle)
  GET  /segmentation/regions      → K-Means (clusters régionaux)
  GET  /stats/region/{reg_id}     → profil statistique d'une région
  GET  /stats/dashboard           → KPIs globaux pour le dashboard
=============================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RGPH 2014 — API Analytique",
    description="Système d'analyse intelligente du recensement marocain 2014",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # En prod : restreindre au domaine React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────

MODELS_DIR = os.environ.get("MODELS_DIR", "outputs/")

REGION_NAMES = {
    1:  "Tanger-Tétouan-Al Hoceïma",
    2:  "Oriental",
    3:  "Fès-Meknès",
    4:  "Rabat-Salé-Kénitra",
    5:  "Béni Mellal-Khénifra",
    6:  "Casablanca-Settat",
    7:  "Marrakech-Safi",
    8:  "Drâa-Tafilalet",
    9:  "Souss-Massa",
    10: "Guelmim-Oued Noun",
    11: "Laâyoune-Sakia El Hamra",
    12: "Dakhla-Oued Ed-Dahab",
}

VULN_LABELS = {
    0: "Non vulnérable",
    1: "Vulnérable",
    2: "Très vulnérable",
}

# ──────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES (au démarrage)
# ──────────────────────────────────────────────────────────────

class ModelStore:
    rf_model      = None   # Random Forest ménage
    xgb_bundle    = None   # XGBoost individu {model, features}
    kmeans_bundle = None   # K-Means {kmeans, scaler, features, cluster_names}
    regional_df   = None   # Profils régionaux agrégés
    predictions_df = None  # Scores individus

model_store = ModelStore()


def load_models():
    """Charge les modèles ML au démarrage de l'application."""
    rf_path      = os.path.join(MODELS_DIR, "rf_menage_classifier.pkl")
    xgb_path     = os.path.join(MODELS_DIR, "xgb_individu_scorer.pkl")
    kmeans_path  = os.path.join(MODELS_DIR, "kmeans_regional.pkl")
    profiles_path = os.path.join(MODELS_DIR, "regional_clusters.csv")
    scores_path  = os.path.join(MODELS_DIR, "individu_scores.csv")

    if os.path.exists(rf_path):
        model_store.rf_model = joblib.load(rf_path)
        print(f"[API] ✓ Random Forest chargé")
    else:
        print(f"[API] ⚠ Random Forest non trouvé ({rf_path})")

    if os.path.exists(xgb_path):
        model_store.xgb_bundle = joblib.load(xgb_path)
        print(f"[API] ✓ XGBoost chargé")
    else:
        print(f"[API] ⚠ XGBoost non trouvé ({xgb_path})")

    if os.path.exists(kmeans_path):
        model_store.kmeans_bundle = joblib.load(kmeans_path)
        print(f"[API] ✓ K-Means chargé")
    else:
        print(f"[API] ⚠ K-Means non trouvé ({kmeans_path})")

    if os.path.exists(profiles_path):
        model_store.regional_df = pd.read_csv(profiles_path)
        print(f"[API] ✓ Profils régionaux chargés ({len(model_store.regional_df)} régions)")
    else:
        print(f"[API] ⚠ Profils régionaux non trouvés — génération mock")
        model_store.regional_df = _mock_regional_profiles()

    if os.path.exists(scores_path):
        model_store.predictions_df = pd.read_csv(scores_path)
        print(f"[API] ✓ Scores individus chargés ({len(model_store.predictions_df):,} lignes)")


def _mock_regional_profiles() -> pd.DataFrame:
    """Données mock pour les tests sans modèles entraînés."""
    rng = np.random.default_rng(42)
    rows = []
    for reg_id, reg_name in REGION_NAMES.items():
        rows.append({
            "reg": reg_id, "region_label": reg_name,
            "profil_label": reg_name,
            "population": rng.integers(100_000, 700_000),
            "age_moyen": round(rng.uniform(22, 35), 2),
            "pct_mineurs": round(rng.uniform(20, 40), 2),
            "pct_seniors": round(rng.uniform(5, 15), 2),
            "pct_femmes": round(rng.uniform(48, 52), 2),
            "niv_edu_moyen": round(rng.uniform(1.5, 4.0), 2),
            "indice_scol_moyen": round(rng.uniform(0.3, 0.7), 2),
            "pct_diplomes": round(rng.uniform(0.1, 0.5), 2),
            "pct_alphabete": round(rng.uniform(0.5, 0.9), 2),
            "taux_emploi": round(rng.uniform(0.2, 0.5), 3),
            "taux_chomage": round(rng.uniform(0.05, 0.2), 3),
            "pct_handicap": round(rng.uniform(0.03, 0.12), 3),
            "ratio_dependance": round(rng.uniform(0.3, 0.6), 3),
            "taux_mortalite_proxy": round(rng.uniform(0.01, 0.08), 4),
            "cluster": rng.integers(0, 4),
            "cluster_label": rng.choice([
                "Cluster 0 — Développé & Actif",
                "Cluster 1 — En Transition",
                "Cluster 2 — Vulnérable & Dépendant",
                "Cluster 3 — Rural Traditionnel",
            ]),
        })
    return pd.DataFrame(rows)


@app.on_event("startup")
async def startup_event():
    load_models()


# ──────────────────────────────────────────────────────────────
# SCHÉMAS PYDANTIC
# ──────────────────────────────────────────────────────────────

class MenageInput(BaseModel):
    """Profil agrégé d'un ménage pour la classification RF."""
    milieu: int = Field(..., ge=1, le=2, description="1=Urbain, 2=Rural")
    province: Optional[float] = Field(None, description="Code province")
    taille_menage: int = Field(..., ge=1, le=20)
    nb_femmes: int = Field(..., ge=0)
    nb_mineurs: int = Field(..., ge=0, description="Membres < 15 ans")
    nb_seniors: int = Field(..., ge=0, description="Membres >= 60 ans")
    age_moyen: float = Field(..., ge=0, le=100)
    nb_diplomes: int = Field(..., ge=0)
    nb_scolarises: int = Field(..., ge=0)
    nb_occupes: int = Field(..., ge=0)
    nb_chomeurs: int = Field(..., ge=0)
    nb_handicapes: int = Field(0, ge=0)
    indice_scol_moy: float = Field(..., ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "milieu": 1, "province": 141, "taille_menage": 5,
                "nb_femmes": 2, "nb_mineurs": 2, "nb_seniors": 0,
                "age_moyen": 28.5, "nb_diplomes": 2, "nb_scolarises": 2,
                "nb_occupes": 2, "nb_chomeurs": 0, "nb_handicapes": 0,
                "indice_scol_moy": 0.55
            }
        }


class IndividuInput(BaseModel):
    """Profil individuel pour le scoring XGBoost."""
    reg: int = Field(..., ge=1, le=12, description="Code région (1-12)")
    mil: int = Field(..., ge=1, le=2, description="1=Urbain, 2=Rural")
    sexe: int = Field(..., ge=1, le=2, description="1=Homme, 2=Femme")
    age5: float = Field(..., ge=0, le=75, description="Âge en tranches de 5 ans")
    niv_et_agr: Optional[int] = Field(None, ge=0, le=6, description="Niveau éducation agrégé")
    lir_ecr: Optional[int] = Field(None, ge=1, le=3, description="1=lit&écrit, 2=lit, 3=ne sait pas")
    ty_act: Optional[int] = Field(None, ge=0, le=5, description="Type d'activité")
    sit_handicap: Optional[int] = Field(None, ge=1, le=4, description="1=sans, 4=sévère")
    enf_viv: Optional[float] = Field(None, ge=0, description="Enfants vivants (femmes 15-49)")
    enf_dec: Optional[float] = Field(None, ge=0, description="Enfants décédés (femmes 15-49)")

    class Config:
        json_schema_extra = {
            "example": {
                "reg": 6, "mil": 1, "sexe": 1, "age5": 30,
                "niv_et_agr": 4, "lir_ecr": 1, "ty_act": 0,
                "sit_handicap": 1, "enf_viv": None, "enf_dec": None
            }
        }


# ──────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Système"])
async def health_check():
    """Health check — état des modèles chargés."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "project": "RGPH 2014 — Système Analytique Intelligent",
        "models": {
            "random_forest": model_store.rf_model is not None,
            "xgboost":       model_store.xgb_bundle is not None,
            "kmeans":        model_store.kmeans_bundle is not None,
        }
    }


@app.get("/regions", tags=["Référentiel"])
async def get_regions():
    """Retourne la liste des 12 régions marocaines."""
    return {
        "count": len(REGION_NAMES),
        "regions": [
            {"id": k, "nom": v} for k, v in REGION_NAMES.items()
        ]
    }


# ── Random Forest : classification ménage ────────────────────

@app.post("/predict/menage", tags=["Prédiction"])
async def predict_menage(data: MenageInput):
    """
    Classifie un ménage en 3 niveaux de vulnérabilité.
    Utilise le modèle Random Forest entraîné sur les profils ménage RGPH.
    
    Retourne :
    - classe : 0 (non vulnérable), 1 (vulnérable), 2 (très vulnérable)
    - label  : libellé de la classe
    - probas : probabilités par classe
    - score  : score continu [0,1]
    """
    if model_store.rf_model is None:
        raise HTTPException(503, "Modèle Random Forest non disponible")

    # Construction du vecteur de features dans l'ordre du modèle
    rf = model_store.rf_model
    feature_names = rf.feature_names_in_ if hasattr(rf, "feature_names_in_") else [
        "milieu", "province", "taille_menage", "nb_femmes", "nb_mineurs",
        "nb_seniors", "age_moyen", "ratio_dependance", "ratio_femmes",
        "indice_scol_moy", "nb_diplomes", "nb_scolarises",
        "nb_occupes", "nb_chomeurs", "taux_activite", "ratio_emploi",
        "nb_handicapes", "prop_marocains",
    ]

    # Calcul des features dérivées
    taille = max(data.taille_menage, 1)
    features = {
        "milieu":           data.milieu,
        "province":         data.province or 0,
        "taille_menage":    data.taille_menage,
        "nb_femmes":        data.nb_femmes,
        "nb_mineurs":       data.nb_mineurs,
        "nb_seniors":       data.nb_seniors,
        "age_moyen":        data.age_moyen,
        "ratio_dependance": (data.nb_mineurs + data.nb_seniors) / taille,
        "ratio_femmes":     data.nb_femmes / taille,
        "indice_scol_moy":  data.indice_scol_moy,
        "nb_diplomes":      data.nb_diplomes,
        "nb_scolarises":    data.nb_scolarises,
        "nb_occupes":       data.nb_occupes,
        "nb_chomeurs":      data.nb_chomeurs,
        "taux_activite":    data.nb_occupes / taille,
        "ratio_emploi":     data.nb_occupes / taille,
        "nb_handicapes":    data.nb_handicapes,
        "prop_marocains":   1.0,  # défaut marocain
    }

    X = pd.DataFrame([{f: features.get(f, 0) for f in feature_names}])
    classe  = int(rf.predict(X)[0])
    probas  = rf.predict_proba(X)[0].tolist()
    score   = round(probas[1] * 0.5 + probas[2] * 1.0, 4)

    return {
        "classe": classe,
        "label":  VULN_LABELS[classe],
        "probas": {
            "non_vulnerable": round(probas[0], 4),
            "vulnerable":     round(probas[1], 4),
            "tres_vulnerable": round(probas[2], 4),
        },
        "score_continu": score,
        "model": "RandomForest",
    }


# ── XGBoost : scoring individuel ─────────────────────────────

@app.post("/predict/individu", tags=["Prédiction"])
async def predict_individu(data: IndividuInput):
    """
    Calcule le score de vulnérabilité d'un individu.
    Utilise XGBoost qui gère nativement les valeurs manquantes.
    
    Retourne :
    - classe        : 0/1/2
    - label         : libellé
    - probas        : probabilités par classe
    - score_continu : [0,1] utilisable pour la heatmap
    """
    if model_store.xgb_bundle is None:
        raise HTTPException(503, "Modèle XGBoost non disponible")

    xgb_model = model_store.xgb_bundle["model"]
    feat_names = model_store.xgb_bundle["features"]

    # Features synthétiques dérivées
    est_occupe  = int(data.ty_act == 0) if data.ty_act is not None else np.nan
    est_chomeur = int(data.ty_act in [1, 2]) if data.ty_act is not None else np.nan

    sit_h = data.sit_handicap or 1
    a_handicap = int(sit_h > 2)

    niv = data.niv_et_agr or 0
    lir = data.lir_ecr
    lir_score = {1: 1.0, 2: 0.5, 3: 0.0}.get(lir, 0.5) if lir else 0.5
    indice_scol = (niv / 5 * 0.5) + (lir_score * 0.3)

    raw = {
        "reg": data.reg, "mil": data.mil,
        "sexe": data.sexe, "AGE5": data.age5,
        "LIEN_CM": np.nan,
        "HANDI_VIS": 1, "HANDI_AUD": 1, "HANDI_MOB": 1,
        "HANDI_MEM": 1, "HANDI_ENTR": 1, "HANDI_COM": 1,
        "SIT_HANDICAP": sit_h,
        "NIV_ET_AGR": data.niv_et_agr,
        "LIR_ECR": data.lir_ecr,
        "a_diplome_eg": 1 if (data.niv_et_agr or 0) >= 3 else 0,
        "a_formation_pro": 0,
        "scolarise_actuel": 0,
        "TY_ACT": data.ty_act,
        "STAT_PROF": np.nan,
        "ACT_SECTEUR": np.nan,
        "indice_scolarisation": indice_scol,
        "est_occupe": est_occupe,
        "est_chomeur": est_chomeur,
        "a_handicap": a_handicap,
        "ENF_VIV": data.enf_viv,
        "ENF_DEC": data.enf_dec,
    }

    X = pd.DataFrame([{f: raw.get(f, np.nan) for f in feat_names}])
    X = X.astype(float)   # XGBoost exige float — convertit aussi les object/None
    classe  = int(xgb_model.predict(X)[0])
    probas  = xgb_model.predict_proba(X)[0].tolist()
    score   = round(probas[1] * 0.5 + probas[2] * 1.0, 4)

    return {
        "classe": classe,
        "label":  VULN_LABELS[classe],
        "probas": {
            "non_vulnerable":  round(probas[0], 4),
            "vulnerable":      round(probas[1], 4),
            "tres_vulnerable": round(probas[2], 4),
        },
        "score_continu": score,
        "model": "XGBoost",
    }


# ── K-Means : segmentation régionale ─────────────────────────

@app.get("/segmentation/regions", tags=["Segmentation"])
async def get_regional_segmentation():
    """
    Retourne les clusters régionaux calculés par K-Means.
    Utilisé par le dashboard pour colorier la carte choroplèthe.
    """
    if model_store.regional_df is None:
        raise HTTPException(503, "Données régionales non disponibles")

    df = model_store.regional_df

    regions = []
    for _, row in df.iterrows():
        region = {
            "reg_id":        int(row.get("reg", 0)),
            "nom":           str(row.get("region_label", row.get("profil_label", ""))),
            "cluster":       int(row.get("cluster", 0)),
            "cluster_label": str(row.get("cluster_label", "")),
        }
        # Indicateurs socio-économiques
        for col in ["taux_emploi", "taux_chomage", "niv_edu_moyen",
                    "pct_alphabete", "ratio_dependance", "pct_handicap",
                    "age_moyen", "pct_mineurs", "pct_seniors",
                    "indice_scol_moyen", "taux_mortalite_proxy"]:
            if col in row:
                val = row[col]
                region[col] = round(float(val), 4) if pd.notna(val) else None
        regions.append(region)

    # Résumé par cluster
    clusters_summary = []
    if "cluster" in df.columns and "cluster_label" in df.columns:
        for c_id in sorted(df["cluster"].unique()):
            grp = df[df["cluster"] == c_id]
            clusters_summary.append({
                "cluster_id": int(c_id),
                "label": str(grp["cluster_label"].iloc[0]),
                "nb_regions": len(grp),
                "regions": grp.get("region_label", grp["profil_label"]).tolist(),
            })

    return {
        "k": int(df["cluster"].nunique()) if "cluster" in df.columns else 0,
        "regions": regions,
        "clusters": clusters_summary,
    }


# ── Profil statistique d'une région ──────────────────────────

@app.get("/stats/region/{reg_id}", tags=["Statistiques"])
async def get_region_stats(reg_id: int):
    """
    Retourne le profil socio-économique complet d'une région.
    """
    if reg_id not in REGION_NAMES:
        raise HTTPException(404, f"Région {reg_id} introuvable (1-12)")

    df = model_store.regional_df
    if df is None:
        raise HTTPException(503, "Données régionales non disponibles")

    row = df[df["reg"] == reg_id]
    if row.empty:
        raise HTTPException(404, f"Pas de données pour la région {reg_id}")

    row = row.iloc[0]
    profil = {"reg_id": reg_id, "nom": REGION_NAMES[reg_id]}
    for col in df.columns:
        if col not in ["reg", "region_label", "profil_label"]:
            val = row[col]
            try:
                profil[col] = round(float(val), 4) if pd.notna(val) else None
            except (TypeError, ValueError):
                profil[col] = str(val) if pd.notna(val) else None

    return profil


# ── KPIs globaux pour le dashboard ───────────────────────────

@app.get("/stats/dashboard", tags=["Statistiques"])
async def get_dashboard_kpis():
    """
    KPIs agrégés au niveau national pour le dashboard.
    Retourne les indicateurs clés affichés en haut du dashboard React.
    """
    df = model_store.regional_df
    pred = model_store.predictions_df

    kpis = {
        "timestamp": datetime.now().isoformat(),
        "source":    "RGPH 2014 — HCP Maroc",
    }

    # KPIs régionaux
    if df is not None:
        kpis["national"] = {
            "nb_regions":          len(df),
            "age_moyen_national":  round(float(df["age_moyen"].mean()), 1) if "age_moyen" in df.columns else None,
            "taux_emploi_moyen":   round(float(df["taux_emploi"].mean()), 3) if "taux_emploi" in df.columns else None,
            "taux_chomage_moyen":  round(float(df["taux_chomage"].mean()), 3) if "taux_chomage" in df.columns else None,
            "pct_alphabete_moyen": round(float(df["pct_alphabete"].mean()), 3) if "pct_alphabete" in df.columns else None,
            "niv_edu_moyen":       round(float(df["niv_edu_moyen"].mean()), 2) if "niv_edu_moyen" in df.columns else None,
            "pct_handicap_moyen":  round(float(df["pct_handicap"].mean()), 4) if "pct_handicap" in df.columns else None,
        }
        if "cluster" in df.columns:
            kpis["national"]["nb_clusters"] = int(df["cluster"].nunique())

    # KPIs individuels
    if pred is not None and "classe_vuln_indiv" in pred.columns:
        dist = pred["classe_vuln_indiv"].value_counts(normalize=True).sort_index()
        kpis["vulnerabilite_individuelle"] = {
            "pct_non_vulnerable":  round(float(dist.get(0, 0)) * 100, 1),
            "pct_vulnerable":      round(float(dist.get(1, 0)) * 100, 1),
            "pct_tres_vulnerable": round(float(dist.get(2, 0)) * 100, 1),
            "nb_individus":        len(pred),
        }

        if "reg" in pred.columns and "score_continu" in pred.columns:
            top_vuln = (
                pred.groupby("reg")["score_continu"]
                .mean()
                .sort_values(ascending=False)
                .head(3)
            )
            kpis["top_regions_vulnerables"] = [
                {"reg_id": int(r), "nom": REGION_NAMES.get(r, str(r)),
                 "score_moyen": round(float(s), 4)}
                for r, s in top_vuln.items()
            ]

    return kpis


# ── Prédiction batch ──────────────────────────────────────────

@app.post("/predict/batch/menage", tags=["Prédiction"])
async def predict_menage_batch(data: List[MenageInput]):
    """
    Prédit la classe de vulnérabilité pour une liste de ménages.
    Maximum 1000 ménages par requête.
    """
    if len(data) > 1000:
        raise HTTPException(400, "Maximum 1000 ménages par requête batch")
    if model_store.rf_model is None:
        raise HTTPException(503, "Modèle Random Forest non disponible")

    results = []
    for item in data:
        result = await predict_menage(item)
        results.append(result)

    return {"count": len(results), "predictions": results}
