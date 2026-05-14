"""
=============================================================
  RGPH 2014 — Segmentation Régionale K-Means
  Projet : Système Analytique Intelligent (Soutenance)
=============================================================
Objectif : regrouper les 12 régions marocaines en clusters
homogènes selon leurs profils socio-économiques agrégés.

Grain d'analyse : région × milieu (urbain/rural) → 24 profils
puis projection au niveau région pure pour la carte choroplèthe.

Structure :
  1. Configuration
  2. Construction des profils régionaux
  3. Préparation & normalisation
  4. Choix optimal de K (Elbow + Silhouette + Calinski-Harabasz)
  5. K-Means final + analyse des clusters
  6. Visualisations (PCA 2D, profils radar, heatmap)
  7. Export résultats
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ──────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────

DATA_INDIVIDU  = "Individu.csv"
OUTPUT_DIR     = "outputs/"
MODEL_PATH     = os.path.join(OUTPUT_DIR, "kmeans_regional.pkl")
PROFILES_PATH  = os.path.join(OUTPUT_DIR, "regional_profiles.csv")
CLUSTERS_PATH  = os.path.join(OUTPUT_DIR, "regional_clusters.csv")
REPORT_PATH    = os.path.join(OUTPUT_DIR, "kmeans_report.txt")
PLOTS_DIR      = os.path.join(OUTPUT_DIR, "plots/")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
K_RANGE      = range(2, 9)   # on teste K de 2 à 8

# Noms officiels des 12 régions RGPH 2014
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

MILIEU_NAMES = {1: "Urbain", 2: "Rural"}

# ──────────────────────────────────────────────────────────────
# 2. CHARGEMENT & ETL
# ──────────────────────────────────────────────────────────────

def load_and_prepare(path: str) -> pd.DataFrame:
    """Charge les données et applique l'ETL du pipeline principal.

    Priorité à l'import local car les fichiers sont dans le même dossier.
    Le fallback package est gardé si le projet est installé plus tard.
    """
    try:
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
    except ImportError:
        print("[WARN] rgph_pipeline.py introuvable → données synthétiques internes")
        df = _synthetic_fallback(n=50_000)
    return df


def _synthetic_fallback(n: int = 50_000) -> pd.DataFrame:
    """Données synthétiques minimales si pipeline principal absent."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_men = n // 5
    rows = []
    for i in range(n_men):
        reg = rng.integers(1, 13)
        mil = rng.choice([1, 2], p=[0.6, 0.4])
        sz  = rng.integers(1, 10)
        for j in range(int(sz)):
            age5   = int(rng.integers(0, 16) * 5)
            sexe   = rng.choice([1, 2])
            ty_act = rng.choice([0,1,2,3,4,5])
            rows.append({
                "reg": reg, "mil": mil, "pro": reg*10,
                "MEN_PRO": i+1, "NOR_MEN": j+1,
                "sexe": sexe, "AGE5": age5,
                "NIV_ET_AGR": rng.integers(0, 6),
                "LIR_ECR": rng.choice([1,2,3,np.nan], p=[0.5,0.1,0.2,0.2]),
                "TY_ACT": ty_act if age5 >= 15 else np.nan,
                "SIT_HANDICAP": rng.choice([1,2,3,4], p=[0.85,0.08,0.05,0.02]),
                "ENF_VIV": rng.integers(0,6) if sexe==2 and 15<=age5<50 else np.nan,
                "ENF_DEC": rng.integers(0,3) if sexe==2 and 15<=age5<50 else np.nan,
                "HANDI_VIS": rng.choice([1,2,3,4], p=[0.85,0.08,0.04,0.03]),
                "a_diplome_eg": rng.choice([0,1], p=[0.6,0.4]),
                "indice_scolarisation": rng.uniform(0, 1),
                "est_occupe": int(ty_act == 0) if age5 >= 15 else 0,
                "est_chomeur": int(ty_act in [1,2]) if age5 >= 15 else 0,
                "a_handicap": rng.choice([0,1], p=[0.9,0.1]),
            })
    df = pd.DataFrame(rows)
    print(f"[SYNTH] {df.shape[0]:,} individus × {df.shape[1]} colonnes")
    return df


# ──────────────────────────────────────────────────────────────
# 3. CONSTRUCTION DES PROFILS RÉGIONAUX
# ──────────────────────────────────────────────────────────────

def build_regional_profiles(df: pd.DataFrame,
                             grain: str = "region") -> pd.DataFrame:
    """
    Agrège les données individuelles en profils régionaux.

    grain = "region"        → 12 lignes (une par région)
    grain = "region_milieu" → jusqu'à 24 lignes (région × milieu)
    """
    print(f"\n[KMEANS] Construction des profils régionaux (grain={grain})")

    group_cols = ["reg"] if grain == "region" else ["reg", "mil"]

    agg = df.groupby(group_cols).agg(
        # ── Démographie ──────────────────────────────────────
        population          = ("NOR_MEN", "count"),
        age_moyen           = ("AGE5",    "mean"),
        pct_mineurs         = ("AGE5",    lambda x: (x < 15).mean() * 100),
        pct_seniors         = ("AGE5",    lambda x: (x >= 60).mean() * 100),
        pct_femmes          = ("sexe",    lambda x: (x == 2).mean() * 100),

        # ── Éducation ────────────────────────────────────────
        niv_edu_moyen       = ("NIV_ET_AGR",           "mean"),
        indice_scol_moyen   = ("indice_scolarisation",  "mean"),
        pct_diplomes        = ("a_diplome_eg",          "mean"),

        # ── Alphabétisme ─────────────────────────────────────
        pct_alphabete       = ("LIR_ECR",
                               lambda x: (x == 1).sum() / x.notna().sum()
                               if x.notna().sum() > 0 else np.nan),

        # ── Emploi ───────────────────────────────────────────
        taux_emploi         = ("est_occupe",  "mean"),
        taux_chomage        = ("est_chomeur", "mean"),

        # ── Handicap ─────────────────────────────────────────
        pct_handicap        = ("a_handicap",  "mean"),

        # ── Fécondité ────────────────────────────────────────
        enf_viv_moyen       = ("ENF_VIV", "mean"),
        enf_dec_moyen       = ("ENF_DEC", "mean"),
    ).reset_index()

    # Features dérivées
    agg["ratio_dependance"] = (
        (agg["pct_mineurs"] + agg["pct_seniors"]) / 100
    ).clip(0, 1)

    agg["taux_mortalite_proxy"] = (
        agg["enf_dec_moyen"] /
        (agg["enf_viv_moyen"] + agg["enf_dec_moyen"]).replace(0, np.nan)
    ).fillna(0)

    # Labels lisibles
    agg["region_label"] = agg["reg"].map(REGION_NAMES)
    if grain == "region_milieu":
        agg["milieu_label"] = agg["mil"].map(MILIEU_NAMES)
        agg["profil_label"] = agg["region_label"] + " — " + agg["milieu_label"]
    else:
        agg["profil_label"] = agg["region_label"]

    print(f"[KMEANS] {agg.shape[0]} profils × {agg.shape[1]} indicateurs")
    agg.to_csv(PROFILES_PATH, index=False)
    print(f"[SAVE]   Profils → {PROFILES_PATH}")
    return agg


# ──────────────────────────────────────────────────────────────
# 4. FEATURES POUR LE CLUSTERING
# ──────────────────────────────────────────────────────────────

CLUSTER_FEATURES = [
    "age_moyen",
    "pct_mineurs",
    "pct_seniors",
    "pct_femmes",
    "niv_edu_moyen",
    "indice_scol_moyen",
    "pct_diplomes",
    "pct_alphabete",
    "taux_emploi",
    "taux_chomage",
    "pct_handicap",
    "ratio_dependance",
    "taux_mortalite_proxy",
]


def prepare_cluster_matrix(profiles: pd.DataFrame) -> tuple:
    """
    Normalise les features pour K-Means.
    StandardScaler : moyenne 0, écart-type 1.
    Retourne (X_scaled, scaler, features_used, labels)
    """
    feats = [f for f in CLUSTER_FEATURES if f in profiles.columns]
    X_raw = profiles[feats].copy()

    # Imputation des NaN résiduels (médiane de l'indicateur)
    for col in feats:
        if X_raw[col].isna().any():
            X_raw[col] = X_raw[col].fillna(X_raw[col].median())

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    labels = profiles["profil_label"].tolist()
    print(f"\n[KMEANS] Matrice de clustering : {X_scaled.shape} — {feats}")
    return X_scaled, scaler, feats, labels


# ──────────────────────────────────────────────────────────────
# 5. CHOIX OPTIMAL DE K
# ──────────────────────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray) -> dict:
    """
    Trois critères complémentaires pour choisir K :
    - Inertie (Elbow method)
    - Silhouette Score : cohésion intra-cluster vs séparation inter
    - Calinski-Harabasz : ratio dispersion inter/intra
    """
    print("\n[KMEANS] Recherche du K optimal...")

    results = {"k": [], "inertia": [], "silhouette": [], "calinski": []}

    for k in K_RANGE:
        km = KMeans(n_clusters=k, init="k-means++", n_init=20,
                    max_iter=500, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)

        inertia   = km.inertia_
        sil       = silhouette_score(X_scaled, labels) if k > 1 else 0
        cal       = calinski_harabasz_score(X_scaled, labels) if k > 1 else 0

        results["k"].append(k)
        results["inertia"].append(inertia)
        results["silhouette"].append(sil)
        results["calinski"].append(cal)

        print(f"   K={k}  Inertie={inertia:.2f}  "
              f"Silhouette={sil:.4f}  Calinski={cal:.2f}")

    df_res = pd.DataFrame(results)

    # K optimal = meilleur silhouette score
    k_opt_sil = df_res.loc[df_res["silhouette"].idxmax(), "k"]
    k_opt_cal = df_res.loc[df_res["calinski"].idxmax(), "k"]

    print(f"\n   → K optimal (Silhouette) : {k_opt_sil}")
    print(f"   → K optimal (Calinski)   : {k_opt_cal}")

    # Consensus : si accord → utiliser ce K, sinon prendre silhouette
    k_final = int(k_opt_sil) if k_opt_sil == k_opt_cal else int(k_opt_sil)
    # Pour 12 régions : forcer K=4 si le consensus est > 6 (peu interprétable)
    if k_final > 5:
        k_final = 4
        print(f"   → K réduit à {k_final} pour l'interprétabilité (12 régions)")

    print(f"\n   ✓ K FINAL retenu : {k_final}")

    _plot_k_selection(df_res)
    return {"k_final": k_final, "metrics": df_res}


def _plot_k_selection(df_res: pd.DataFrame) -> None:
    """Graphique Elbow + Silhouette."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Sélection du nombre de clusters K", fontsize=13, fontweight="bold")

    colors = ["#2563EB", "#16A34A", "#DC2626"]
    metrics = [("inertia", "Inertie (Elbow)"),
               ("silhouette", "Silhouette Score"),
               ("calinski", "Calinski-Harabasz")]

    for ax, (col, title), color in zip(axes, metrics, colors):
        ax.plot(df_res["k"], df_res[col], "o-", color=color,
                linewidth=2, markersize=7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Nombre de clusters K")
        ax.set_ylabel(col.capitalize())
        ax.grid(True, alpha=0.3)

        # Marquer le maximum (pour silhouette et calinski)
        if col != "inertia":
            best_k = df_res.loc[df_res[col].idxmax(), "k"]
            best_v = df_res[col].max()
            ax.axvline(best_k, color=color, linestyle="--", alpha=0.5)
            ax.annotate(f"K={best_k}", xy=(best_k, best_v),
                        xytext=(best_k+0.2, best_v),
                        fontsize=9, color=color)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "k_selection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  Graphique K → {path}")


# ──────────────────────────────────────────────────────────────
# 6. K-MEANS FINAL
# ──────────────────────────────────────────────────────────────

def run_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    """Entraîne le K-Means final avec k-means++ et 50 initialisations."""
    print(f"\n[KMEANS] Entraînement final avec K={k}...")
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=50,            # 50 initialisations aléatoires → meilleure stabilité
        max_iter=1000,
        tol=1e-6,
        random_state=RANDOM_STATE,
    )
    km.fit(X_scaled)
    print(f"   Inertie finale : {km.inertia_:.4f}")
    print(f"   Itérations     : {km.n_iter_}")
    return km


def label_clusters(profiles: pd.DataFrame,
                   cluster_labels: np.ndarray,
                   feats: pd.DataFrame,
                   feature_names: list) -> pd.DataFrame:
    """
    Attribue des noms interprétatifs aux clusters
    basés sur leurs centroïdes dans l'espace original.
    """
    profiles = profiles.copy()
    profiles["cluster"] = cluster_labels

    # Centroïdes dans l'espace original (non normalisé)
    centroids = profiles.groupby("cluster")[feature_names].mean()

    # Heuristique de nommage automatique
    # Basé sur taux d'emploi, niveau éducation, ratio dépendance
    cluster_names = {}
    for c in centroids.index:
        row = centroids.loc[c]
        edu  = row.get("niv_edu_moyen", 0)
        emp  = row.get("taux_emploi", 0)
        dep  = row.get("ratio_dependance", 0)
        alph = row.get("pct_alphabete", 0)

        if emp > centroids["taux_emploi"].mean() and edu > centroids["niv_edu_moyen"].mean():
            name = "Développé & Actif"
        elif dep > centroids["ratio_dependance"].mean() and edu < centroids["niv_edu_moyen"].mean():
            name = "Vulnérable & Dépendant"
        elif alph < centroids.get("pct_alphabete", pd.Series()).mean():
            name = "Rural Traditionnel"
        else:
            name = "En Transition"

        cluster_names[c] = f"Cluster {c} — {name}"

    profiles["cluster_label"] = profiles["cluster"].map(cluster_names)
    return profiles, cluster_names


# ──────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ──────────────────────────────────────────────────────────────

PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2"]


def plot_pca_clusters(X_scaled: np.ndarray,
                      profiles: pd.DataFrame,
                      k: int) -> None:
    """Projection PCA 2D des clusters régionaux."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")

    clusters = profiles["cluster"].values
    labels   = profiles["profil_label"].values

    for c in range(k):
        mask = clusters == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=PALETTE[c % len(PALETTE)],
                   s=180, alpha=0.85, edgecolors="white",
                   linewidth=1.5, zorder=3,
                   label=profiles.loc[profiles["cluster"]==c,
                                      "cluster_label"].iloc[0])

    # Annotations régions
    for i, (x, y, lbl) in enumerate(zip(X_2d[:, 0], X_2d[:, 1], labels)):
        ax.annotate(lbl, (x, y),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=7.5, color="#1E293B", alpha=0.85)

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=11)
    ax.set_title("Segmentation régionale — Projection PCA 2D",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pca_clusters.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  PCA 2D → {path}")


def plot_heatmap(profiles: pd.DataFrame, feature_names: list) -> None:
    """Heatmap normalisée : régions × indicateurs, colorée par cluster."""
    feats_plot = [f for f in feature_names if f in profiles.columns]
    data = profiles[feats_plot].copy()

    # Normalisation 0-1 par colonne pour la lisibilité
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
    data_norm.index = profiles["profil_label"]

    # Couleurs de lignes par cluster
    row_colors = profiles["cluster"].map(
        {c: PALETTE[c % len(PALETTE)] for c in profiles["cluster"].unique()}
    )
    row_colors.index = data_norm.index

    fig, ax = plt.subplots(figsize=(14, max(6, len(profiles) * 0.5 + 2)))
    sns.heatmap(
        data_norm,
        ax=ax,
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor="#E2E8F0",
        annot=True, fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "Valeur normalisée [0-1]"},
    )
    ax.set_title("Profils socio-économiques régionaux — RGPH 2014",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Légende clusters
    patches = [
        mpatches.Patch(color=PALETTE[c % len(PALETTE)],
                       label=profiles.loc[profiles["cluster"]==c,
                                          "cluster_label"].iloc[0])
        for c in sorted(profiles["cluster"].unique())
    ]
    ax.legend(handles=patches, loc="upper left",
              bbox_to_anchor=(1.18, 1), fontsize=8, title="Clusters")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "heatmap_regions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  Heatmap → {path}")


def plot_radar(profiles: pd.DataFrame,
               feature_names: list, k: int) -> None:
    """Radar chart des centroïdes par cluster."""
    # Sélection de 6 features lisibles pour le radar
    radar_feats = [f for f in [
        "niv_edu_moyen", "taux_emploi", "pct_alphabete",
        "ratio_dependance", "taux_chomage", "pct_handicap"
    ] if f in profiles.columns][:6]

    if len(radar_feats) < 3:
        return

    centroids = profiles.groupby("cluster")[radar_feats].mean()

    # Normalisation 0-1
    cent_norm = (centroids - centroids.min()) / (
        centroids.max() - centroids.min() + 1e-9
    )

    N = len(radar_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # fermer le polygone

    labels_clean = {
        "niv_edu_moyen":     "Éducation",
        "taux_emploi":       "Emploi",
        "pct_alphabete":     "Alphabétisme",
        "ratio_dependance":  "Dépendance",
        "taux_chomage":      "Chômage",
        "pct_handicap":      "Handicap",
    }

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4.5),
                              subplot_kw=dict(polar=True))
    if k == 1:
        axes = [axes]

    fig.suptitle("Profils radar des clusters régionaux",
                 fontsize=13, fontweight="bold", y=1.02)

    for i, (c, ax) in enumerate(zip(cent_norm.index, axes)):
        values = cent_norm.loc[c].tolist()
        values += values[:1]

        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, values, "o-", linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [labels_clean.get(f, f) for f in radar_feats],
            size=8
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.5", "0.75"], size=6)
        ax.set_title(
            profiles.loc[profiles["cluster"]==c, "cluster_label"].iloc[0],
            size=9, pad=12, color=color, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "radar_clusters.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  Radar → {path}")


# ──────────────────────────────────────────────────────────────
# 8. RAPPORT TEXTUEL
# ──────────────────────────────────────────────────────────────

def build_report(profiles: pd.DataFrame,
                 feature_names: list,
                 k: int,
                 metrics: pd.DataFrame,
                 km: KMeans) -> str:
    """Génère un rapport analytique complet."""

    sil = silhouette_score(
        StandardScaler().fit_transform(
            profiles[feature_names].fillna(profiles[feature_names].median())
        ),
        profiles["cluster"]
    )

    lines = [
        "=" * 65,
        "  RAPPORT — Segmentation Régionale K-Means",
        "  RGPH 2014 — Maroc",
        "=" * 65,
        f"\n  Nombre de clusters : {k}",
        f"  Inertie finale     : {km.inertia_:.4f}",
        f"  Silhouette Score   : {sil:.4f}",
        f"  Itérations         : {km.n_iter_}",
        "\n" + "─" * 65,
        "  COMPOSITION DES CLUSTERS",
        "─" * 65,
    ]

    for c in sorted(profiles["cluster"].unique()):
        grp = profiles[profiles["cluster"] == c]
        label = grp["cluster_label"].iloc[0]
        regions = " | ".join(grp["region_label"].tolist()
                             if "region_label" in grp.columns
                             else grp["profil_label"].tolist())
        lines.append(f"\n  {label} ({len(grp)} profils)")
        lines.append(f"  Régions : {regions}")

        # Moyennes des indicateurs clés
        for feat in ["niv_edu_moyen", "taux_emploi", "taux_chomage",
                     "ratio_dependance", "pct_alphabete"]:
            if feat in grp.columns:
                val = grp[feat].mean()
                lines.append(f"    {feat:<28} : {val:.4f}")

    lines += [
        "\n" + "─" * 65,
        "  MÉTRIQUES DE SÉLECTION DU K",
        "─" * 65,
        metrics.to_string(index=False),
        "\n" + "=" * 65,
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 9. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────

def run_kmeans_pipeline(df: pd.DataFrame,
                        grain: str = "region",
                        force_k: int = None) -> pd.DataFrame:
    """
    Pipeline K-Means complet.

    Args:
        df      : DataFrame post-ETL
        grain   : "region" ou "region_milieu"
        force_k : forcer un K spécifique (None = automatique)

    Returns:
        profiles DataFrame enrichi avec colonnes cluster/cluster_label
    """
    print("\n" + "="*65)
    print("  PHASE K-MEANS — SEGMENTATION RÉGIONALE")
    print("="*65)

    # Profils régionaux
    profiles = build_regional_profiles(df, grain=grain)

    # Matrice normalisée
    X_scaled, scaler, feat_names, labels = prepare_cluster_matrix(profiles)

    # Choix K
    if force_k:
        k = force_k
        k_metrics = pd.DataFrame({"k": [k], "inertia": [0],
                                   "silhouette": [0], "calinski": [0]})
        print(f"\n[KMEANS] K forcé à {k}")
    else:
        k_result  = find_optimal_k(X_scaled)
        k         = k_result["k_final"]
        k_metrics = k_result["metrics"]

    # K-Means final
    km = run_kmeans(X_scaled, k)
    cluster_labels = km.labels_

    # Labellisation
    profiles, cluster_names = label_clusters(
        profiles, cluster_labels,
        profiles[feat_names], feat_names
    )

    # Résumé console
    print("\n[KMEANS] Résultats :")
    for c in sorted(profiles["cluster"].unique()):
        grp = profiles[profiles["cluster"] == c]
        print(f"   {cluster_names[c]} : {grp['profil_label'].tolist()}")

    # Visualisations
    plot_pca_clusters(X_scaled, profiles, k)
    plot_heatmap(profiles, feat_names)
    plot_radar(profiles, feat_names, k)

    # Rapport
    report = build_report(profiles, feat_names, k, k_metrics, km)
    print("\n" + report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[SAVE] Rapport → {REPORT_PATH}")

    # Export CSV
    profiles.to_csv(CLUSTERS_PATH, index=False)
    print(f"[SAVE] Clusters → {CLUSTERS_PATH}")

    # Sauvegarde modèle
    joblib.dump({
        "kmeans":   km,
        "scaler":   scaler,
        "features": feat_names,
        "k":        k,
        "cluster_names": cluster_names,
    }, MODEL_PATH)
    print(f"[SAVE] Modèle → {MODEL_PATH}")

    return profiles


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  RGPH 2014 — Segmentation Régionale K-Means")
    print("="*65 + "\n")

    df = load_and_prepare(DATA_INDIVIDU)
    profiles = run_kmeans_pipeline(df, grain="region")

    print("\n" + "="*65)
    print("  K-Means terminé ✓")
    print("="*65)
    return profiles


if __name__ == "__main__":
    profiles = main()
