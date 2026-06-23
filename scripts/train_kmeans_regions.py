import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "regional_profiles.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "kmeans_regional.pkl"
CLUSTERS_PATH = PROJECT_ROOT / "outputs" / "regional_clusters.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "kmeans_report.txt"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

REGION_NAMES = {
    1:  "Tanger-Tétouan-Al Hoceïma", 2:  "Oriental", 3:  "Fès-Meknès",
    4:  "Rabat-Salé-Kénitra", 5:  "Béni Mellal-Khénifra", 6:  "Casablanca-Settat",
    7:  "Marrakech-Safi", 8:  "Drâa-Tafilalet", 9:  "Souss-Massa",
    10: "Guelmim-Oued Noun", 11: "Laâyoune-Sakia El Hamra", 12: "Dakhla-Oued Ed-Dahab",
}

CLUSTER_FEATURES = [
    "age_moyen", "pct_mineurs", "pct_seniors", "pct_femmes",
    "niv_edu_moyen", "pct_alphabete", "taux_emploi", "taux_chomage",
    "pct_handicap", "ratio_dependance"
]

def run_clustering():
    print(f"Loading profiles from {INPUT_PATH}...")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found. Please run generate_regional_profiles.py first.")

    df = pd.read_csv(INPUT_PATH)
    if len(df) < 3:
        raise ValueError("K-Means needs at least 3 regional profiles.")

    # 1. Prepare and Scale
    feats = [f for f in CLUSTER_FEATURES if f in df.columns]
    if not feats:
        raise ValueError("No clustering features found in regional profiles.")

    X_raw = df[feats].copy()
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"Scaled features: {feats}")

    # 2. Optimal K (Silhouette)
    best_k = 4
    max_sil = -1
    k_metrics = []
    max_k = min(8, len(df) - 1)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        k_metrics.append((k, km.inertia_, sil, calinski))
        if sil > max_sil:
            max_sil = sil
            best_k = k

    print(f"Optimal K found: {best_k} (Silhouette={max_sil:.4f})")

    # 3. Final K-Means
    km = KMeans(n_clusters=best_k, init="k-means++", n_init=50, random_state=42)
    df["cluster"] = km.fit_predict(X_scaled)

    # 4. Labelling centroids
    centroids = df.groupby("cluster")[feats].mean()
    cluster_names = {}
    for c in centroids.index:
        row = centroids.loc[c]
        # Simple heuristic for labeling
        if row.get("taux_emploi", 0) > centroids["taux_emploi"].mean() and row.get("niv_edu_moyen", 0) > centroids["niv_edu_moyen"].mean():
            name = "Développé & Actif"
        elif row.get("ratio_dependance", 0) > centroids["ratio_dependance"].mean():
            name = "Vulnérable & Dépendant"
        elif row.get("pct_alphabete", 0) < centroids.get("pct_alphabete", pd.Series()).mean():
            name = "Rural Traditionnel"
        else:
            name = "En Transition"
        cluster_names[c] = f"Cluster {c} — {name}"

    df["cluster_label"] = df["cluster"].map(cluster_names)

    # 5. Save
    os.makedirs(CLUSTERS_PATH.parent, exist_ok=True)
    df.to_csv(CLUSTERS_PATH, index=False)
    joblib.dump({
        "kmeans": km,
        "scaler": scaler,
        "features": feats,
        "k": best_k,
        "cluster_names": cluster_names,
    }, MODEL_PATH)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=============================================================\n")
        f.write("  RAPPORT - Segmentation Regionale K-Means\n")
        f.write("  RGPH 2014 - Maroc\n")
        f.write("=============================================================\n\n")
        f.write(f"  Nombre de clusters : {best_k}\n")
        f.write(f"  Inertie finale     : {km.inertia_:.4f}\n")
        f.write(f"  Silhouette Score   : {max_sil:.4f}\n")
        f.write(f"  Iterations         : {km.n_iter_}\n\n")
        f.write("  METRIQUES DE SELECTION DU K\n")
        f.write("  k    inertia  silhouette  calinski\n")
        for k, inertia, sil, calinski in k_metrics:
            f.write(f"  {k:<2} {inertia:>9.4f}  {sil:>10.4f}  {calinski:>8.4f}\n")
        f.write("\nCentroids:\n" + centroids.to_string())

    print(f"Model saved to {MODEL_PATH}")
    print(f"Clusters saved to {CLUSTERS_PATH}")

if __name__ == "__main__":
    run_clustering()
