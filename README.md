# 🇲🇦 Système Analytique Intelligent — RGPH 2014

> Plateforme d'analyse intelligente des données du Recensement Général de la Population et de l'Habitat du Maroc (2014), développée dans le cadre d'une soutenance de fin d'études.

---

## 📌 Présentation

Ce projet implémente un **système analytique de bout en bout** appliqué aux micro-données du RGPH 2014 publiées par le **Haut-Commissariat au Plan (HCP)**. Il couvre l'ensemble du cycle de vie d'un projet data science : ingestion, nettoyage, feature engineering, modélisation ML, exposition via API REST et visualisation interactive.

L'objectif analytique est triple :
- **Classifier** les ménages selon leur niveau de vulnérabilité socio-économique
- **Scorer** les individus par un indice de vulnérabilité composite
- **Segmenter** les régions du Maroc selon leurs profils démographiques et socio-économiques

---

##  Données

| Fichier | Lignes | Variables | Description |
|---|---|---|---|
| `Individu.csv` | 3 341 426 | 48 | Micro-données individuelles |
| `Menage.csv` | — | 42 | Micro-données ménages |

- **Source** : RGPH 2014 — Haut-Commissariat au Plan, Maroc
- **Format original** : STATA (.dta) → converti en CSV
- **Couverture** : 12 régions, 72 provinces
- **Poids de sondage** : `pds ≈ 10` (échantillon au 1/10ème, poids constant)
- **Population représentée** : ~33,4 millions d'habitants

---

##  Architecture

```
rgph_pipeline.py      ← ETL + Random Forest (classification ménage)
     ↓
rgph_xgboost.py       ← Scoring individuel (XGBoost, vulnérabilité)
     ↓
rgph_kmeans.py        ← Segmentation régionale (K-Means)
     ↓
main.py               ← API REST (FastAPI + uvicorn)
     ↓
dashboard/            ← Interface React (4 onglets, Leaflet.js)
```

>  L'ordre d'exécution est séquentiel : chaque script dépend des sorties du précédent.

---

##  Modèles ML

### 1. Random Forest — Classification ménage
- **Cible** : ménage pauvre / non-pauvre (proxy composite)
- **Features** : agrégats ménage (taille, taux d'activité, niveau d'éducation, équipement)
- **Particularité** : variables de fécondité agrégées au niveau ménage avant modélisation

### 2. XGBoost — Score de vulnérabilité individuel
- **Cible** : score de vulnérabilité discrétisé en terciles (0 / 1 / 2)
- **Features** : variables individuelles avec gestion native des NaN conditionnels
- **Avantage** : XGBoost tolère les NaN structurels sans imputation artificielle

### 3. K-Means — Segmentation régionale
- **Périmètre** : 12 régions administratives
- **Sélection de K** : 3 critères simultanés (Elbow, Silhouette, Calinski-Harabasz), plafonné à K=5
- **Features** : ratios régionaux (taux d'activité, alphabétisation, scolarisation, part agriculture)

---

## 🔌 API REST

```bash
uvicorn main:app --reload
```

| Endpoint | Méthode | Description |
|---|---|---|
| `/predict/menage` | POST | Classifie un ménage |
| `/predict/individu` | POST | Score de vulnérabilité individuel |
| `/predict/region` | GET | Profil de segmentation par région |
| `/docs` | GET | Documentation Swagger auto-générée |

---

##  Dashboard

Interface React 4 onglets :

| Onglet | Contenu |
|---|---|
| Vue globale | KPIs nationaux, distributions |
| Carte régionale | Choroplèthe Leaflet.js (12 régions) |
| Modèles ML | Métriques, matrices de confusion |
| Simulateur | Saisie manuelle → prédiction temps réel |

---

##  Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-username>/sai-rgph-2014.git
cd sai-rgph-2014

# 2. Installer les dépendances Python
pip install -r requirements.txt

# 3. Placer les données (ou utiliser le fallback synthétique)
cp /path/to/Individu.csv data/
cp /path/to/Menage.csv data/

# 4. Exécuter le pipeline dans l'ordre
python rgph_pipeline.py
python rgph_xgboost.py
python rgph_kmeans.py

# 5. Lancer l'API
uvicorn main:app --reload

# 6. Lancer le dashboard
cd dashboard && npm install && npm start
```

>  Si `Individu.csv` est absent, tous les scripts basculent automatiquement sur des données synthétiques pour rester exécutables.

---

##  Stack technique

| Couche | Technologies |
|---|---|
| Données | Python, Pandas, pyreadstat |
| ML | scikit-learn, XGBoost |
| API | FastAPI, uvicorn |
| Frontend | React, Leaflet.js, Recharts |
| Visualisation | Matplotlib, Seaborn |

---

## 🗂️ Structure du projet

```
sai-rgph-2014/
├── data/                  # Micro-données (non versionnées)
├── models/                # Modèles entraînés (.pkl)
├── outputs/               # Résultats ETL et features engineered
├── dashboard/             # Application React
├── rgph_pipeline.py
├── rgph_xgboost.py
├── rgph_kmeans.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 📝 Contexte académique

Projet réalisé dans le cadre d'une **soutenance de fin d'études** en data science. Les données RGPH 2014 sont publiques et accessibles via le portail du [Haut-Commissariat au Plan](https://www.hcp.ma).

---

## ⚖️ Licence

Les micro-données RGPH appartiennent au Haut-Commissariat au Plan — Maroc. Ce projet est à vocation académique uniquement.
