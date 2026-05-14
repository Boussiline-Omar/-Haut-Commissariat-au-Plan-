# 🇲🇦 Système Analytique Intelligent — RGPH 2014

> Plateforme d'analyse intelligente des micro-données du Recensement Général de la Population et de l'Habitat du Maroc 2014, développée dans le cadre d'une soutenance de fin d'études.

---

## 📌 Présentation

Ce projet couvre un pipeline **data science de bout en bout** : ingestion des données, nettoyage, feature engineering, modélisation ML, exposition des résultats via API REST et visualisation dans un dashboard interactif.

Objectifs analytiques :

- classifier les ménages selon un niveau de vulnérabilité socio-économique ;
- scorer les individus avec un indice composite de vulnérabilité ;
- segmenter les régions du Maroc selon des profils démographiques et socio-économiques.

> ⚠️ Les scores de vulnérabilité sont des **proxys académiques construits** à partir de variables RGPH. Ils ne remplacent pas un indice officiel du HCP ou un revenu déclaré.

---

## 📊 Données

| Fichier | Description |
|---|---|
| `Individu.csv` | Micro-données individuelles RGPH 2014 |
| `Menage.csv` | Micro-données ménages RGPH 2014, conservées mais non obligatoires dans le pipeline actuel |

- Source : RGPH 2014 — Haut-Commissariat au Plan, Maroc
- Couverture : 12 régions marocaines
- Format accepté par le code : `.csv`, `.dta`, `.sav`
- Fallback : si `Individu.csv` est absent, les scripts génèrent un jeu de données synthétique pour les tests.

---

## 🧱 Architecture

```text
rgph_pipeline.py      → ETL + Random Forest, classification ménage
rgph_xgboost.py      → Scoring individuel avec XGBoost
rgph_kmeans.py       → Segmentation régionale avec K-Means
main.py              → API REST FastAPI
rgph_dashboard.html  → Dashboard HTML/CSS/JS avec Chart.js
```

Ordre conseillé d'exécution :

```bash
python rgph_pipeline.py
python rgph_xgboost.py
python rgph_kmeans.py
uvicorn main:app --reload
```

---

## 🤖 Modèles ML

### 1. Random Forest — Classification ménage

- Grain : ménage agrégé à partir du fichier individus
- Cible : classe de vulnérabilité proxy en 3 niveaux
- Variables : taille du ménage, dépendance, emploi, éducation, handicap, milieu

### 2. XGBoost — Score de vulnérabilité individuel

- Grain : individu
- Cible : score composite discrétisé en terciles `0 / 1 / 2`
- Avantage : XGBoost gère naturellement les valeurs manquantes structurelles

### 3. K-Means — Segmentation régionale

- Grain : région
- Objectif : regrouper les 12 régions selon leurs indicateurs socio-économiques
- Choix de K : Elbow, Silhouette, Calinski-Harabasz

---

## 🔌 API REST

Lancement :

```bash
uvicorn main:app --reload
```

| Endpoint | Méthode | Description |
|---|---:|---|
| `/` | GET | Health check |
| `/regions` | GET | Liste des 12 régions |
| `/predict/menage` | POST | Classification d'un ménage avec Random Forest |
| `/predict/individu` | POST | Scoring d'un individu avec XGBoost |
| `/predict/batch/menage` | POST | Prédiction batch ménages |
| `/segmentation/regions` | GET | Clusters régionaux K-Means |
| `/stats/region/{reg_id}` | GET | Profil statistique d'une région |
| `/stats/dashboard` | GET | KPIs globaux du dashboard |
| `/docs` | GET | Documentation Swagger |

---

## 🖥️ Dashboard

Le dashboard fourni est un fichier autonome :

```bash
start rgph_dashboard.html   # Windows
# ou ouvrir le fichier directement dans le navigateur
```

Contenu :

| Onglet | Contenu |
|---|---|
| Vue globale | KPIs nationaux, donut de vulnérabilité, top régions |
| Carte régionale | Carte SVG interactive par région |
| Modèles ML | Comparaison des métriques et importance des variables |
| Simulateur | Formulaire de scoring individuel simplifié |

---

## ⚙️ Installation

```bash
# 1. Créer un environnement virtuel
python -m venv .venv

# 2. Activer l'environnement
# Windows PowerShell
.venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Placer les données à la racine du projet
# Individu.csv
# Menage.csv, optionnel

# 5. Exécuter les scripts
python rgph_pipeline.py
python rgph_xgboost.py
python rgph_kmeans.py

# 6. Lancer l'API
uvicorn main:app --reload
```

---

## 🧰 Stack technique

| Couche | Technologies |
|---|---|
| Données | Python, Pandas, NumPy, pyreadstat |
| ML | scikit-learn, XGBoost, SHAP |
| API | FastAPI, Uvicorn, Pydantic |
| Visualisation Python | Matplotlib, Seaborn |
| Dashboard | HTML, CSS, JavaScript, Chart.js |

---

## 🗂️ Structure du projet

```text
sai-rgph-2014/
├── Individu.csv              # non versionné
├── Menage.csv                # non versionné
├── outputs/                  # modèles, rapports, exports
├── rgph_pipeline.py
├── rgph_xgboost.py
├── rgph_kmeans.py
├── main.py
├── rgph_dashboard.html
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📝 Contexte académique

Projet réalisé dans le cadre d'une soutenance de fin d'études en data science. Les micro-données RGPH appartiennent au Haut-Commissariat au Plan — Maroc. Ce dépôt est à vocation académique uniquement.
