from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("Statistique_Avancee_RGPH.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


cells: list[dict] = [
    md(
        r"""
# Statistique Avancee - RGPH 2014

**Projet SFE : Systeme Analytique Intelligent - RGPH 2014**

Ce notebook propose une analyse statistique avancee des micro-donnees du Recensement General de la Population et de l'Habitat du Maroc 2014. Il couvre l'inspection des donnees, les statistiques descriptives, les analyses demographiques, education, emploi, handicap, les tests statistiques, la correlation, la PCA et une segmentation regionale par K-Means.

Les scores de vulnerabilite produits ici sont des **proxies academiques** construits a partir des variables disponibles. Ils servent a l'analyse, a la soutenance et au dashboard, mais ne doivent pas etre presentes comme des indicateurs officiels du HCP.
"""
    ),
    md(
        r"""
## 1. Importation des bibliotheques

Cette section importe les bibliotheques necessaires et applique un style visuel sobre pour les graphiques. Le notebook est concu pour eviter les erreurs lorsque certaines colonnes sont absentes.
"""
    ),
    code(
        r"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency, kruskal, f_oneway, shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from IPython.display import display, Markdown
except ImportError:
    def display(obj):
        print(obj)
    def Markdown(text):
        return text

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Environnement pret.")
"""
    ),
    md(
        r"""
## 2. Chargement des donnees et inspection automatique des colonnes

Le notebook charge `Individu.csv` et tente de charger `Menage.csv` si disponible. Ensuite, il inspecte les colonnes reelles et cree un dictionnaire `COLS` qui mappe les variables conceptuelles vers les noms effectivement presents dans les donnees.
"""
    ),
    code(
        r"""
INDIVIDU_PATH = Path("Individu.csv")
MENAGE_PATH = Path("Menage.csv")

def load_csv_if_exists(path: Path, required: bool = False) -> pd.DataFrame | None:
    if not path.exists():
        message = f"Fichier absent: {path}"
        if required:
            raise FileNotFoundError(message)
        print(message)
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"{path.name} charge: {df.shape[0]:,} lignes x {df.shape[1]:,} colonnes")
        return df
    except Exception as exc:
        if required:
            raise
        print(f"Impossible de charger {path}: {exc}")
        return None

df_ind = load_csv_if_exists(INDIVIDU_PATH, required=True)
df_men = load_csv_if_exists(MENAGE_PATH, required=False)

print("\nColonnes detectees dans Individu.csv:")
display(pd.DataFrame({"colonne": df_ind.columns, "type": [str(t) for t in df_ind.dtypes]}))

display(df_ind.head())
"""
    ),
    code(
        r"""
# Adaptation automatique aux noms reels des colonnes.
# Chaque cle represente une variable conceptuelle; la valeur sera le nom reel trouve dans le CSV.

COLUMN_ALIASES = {
    "region": ["reg", "REG", "region", "region_id", "code_region"],
    "province": ["pro", "province", "code_province"],
    "milieu": ["mil", "MIL", "milieu"],
    "menage_id": ["MEN_PRO", "menage_id", "id_menage"],
    "ordre_menage": ["NOR_MEN", "ordre_menage"],
    "lien_cm": ["LIEN_CM", "lien_cm"],
    "sexe": ["sexe", "SEXE", "sex"],
    "age5": ["AGE5", "age5", "age"],
    "age1": ["AGE1", "age1"],
    "alphab": ["LIR_ECR", "lir_ecr", "alphabetisme"],
    "educ": ["NIV_ET_AGR", "niv_et_agr", "niveau_education"],
    "educ_detail": ["NIV_ET", "niv_et"],
    "scolarise": ["scol", "SCOL"],
    "diplome": ["EG_DIP_GG_DET", "EG_DIP_GG", "diplome"],
    "formation_pro": ["FP_DIP_GG", "formation_pro"],
    "ty_act": ["TY_ACT", "ty_act"],
    "stat_prof": ["STAT_PROF", "stat_prof"],
    "secteur": ["ACT_SECTEUR", "act_secteur"],
    "sit_handicap": ["SIT_HANDICAP", "sit_handicap"],
    "handi_vis": ["HANDI_VIS"],
    "handi_aud": ["HANDI_AUD"],
    "handi_mob": ["HANDI_MOB"],
    "handi_mem": ["HANDI_MEM"],
    "handi_entr": ["HANDI_ENTR"],
    "handi_com": ["HANDI_COM"],
    "enf_viv": ["ENF_VIV"],
    "enf_dec": ["ENF_DEC"],
}

def resolve_columns(df: pd.DataFrame, aliases: dict[str, list[str]]) -> dict[str, str | None]:
    lower_lookup = {c.lower(): c for c in df.columns}
    resolved = {}
    for key, candidates in aliases.items():
        found = None
        for cand in candidates:
            if cand in df.columns:
                found = cand
                break
            if cand.lower() in lower_lookup:
                found = lower_lookup[cand.lower()]
                break
        resolved[key] = found
    return resolved

COLS = resolve_columns(df_ind, COLUMN_ALIASES)
cols_table = pd.DataFrame(
    [{"variable": k, "colonne_reelle": v, "disponible": v is not None} for k, v in COLS.items()]
)
display(cols_table)

missing_concepts = cols_table.loc[~cols_table["disponible"], "variable"].tolist()
print("Variables conceptuelles absentes:", missing_concepts if missing_concepts else "Aucune variable critique absente.")
"""
    ),
    md(
        r"""
## 3. Nettoyage statistique de base

On quantifie les valeurs manquantes, les doublons, les types de colonnes et les valeurs incoherentes sur les variables RGPH principales. Les NaN conditionnels ne sont pas supprimes automatiquement, car ils peuvent signifier "non concerne" selon l'age, le sexe ou le statut d'activite.
"""
    ),
    code(
        r"""
def existing(keys: list[str]) -> list[str]:
    return [COLS[k] for k in keys if COLS.get(k) is not None]

missing_stats = (
    df_ind.isna().sum()
    .to_frame("nb_missing")
    .assign(pct_missing=lambda d: (d["nb_missing"] / len(df_ind) * 100).round(2))
    .sort_values("pct_missing", ascending=False)
)
display(missing_stats.head(30))

numeric_cols = df_ind.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in df_ind.columns if c not in numeric_cols]

print(f"Colonnes numeriques: {len(numeric_cols)}")
print(f"Colonnes categorielles/non numeriques: {len(categorical_cols)}")
print(f"Doublons exacts: {df_ind.duplicated().sum():,}")
"""
    ),
    code(
        r"""
EXPECTED_RANGES = {
    "region": set(range(1, 13)),
    "milieu": {1, 2},
    "sexe": {1, 2},
    "alphab": {1, 2, 3},
    "ty_act": {0, 1, 2, 3, 4, 5},
    "sit_handicap": {1, 2, 3, 4},
}

incoherences = []
for key, allowed in EXPECTED_RANGES.items():
    col = COLS.get(key)
    if col is None:
        continue
    values = pd.to_numeric(df_ind[col], errors="coerce")
    invalid = values.dropna()[~values.dropna().isin(allowed)]
    incoherences.append({
        "variable": key,
        "colonne": col,
        "nb_invalides": int(invalid.shape[0]),
        "exemples": sorted(invalid.unique().tolist())[:10],
    })

age_col = COLS.get("age5")
if age_col:
    age = pd.to_numeric(df_ind[age_col], errors="coerce")
    invalid_age = age.dropna()[(age.dropna() < 0) | (age.dropna() > 120)]
    incoherences.append({
        "variable": "age5",
        "colonne": age_col,
        "nb_invalides": int(invalid_age.shape[0]),
        "exemples": sorted(invalid_age.unique().tolist())[:10],
    })

display(pd.DataFrame(incoherences))
"""
    ),
    md(
        r"""
## 4. Statistiques descriptives avancees

Le tableau `desc_stats_advanced` resume les variables numeriques avec moyenne, mediane, ecart-type, variance, quartiles, skewness et kurtosis.
"""
    ),
    code(
        r"""
def advanced_describe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "variable": col,
            "count": int(s.count()),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "variance": s.var(),
            "min": s.min(),
            "q1": s.quantile(0.25),
            "q3": s.quantile(0.75),
            "max": s.max(),
            "skewness": s.skew(),
            "kurtosis": s.kurtosis(),
        })
    return pd.DataFrame(rows).round(4)

desc_stats_advanced = advanced_describe(df_ind, numeric_cols)
display(desc_stats_advanced)
"""
    ),
    md(
        r"""
## 5. Construction des indicateurs socio-economiques

Cette cellule cree les indicateurs utilises par les analyses suivantes. Les formules reprennent la logique du projet : education, emploi, handicap, age vulnerable et score proxy de vulnerabilite.
"""
    ),
    code(
        r"""
df = df_ind.copy()

def to_num(col: str | None, default=np.nan) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

reg = to_num(COLS.get("region"))
mil = to_num(COLS.get("milieu"))
sexe = to_num(COLS.get("sexe"))
age5 = to_num(COLS.get("age5"))
educ = to_num(COLS.get("educ"))
alphab = to_num(COLS.get("alphab"))
ty_act = to_num(COLS.get("ty_act"))
sit_handicap = to_num(COLS.get("sit_handicap"))

if "a_diplome_eg" not in df.columns:
    diplome_col = COLS.get("diplome")
    df["a_diplome_eg"] = to_num(diplome_col).notna().astype(int) if diplome_col else 0

if "a_formation_pro" not in df.columns:
    fp_col = COLS.get("formation_pro")
    df["a_formation_pro"] = to_num(fp_col).notna().astype(int) if fp_col else 0

if "scolarise_actuel" not in df.columns:
    scol_col = COLS.get("scolarise")
    df["scolarise_actuel"] = (to_num(scol_col) == 1).astype(int) if scol_col else 0

if "indice_scolarisation" not in df.columns:
    score_edu = pd.Series(0.0, index=df.index)
    if COLS.get("educ"):
        score_edu += educ.clip(0, 5).fillna(0) / 5 * 0.5
    if COLS.get("alphab"):
        score_edu += alphab.map({1: 1.0, 2: 0.5, 3: 0.0}).fillna(0.5) * 0.3
    score_edu += df["a_diplome_eg"].fillna(0) * 0.2
    df["indice_scolarisation"] = score_edu.clip(0, 1)

if "est_occupe" not in df.columns:
    df["est_occupe"] = (ty_act == 0).astype(int)

if "est_chomeur" not in df.columns:
    df["est_chomeur"] = ty_act.isin([1, 2]).astype(int)

handi_cols = [c for c in existing(["handi_vis", "handi_aud", "handi_mob", "handi_mem", "handi_com"]) if c in df.columns]
if "a_handicap" not in df.columns:
    if handi_cols:
        df["a_handicap"] = (df[handi_cols].apply(pd.to_numeric, errors="coerce").max(axis=1) > 1).astype(int)
    elif COLS.get("sit_handicap"):
        df["a_handicap"] = sit_handicap.gt(1).fillna(False).astype(int)
    else:
        df["a_handicap"] = 0

score = pd.Series(0.0, index=df.index)
score += (mil == 2).astype(float) * 1.5
score += ((5 - educ.clip(0, 5)) / 5).fillna(0.5) * 2.0
score += alphab.map({1: 0.0, 2: 0.5, 3: 1.0}).fillna(0.5) * 1.5
adulte = age5 >= 15
score += (adulte & ty_act.isin([3, 5])).astype(float) * 1.5
score += (adulte & ty_act.isin([1, 2])).astype(float) * 1.0
score += sit_handicap.map({1: 0.0, 2: 0.5, 3: 1.0, 4: 2.0}).fillna(0.0)
score += ((age5 < 5) | (age5 >= 65)).astype(float) * 0.5
score += (1 - df["a_diplome_eg"]).fillna(0) * 0.5

df["score_vulnerabilite_proxy"] = score
df["classe_vulnerabilite"] = pd.qcut(
    score.rank(method="first"), q=3, labels=[0, 1, 2]
).astype(int)

display(df[["indice_scolarisation", "est_occupe", "est_chomeur", "a_handicap", "score_vulnerabilite_proxy", "classe_vulnerabilite"]].head())
"""
    ),
    md(
        r"""
**Note methodologique.** Le score `score_vulnerabilite_proxy` est un score academique. Il combine le milieu rural, la faible education, l'analphabetisme, le chomage ou l'inactivite, le handicap et l'age vulnerable. Il ne remplace pas un revenu observe, un indice officiel de pauvrete ou une mesure institutionnelle du HCP.
"""
    ),
    md(
        r"""
## 6. Analyse demographique

On analyse ici la repartition par region, milieu, sexe et age. Les libelles numeriques du RGPH sont conserves pour rester proches des donnees sources.
"""
    ),
    code(
        r"""
region_col = COLS.get("region")
milieu_col = COLS.get("milieu")
sexe_col = COLS.get("sexe")
age_col = COLS.get("age5")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

if region_col:
    df[region_col].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#2563EB")
    axes[0].set_title("Distribution par region")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Effectif")

if sexe_col:
    df[sexe_col].value_counts().sort_index().plot(kind="bar", ax=axes[1], color="#16A34A")
    axes[1].set_title("Distribution par sexe")
    axes[1].set_xlabel("Sexe")

if milieu_col:
    df[milieu_col].value_counts().sort_index().plot(kind="bar", ax=axes[2], color="#D97706")
    axes[2].set_title("Distribution par milieu")
    axes[2].set_xlabel("Milieu")

plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
def pct(condition: pd.Series) -> float:
    return float(condition.mean() * 100)

group_cols = [c for c in [region_col] if c]
if group_cols and age_col:
    demo_region = df.groupby(region_col).agg(
        population=(age_col, "count"),
        age_moyen=(age_col, "mean"),
        pct_mineurs=(age_col, lambda s: pct(pd.to_numeric(s, errors="coerce") < 15)),
        pct_seniors=(age_col, lambda s: pct(pd.to_numeric(s, errors="coerce") >= 60)),
    ).reset_index()
    demo_region["ratio_dependance"] = ((demo_region["pct_mineurs"] + demo_region["pct_seniors"]) / 100).round(4)
    display(demo_region.round(3))

    sns.barplot(data=demo_region, x=region_col, y="age_moyen", color="#2563EB")
    plt.title("Age moyen par region")
    plt.xlabel("Region")
    plt.ylabel("Age moyen approx. AGE5")
    plt.show()
else:
    print("Colonnes region/age absentes: analyse demographique regionale ignoree.")
"""
    ),
    md(
        r"""
## 7. Analyse education

L'analyse porte sur le niveau d'education agrege, l'alphabetisme et l'indice de scolarisation construit.
"""
    ),
    code(
        r"""
education_cols = [c for c in [region_col, milieu_col, COLS.get("educ"), COLS.get("alphab"), "indice_scolarisation"] if c]
display(df[education_cols].head())

if region_col and COLS.get("alphab") and COLS.get("educ"):
    edu_region = df.groupby(region_col).agg(
        taux_alphabetisme=(COLS["alphab"], lambda s: (pd.to_numeric(s, errors="coerce") == 1).mean() * 100),
        niveau_education_moyen=(COLS["educ"], "mean"),
        indice_scolarisation_moyen=("indice_scolarisation", "mean"),
    ).reset_index()
    display(edu_region.round(3))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(data=edu_region, x=region_col, y="taux_alphabetisme", ax=axes[0], color="#16A34A")
    axes[0].set_title("Taux d'alphabetisme par region")
    axes[0].set_ylabel("% alphabetises")

    sns.barplot(data=edu_region, x=region_col, y="niveau_education_moyen", ax=axes[1], color="#2563EB")
    axes[1].set_title("Niveau moyen d'education par region")
    plt.tight_layout()
    plt.show()

    heat = edu_region.set_index(region_col)[["taux_alphabetisme", "niveau_education_moyen", "indice_scolarisation_moyen"]]
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Heatmap education par region")
    plt.show()
else:
    print("Colonnes education/alphabetisme insuffisantes.")

if milieu_col:
    display(df.groupby(milieu_col)[["indice_scolarisation"]].mean().round(4))
"""
    ),
    md(
        r"""
## 8. Analyse emploi

On calcule les taux d'emploi et de chomage par region, par milieu et par sexe lorsque les colonnes sont disponibles.
"""
    ),
    code(
        r"""
if region_col:
    emploi_region = df.groupby(region_col).agg(
        taux_emploi=("est_occupe", lambda s: s.mean() * 100),
        taux_chomage=("est_chomeur", lambda s: s.mean() * 100),
    ).reset_index()
    display(emploi_region.round(3))

    emploi_region_melt = emploi_region.melt(id_vars=region_col, var_name="indicateur", value_name="pourcentage")
    sns.barplot(data=emploi_region_melt, x=region_col, y="pourcentage", hue="indicateur")
    plt.title("Taux d'emploi et de chomage par region")
    plt.xlabel("Region")
    plt.ylabel("%")
    plt.legend(title="")
    plt.show()

if milieu_col:
    display(df.groupby(milieu_col)[["est_occupe", "est_chomeur"]].mean().mul(100).round(2))

if sexe_col:
    display(df.groupby(sexe_col)[["est_occupe", "est_chomeur"]].mean().mul(100).round(2))

secteur_col = COLS.get("secteur")
if secteur_col:
    secteur_counts = df[secteur_col].value_counts(dropna=False).head(15)
    secteur_counts.plot(kind="bar", color="#7C3AED")
    plt.title("Top secteurs d'activite")
    plt.xlabel("Secteur")
    plt.ylabel("Effectif")
    plt.show()
"""
    ),
    md(
        r"""
## 9. Analyse handicap

Cette section analyse la situation de handicap globale et par type de difficulte.
"""
    ),
    code(
        r"""
handicap_vars = [c for c in [COLS.get("sit_handicap")] + handi_cols if c]
print("Colonnes handicap utilisees:", handicap_vars)

if region_col:
    handicap_region = df.groupby(region_col).agg(
        taux_handicap=("a_handicap", lambda s: s.mean() * 100),
        score_vulnerabilite_moyen=("score_vulnerabilite_proxy", "mean"),
    ).reset_index()
    display(handicap_region.round(3))

    sns.barplot(data=handicap_region, x=region_col, y="taux_handicap", color="#DC2626")
    plt.title("Taux de handicap par region")
    plt.xlabel("Region")
    plt.ylabel("% avec handicap")
    plt.show()

if age_col:
    age_handicap = df.groupby(age_col)["a_handicap"].mean().mul(100).reset_index()
    sns.lineplot(data=age_handicap, x=age_col, y="a_handicap", marker="o", color="#DC2626")
    plt.title("Taux de handicap par tranche d'age")
    plt.xlabel("AGE5")
    plt.ylabel("% avec handicap")
    plt.show()

sns.boxplot(data=df, x="classe_vulnerabilite", y="score_vulnerabilite_proxy", color="#F59E0B")
plt.title("Score proxy selon la classe de vulnerabilite")
plt.xlabel("Classe de vulnerabilite")
plt.ylabel("Score proxy")
plt.show()
"""
    ),
    md(
        r"""
## 10. Tests statistiques avances

Chaque test est accompagne de son interpretation. Les tests utilisent un echantillon lorsque c'est necessaire pour garder le notebook rapide sur un grand fichier.
"""
    ),
    code(
        r"""
test_results = []

def interpret_pvalue(p_value: float, alpha: float = 0.05) -> str:
    if pd.isna(p_value):
        return "Test non interpretable."
    return "Relation statistiquement significative au seuil 5%." if p_value < alpha else "Pas de preuve statistique suffisante au seuil 5%."

def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str, label: str):
    tab = pd.crosstab(df[col_a], df[col_b])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return
    chi2, p_value, dof, expected = chi2_contingency(tab)
    test_results.append({
        "test": label,
        "H0": f"{col_a} et {col_b} sont independantes.",
        "H1": f"{col_a} et {col_b} sont associees.",
        "statistique": chi2,
        "p_value": p_value,
        "interpretation": interpret_pvalue(p_value),
    })

if milieu_col:
    chi_square_test(df, milieu_col, "classe_vulnerabilite", "Chi-square: milieu vs classe_vulnerabilite")

if sexe_col:
    chi_square_test(df, sexe_col, "classe_vulnerabilite", "Chi-square: sexe vs classe_vulnerabilite")

if region_col:
    groups = [g["score_vulnerabilite_proxy"].dropna().values for _, g in df.groupby(region_col)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        stat_kw, p_kw = kruskal(*groups)
        test_results.append({
            "test": "Kruskal-Wallis: score de vulnerabilite entre regions",
            "H0": "Les distributions du score sont identiques entre regions.",
            "H1": "Au moins une region presente une distribution differente.",
            "statistique": stat_kw,
            "p_value": p_kw,
            "interpretation": interpret_pvalue(p_kw),
        })

normality_sample = df["score_vulnerabilite_proxy"].dropna().sample(
    n=min(5000, df["score_vulnerabilite_proxy"].notna().sum()),
    random_state=RANDOM_STATE
)
if len(normality_sample) >= 3:
    stat_shapiro, p_shapiro = shapiro(normality_sample)
    test_results.append({
        "test": "Shapiro: normalite du score proxy sur echantillon",
        "H0": "Le score suit une distribution normale.",
        "H1": "Le score ne suit pas une distribution normale.",
        "statistique": stat_shapiro,
        "p_value": p_shapiro,
        "interpretation": interpret_pvalue(p_shapiro),
    })

tests_df = pd.DataFrame(test_results)
display(tests_df)

for _, row in tests_df.iterrows():
    display(Markdown(
        f"**{row['test']}**  \n"
        f"- H0: {row['H0']}  \n"
        f"- H1: {row['H1']}  \n"
        f"- p-value: `{row['p_value']:.4g}`  \n"
        f"- Interpretation: {row['interpretation']}"
    ))
"""
    ),
    md(
        r"""
## 11. Analyse de correlation

On etudie les correlations Pearson et Spearman entre les variables numeriques d'interet.
"""
    ),
    code(
        r"""
corr_candidates = [
    COLS.get("age5"),
    COLS.get("educ"),
    COLS.get("alphab"),
    "indice_scolarisation",
    "est_occupe",
    "est_chomeur",
    "a_handicap",
    "score_vulnerabilite_proxy",
]
corr_cols = [c for c in corr_candidates if c and c in df.columns]
corr_df = df[corr_cols].apply(pd.to_numeric, errors="coerce")

pearson_corr = corr_df.corr(method="pearson")
spearman_corr = corr_df.corr(method="spearman")

display(pearson_corr.round(3))

sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Matrice de correlation Pearson")
plt.show()

strong_pairs = []
for i, c1 in enumerate(pearson_corr.columns):
    for c2 in pearson_corr.columns[i+1:]:
        val = pearson_corr.loc[c1, c2]
        if pd.notna(val) and abs(val) >= 0.4:
            strong_pairs.append((c1, c2, val))

if strong_pairs:
    display(pd.DataFrame(strong_pairs, columns=["variable_1", "variable_2", "correlation"]).round(3))
else:
    print("Aucune correlation absolue >= 0.40 detectee parmi les variables selectionnees.")
"""
    ),
    md(
        r"""
## 12. PCA - Analyse en composantes principales

La PCA est appliquee sur les indicateurs regionaux. Elle permet de projeter les regions sur deux axes synthetiques pour faciliter l'interpretation visuelle.
"""
    ),
    code(
        r"""
def build_regional_table(df: pd.DataFrame) -> pd.DataFrame:
    if region_col is None:
        raise ValueError("La colonne region est indispensable pour l'analyse regionale.")

    agg_spec = {
        "population": (region_col, "count"),
        "score_vulnerabilite_moyen": ("score_vulnerabilite_proxy", "mean"),
        "indice_scolarisation_moyen": ("indice_scolarisation", "mean"),
        "taux_emploi": ("est_occupe", "mean"),
        "taux_chomage": ("est_chomeur", "mean"),
        "pct_handicap": ("a_handicap", "mean"),
    }
    if age_col:
        agg_spec.update({
            "age_moyen": (age_col, "mean"),
            "pct_mineurs": (age_col, lambda s: (pd.to_numeric(s, errors="coerce") < 15).mean() * 100),
            "pct_seniors": (age_col, lambda s: (pd.to_numeric(s, errors="coerce") >= 60).mean() * 100),
        })
    if COLS.get("alphab"):
        agg_spec["pct_alphabete"] = (COLS["alphab"], lambda s: (pd.to_numeric(s, errors="coerce") == 1).mean() * 100)
    if COLS.get("educ"):
        agg_spec["niv_edu_moyen"] = (COLS["educ"], "mean")

    regional = df.groupby(region_col).agg(**agg_spec).reset_index()
    if "pct_mineurs" in regional.columns and "pct_seniors" in regional.columns:
        regional["ratio_dependance"] = ((regional["pct_mineurs"] + regional["pct_seniors"]) / 100).clip(0, 1)
    return regional

stats_regions = build_regional_table(df)
display(stats_regions.round(3))

pca_features = [
    c for c in [
        "age_moyen", "pct_mineurs", "pct_seniors", "pct_alphabete",
        "niv_edu_moyen", "indice_scolarisation_moyen", "taux_emploi",
        "taux_chomage", "pct_handicap", "ratio_dependance",
        "score_vulnerabilite_moyen",
    ] if c in stats_regions.columns
]

X_reg = stats_regions[pca_features].apply(pd.to_numeric, errors="coerce")
X_reg = X_reg.fillna(X_reg.median(numeric_only=True))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reg)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_scaled)

pca_regions = stats_regions[[region_col]].copy()
pca_regions["PC1"] = coords[:, 0]
pca_regions["PC2"] = coords[:, 1]
pca_regions["variance_PC1"] = pca.explained_variance_ratio_[0]
pca_regions["variance_PC2"] = pca.explained_variance_ratio_[1]
display(pca_regions.round(4))

sns.scatterplot(data=pca_regions, x="PC1", y="PC2", s=120, color="#2563EB")
for _, row in pca_regions.iterrows():
    plt.text(row["PC1"], row["PC2"], str(int(row[region_col])), fontsize=9, ha="left", va="bottom")
plt.axhline(0, color="gray", linewidth=0.8)
plt.axvline(0, color="gray", linewidth=0.8)
plt.title(f"PCA des regions - variance expliquee: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

loadings = pd.DataFrame(pca.components_.T, index=pca_features, columns=["PC1", "PC2"]).sort_values("PC1", key=np.abs, ascending=False)
display(loadings.round(3))
"""
    ),
    md(
        r"""
## 13. Segmentation statistique regionale par K-Means

On segmente les regions a partir des indicateurs regionaux. Le nombre de clusters est choisi avec l'elbow method et le silhouette score, puis une projection PCA coloree permet de visualiser les groupes.
"""
    ),
    code(
        r"""
k_values = list(range(2, min(8, len(stats_regions) - 1) + 1))
k_metrics = []

for k in k_values:
    model = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
    k_metrics.append({"k": k, "inertia": model.inertia_, "silhouette": sil})

k_metrics_df = pd.DataFrame(k_metrics)
display(k_metrics_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.lineplot(data=k_metrics_df, x="k", y="inertia", marker="o", ax=axes[0], color="#2563EB")
axes[0].set_title("Elbow method")
sns.lineplot(data=k_metrics_df, x="k", y="silhouette", marker="o", ax=axes[1], color="#16A34A")
axes[1].set_title("Silhouette score")
plt.tight_layout()
plt.show()

best_k = int(k_metrics_df.sort_values("silhouette", ascending=False).iloc[0]["k"]) if not k_metrics_df.empty else 2
print(f"K retenu automatiquement selon silhouette: {best_k}")

kmeans_final = KMeans(n_clusters=best_k, n_init=50, random_state=RANDOM_STATE)
stats_regions["cluster_statistique"] = kmeans_final.fit_predict(X_scaled)
pca_regions["cluster_statistique"] = stats_regions["cluster_statistique"]

cluster_profile = stats_regions.groupby("cluster_statistique")[pca_features].mean().round(3)
display(cluster_profile)

sns.scatterplot(data=pca_regions, x="PC1", y="PC2", hue="cluster_statistique", palette="tab10", s=140)
for _, row in pca_regions.iterrows():
    plt.text(row["PC1"], row["PC2"], str(int(row[region_col])), fontsize=9, ha="left", va="bottom")
plt.title("PCA des regions coloree par cluster statistique")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.show()
"""
    ),
    md(
        r"""
## 14. Export des resultats

Les resultats utiles au rapport, au dashboard et a l'API sont exportes dans `outputs/`.
"""
    ),
    code(
        r"""
stats_regions.to_csv(OUTPUT_DIR / "stats_regions.csv", index=False)
pearson_corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
pca_regions.to_csv(OUTPUT_DIR / "pca_regions.csv", index=False)
stats_regions.to_csv(OUTPUT_DIR / "clusters_statistiques.csv", index=False)

report_lines = []
report_lines.append("RAPPORT STATISTIQUE AVANCE - RGPH 2014")
report_lines.append("=" * 60)
report_lines.append(f"Nombre d'individus analyses: {len(df):,}")
report_lines.append(f"Nombre de colonnes: {df.shape[1]:,}")
report_lines.append("")
report_lines.append("Colonnes detectees:")
for key, value in COLS.items():
    report_lines.append(f"- {key}: {value}")
report_lines.append("")
report_lines.append("Tests statistiques:")
if "tests_df" in globals() and not tests_df.empty:
    for _, row in tests_df.iterrows():
        report_lines.append(f"- {row['test']}: p-value={row['p_value']:.6g}; {row['interpretation']}")
report_lines.append("")
report_lines.append("Clusters statistiques regionaux:")
report_lines.append(cluster_profile.to_string())

with open(OUTPUT_DIR / "advanced_statistics_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Exports termines:")
for filename in [
    "stats_regions.csv",
    "correlation_matrix.csv",
    "advanced_statistics_report.txt",
    "pca_regions.csv",
    "clusters_statistiques.csv",
]:
    print("-", OUTPUT_DIR / filename)
"""
    ),
    md(
        r"""
## 15. Conclusion

Ce notebook fournit une base statistique avancee pour la soutenance du projet RGPH 2014. Les analyses permettent d'identifier les dimensions associees a la vulnerabilite socio-economique : milieu, education, alphabetisme, emploi, handicap et structure d'age.

Les regions les plus vulnerables et les facteurs les plus associes a la vulnerabilite doivent etre lus directement a partir des tableaux calcules ci-dessus, notamment `stats_regions`, `cluster_profile`, la matrice de correlation et les tests statistiques.

Limites principales :

- Le score de vulnerabilite est un proxy academique, construit a partir des variables disponibles.
- Les resultats ne remplacent pas un indicateur officiel du HCP.
- Les tests statistiques indiquent des associations, pas des causalites.
- Les valeurs manquantes conditionnelles doivent etre interpretees avec prudence.

Utilite pour le projet :

- `stats_regions.csv` peut alimenter le dashboard regional.
- `correlation_matrix.csv` documente les relations entre indicateurs.
- `clusters_statistiques.csv` complete la segmentation regionale.
- `advanced_statistics_report.txt` fournit une synthese exploitable dans le rapport SFE.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(NOTEBOOK_PATH.resolve())
