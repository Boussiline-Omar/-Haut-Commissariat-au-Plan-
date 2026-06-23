import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_INDIVIDU = PROJECT_ROOT / "Individu.csv"
OUTPUT_PROFILES = PROJECT_ROOT / "outputs" / "regional_profiles.csv"

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

def compute_regional_indicators(input_path=DATA_INDIVIDU, output_path=OUTPUT_PROFILES):
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Loading {input_path}...")
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found.")

    df = pd.read_csv(input_path, low_memory=False)

    # Ensure pds exists for population weighting
    if "pds" not in df.columns:
        print("Warning: 'pds' column missing, using row counts as population.")
        df["pds"] = 1.0
    df["pds"] = pd.to_numeric(df["pds"], errors="coerce").fillna(0.0)

    # --- Working Age Definition ---
    # In this project AGE5 stores the lower bound of a 5-year age band:
    # 0, 5, 10, 15, ..., 75. It is not age // 5.
    df["AGE5"] = pd.to_numeric(df["AGE5"], errors="coerce")
    df["is_working_age"] = (df["AGE5"] >= 15) & (df["AGE5"] < 65)
    df["is_minor"] = df["AGE5"] < 15
    df["is_senior"] = df["AGE5"] >= 65

    # --- Activity Definition ---
    # TY_ACT: 0=occupied, 1-2=unemployed, 3-5=inactive
    df["TY_ACT"] = pd.to_numeric(df["TY_ACT"], errors="coerce")
    df["is_occupied"] = ((df["TY_ACT"] == 0) & df["is_working_age"]).astype(float)
    df["is_unemployed"] = (df["TY_ACT"].isin([1, 2]) & df["is_working_age"]).astype(float)
    df["is_active"] = (df["TY_ACT"].isin([0, 1, 2]) & df["is_working_age"]).astype(float)

    print("Aggregating by region...")

    # We need weighted means for rates.
    # Weighted Mean = sum(value * weight) / sum(weight)

    def weighted_mean(group, col):
        if col not in group.columns:
            return np.nan
        weights = group["pds"]
        vals = pd.to_numeric(group[col], errors="coerce")
        valid = vals.notna() & weights.notna()
        weight_sum = weights[valid].sum()
        return (vals[valid] * weights[valid]).sum() / weight_sum if weight_sum > 0 else np.nan

    def numeric_series(group, col, default=np.nan):
        if col in group.columns:
            return pd.to_numeric(group[col], errors="coerce")
        return pd.Series(default, index=group.index)

    regions = []
    for reg_id in range(1, 13):
        reg_df = df[df["reg"] == reg_id]
        if reg_df.empty:
            continue

        # Population
        pop = reg_df["pds"].sum()

        # 1. Taux d'emploi (Occupied / Working-Age Pop)
        # Working age pop for this region
        wa_pop_df = reg_df[reg_df["is_working_age"]]
        wa_pop = wa_pop_df["pds"].sum()

        occ_pop = (reg_df["is_occupied"] * reg_df["pds"]).sum()
        taux_emploi = occ_pop / wa_pop if wa_pop > 0 else np.nan

        # 2. Taux de chômage (Unemployed / Active Pop)
        active_pop = (reg_df["is_active"] * reg_df["pds"]).sum()
        unemp_pop = (reg_df["is_unemployed"] * reg_df["pds"]).sum()
        taux_chomage = unemp_pop / active_pop if active_pop > 0 else np.nan

        # 3. Niv Edu Moyen (Weighted average of NIV_ET_AGR)
        # Using raw codes as ordinal approximation
        niv_edu = weighted_mean(reg_df, "NIV_ET_AGR")

        # 4. Pct Alphabete (LIR_ECR == 1)
        # 1: reads and writes
        lir_ecr = numeric_series(reg_df, "LIR_ECR")
        alpha_pop = ((lir_ecr == 1) * reg_df["pds"]).sum()
        pct_alpha = alpha_pop / pop if pop > 0 else np.nan

        # 5. Pct Handicap (SIT_HANDICAP > 1)
        sit_handicap = numeric_series(reg_df, "SIT_HANDICAP")
        handi_pop = ((sit_handicap > 1) * reg_df["pds"]).sum()
        pct_handi = handi_pop / pop if pop > 0 else np.nan

        # 6. Ratio Dépendance ((Minors + Seniors) / Working Age Pop)
        dependents_pop = ((reg_df["is_minor"] | reg_df["is_senior"]) * reg_df["pds"]).sum()
        ratio_dep = dependents_pop / wa_pop if wa_pop > 0 else np.nan

        # 7. Age Moyen (Weighted)
        # Approximate actual age as the midpoint of the AGE5 band.
        real_age = reg_df["AGE5"] + 2.5
        age_moyen = (real_age * reg_df["pds"]).sum() / pop if pop > 0 else np.nan

        # 8. Population components for K-Means
        pct_mineurs = ((reg_df["is_minor"]) * reg_df["pds"]).sum() / pop * 100 if pop > 0 else np.nan
        pct_seniors = ((reg_df["is_senior"]) * reg_df["pds"]).sum() / pop * 100 if pop > 0 else np.nan
        sexe = numeric_series(reg_df, "sexe")
        pct_femmes = ((sexe == 2) * reg_df["pds"]).sum() / pop * 100 if pop > 0 else np.nan

        regions.append({
            "reg": reg_id,
            "region_label": REGION_NAMES[reg_id],
            "profil_label": REGION_NAMES[reg_id],
            "population": pop,
            "age_moyen": age_moyen,
            "pct_mineurs": pct_mineurs,
            "pct_seniors": pct_seniors,
            "pct_femmes": pct_femmes,
            "niv_edu_moyen": niv_edu,
            "pct_alphabete": pct_alpha,
            "taux_emploi": taux_emploi,
            "taux_chomage": taux_chomage,
            "pct_handicap": pct_handi,
            "ratio_dependance": ratio_dep,
        })

    df_res = pd.DataFrame(regions)

    # Save outputs
    os.makedirs(output_path.parent, exist_ok=True)
    df_res.to_csv(output_path, index=False)
    print(f"Successfully saved regional profiles to {output_path}")
    return df_res

if __name__ == "__main__":
    compute_regional_indicators()
