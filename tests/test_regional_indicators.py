import pandas as pd
from scripts.generate_regional_profiles import compute_regional_indicators


def test_indicators_logic(tmp_path):
    """
    Test that the logic in generate_regional_profiles.py
    calculates rates using the correct denominators.
    """
    # Create a dummy Individu.csv.
    # Reg 1:
    # - 100 people total
    # - 60 working age (15-64)
    # - 40 occupied
    # - 10 unemployed
    # - 10 inactive
    # - 40 minors/seniors

    data = []
    # Working age population (60 people)
    for i in range(60):
        act = 0 if i < 40 else (1 if i < 50 else 3)
        data.append({"reg": 1, "pds": 1, "AGE5": 25, "TY_ACT": act, "SIT_HANDICAP": 1, "LIR_ECR": 1, "NIV_ET_AGR": 3, "sexe": 1})
    # Non working age (40 people)
    for i in range(40):
        age5 = 10 if i < 20 else 70
        data.append({"reg": 1, "pds": 1, "AGE5": age5, "TY_ACT": 5, "SIT_HANDICAP": 1, "LIR_ECR": 1, "NIV_ET_AGR": 0, "sexe": 1})

    df = pd.DataFrame(data)
    input_path = tmp_path / "Individu.csv"
    output_path = tmp_path / "regional_profiles.csv"
    df.to_csv(input_path, index=False)

    compute_regional_indicators(input_path=input_path, output_path=output_path)
    res = pd.read_csv(output_path)
    reg1 = res[res["reg"] == 1].iloc[0]

    # Correct calculations:
    # Taux emploi = occupied / working_age = 40 / 60 = 0.666...
    # Taux chômage = unemployed / (occupied + unemployed) = 10 / (40 + 10) = 0.2

    assert abs(reg1["taux_emploi"] - 0.6666) < 0.01
    assert abs(reg1["taux_chomage"] - 0.2) < 0.01
    assert abs(reg1["ratio_dependance"] - (40 / 60)) < 0.01
    assert 0 <= reg1["taux_emploi"] <= 1
    assert 0 <= reg1["taux_chomage"] <= 1
    assert 0 <= reg1["age_moyen"] <= 100

if __name__ == "__main__":
    test_indicators_logic()
