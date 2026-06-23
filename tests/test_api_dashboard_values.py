import asyncio
import pandas as pd

from main import get_regional_segmentation, model_store


def test_api_consistency():
    """Verify that API values are emitted from the regional dataframe exactly."""
    regional_df = pd.DataFrame([
        {
            "reg": 1,
            "region_label": "Region A",
            "profil_label": "Region A",
            "cluster": 2,
            "cluster_label": "Cluster 2 - Test",
            "taux_emploi": 0.42,
            "taux_chomage": 0.13,
            "niv_edu_moyen": 3.1,
            "pct_alphabete": 0.76,
            "ratio_dependance": 0.55,
            "pct_handicap": 0.08,
            "age_moyen": 36.5,
        }
    ])

    original = model_store.regional_df
    model_store.regional_df = regional_df
    try:
        resp = asyncio.run(get_regional_segmentation())
        region = resp["regions"][0]
        assert region["reg_id"] == 1
        assert region["taux_emploi"] == 0.42
        assert region["cluster"] == 2
        assert region["cluster_label"] == "Cluster 2 - Test"
    finally:
        model_store.regional_df = original
