import asyncio
import numpy as np

from main import IndividuInput, model_store, predict_individu


class DummyXGBModel:
    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, _):
        return np.asarray([self.proba], dtype=float)


FEATURES = [
    "reg", "mil", "sexe", "AGE5", "LIEN_CM",
    "SIT_HANDICAP", "NIV_ET_AGR", "LIR_ECR",
    "a_diplome_eg", "TY_ACT", "indice_scolarisation",
    "est_occupe", "est_chomeur", "a_handicap",
    "ENF_VIV", "ENF_DEC",
]


def test_predict_individu_uses_continuous_vulnerability_score():
    original = model_store.xgb_bundle
    model_store.xgb_bundle = {"model": DummyXGBModel([1, 0, 0]), "features": FEATURES}
    try:
        result = asyncio.run(predict_individu(IndividuInput(
            reg=1, mil=1, sexe=1, age5=32,
            niv_et_agr=4, lir_ecr=1, ty_act=0,
            sit_handicap=1, enf_viv=None, enf_dec=None,
        )))
        assert result["classe"] == 0
        assert result["score_continu"] < 0.1
        assert result["age5_band"] == 30
        assert result["probas"]["non_vulnerable"] < 1.0
    finally:
        model_store.xgb_bundle = original


def test_predict_individu_high_risk_profile_is_vulnerable():
    original = model_store.xgb_bundle
    model_store.xgb_bundle = {"model": DummyXGBModel([0, 0, 1]), "features": FEATURES}
    try:
        result = asyncio.run(predict_individu(IndividuInput(
            reg=8, mil=2, sexe=2, age5=70,
            niv_et_agr=0, lir_ecr=3, ty_act=5,
            sit_handicap=4, enf_viv=None, enf_dec=None,
        )))
        assert result["classe"] == 2
        assert result["score_continu"] > 0.9
    finally:
        model_store.xgb_bundle = original
