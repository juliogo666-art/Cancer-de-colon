import asyncio
from types import SimpleNamespace

import numpy as np

from src.api import main_api


class FakeRiskModel:
    def __init__(self, probabilities):
        self.probabilities = np.array([probabilities], dtype=float)
        self.last_features = None

    def predict_proba(self, features):
        self.last_features = features
        return self.probabilities


class FakePredictionLogger:
    def __init__(self):
        self.risk_predictions = []

    def log_risk_prediction(self, **kwargs):
        self.risk_predictions.append(kwargs)


def make_request_state(modelo_ml=None, modelo_ml_triage=None):
    state = SimpleNamespace(
        modelo_ml=modelo_ml,
        modelo_ml_triage=modelo_ml_triage,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_predict_risk_uses_full_clinical_model_when_analytics_are_present(monkeypatch):
    model = FakeRiskModel([0.10, 0.20, 0.70])
    fake_logger = FakePredictionLogger()
    monkeypatch.setattr(main_api, "logger", fake_logger)

    result = asyncio.run(
        main_api.predict_risk(
            request=make_request_state(modelo_ml=model),
            patient_id=123,
            smoking=8,
            alcohol_use=5,
            obesity=6,
            family_history=1,
            diet_red_meat=7,
            diet_salted_processed=6,
            fruit_veg_intake=2,
            physical_activity=3,
            bmi=31.5,
            fobt_resultado=1,
            cea_level=8.2,
        )
    )

    assert result["risk_level"] == "High"
    assert result["risk_score"] == 0.795
    assert result["probabilities"] == {"Low": 0.1, "Medium": 0.2, "High": 0.7}
    assert model.last_features.shape == (1, 11)
    assert "FOBT_Resultado_n" in result["features_used"]
    assert "CEA_Level_ng_mL" in result["features_used"]
    assert fake_logger.risk_predictions[0]["patient_id"] == 123
    assert fake_logger.risk_predictions[0]["risk_level"] == "High"


def test_predict_risk_uses_triage_model_when_analytics_are_missing(monkeypatch):
    triage_model = FakeRiskModel([0.75, 0.20, 0.05])
    fake_logger = FakePredictionLogger()
    monkeypatch.setattr(main_api, "logger", fake_logger)

    result = asyncio.run(
        main_api.predict_risk(
            request=make_request_state(modelo_ml_triage=triage_model),
            patient_id=None,
            smoking=2,
            alcohol_use=1,
            obesity=1,
            family_history=0,
            diet_red_meat=2,
            diet_salted_processed=2,
            fruit_veg_intake=8,
            physical_activity=7,
            bmi=23.0,
            fobt_resultado=-1,
            cea_level=-1.0,
        )
    )

    assert result["risk_level"] == "Low"
    assert triage_model.last_features.shape == (1, 9)
    assert len(result["features_used"]) == 9
    assert "FOBT_Resultado_n" not in result["features_used"]
    assert "CEA_Level_ng_mL" not in result["features_used"]
    assert fake_logger.risk_predictions[0]["risk_level"] == "Low"
