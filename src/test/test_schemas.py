"""
=============================================================================
Tests para los esquemas Pydantic de validación
=============================================================================
Comprueban que los schemas de pacientes, predicciones e imágenes:
    1. Aceptan datos válidos sin errores
    2. Rechazan datos inválidos (BMI negativo, Smoking > 10, etc.)
    3. Los campos opcionales pueden ser None

Ejecutar con:
    pytest src/test/test_schemas.py -v
=============================================================================
"""

import pytest
from pydantic import ValidationError

from src.schemas.patient_schemas import PatientCreate, PatientUpdate, PatientResponse
from src.schemas.prediction_schemas import (
    RiskPredictionRequest,
    RiskPredictionResponse,
    RiskProbabilities,
)
from src.schemas.image_schemas import ImageAnalysisResponse, ColonoscopyAnalysisResponse


###############################################################################
# Test de schemas de pacientes
###############################################################################


class TestPatientCreate:
    """Comprueba la validación al crear un paciente nuevo."""

    def test_paciente_valido(self):
        """Un paciente con datos correctos debe crearse sin errores."""
        paciente = PatientCreate(
            patient_id=1,
            smoking=5,
            alcohol_use=3,
            obesity=2,
            family_history=1,
            diet_red_meat=4,
            diet_salted_processed=3,
            fruit_veg_intake=7,
            physical_activity=6,
            bmi=25.5,
            overall_risk_score=0.3,
            risk_level="Low",
        )
        assert paciente.patient_id == 1
        assert paciente.smoking == 5

    def test_smoking_mayor_que_10_falla(self):
        """Smoking no puede ser mayor que 10."""
        with pytest.raises(ValidationError):
            PatientCreate(patient_id=1, smoking=15)

    def test_bmi_negativo_falla(self):
        """El BMI no puede ser menor que 10."""
        with pytest.raises(ValidationError):
            PatientCreate(patient_id=1, bmi=5.0)

    def test_family_history_solo_0_o_1(self):
        """Family history solo acepta 0 (No) o 1 (Sí)."""
        with pytest.raises(ValidationError):
            PatientCreate(patient_id=1, family_history=3)


class TestPatientUpdate:
    """Comprueba la validación al actualizar un paciente."""

    def test_todos_los_campos_opcionales(self):
        """Se puede crear un PatientUpdate sin ningún campo (todo None)."""
        actualizacion = PatientUpdate()
        assert actualizacion.smoking is None
        assert actualizacion.bmi is None

    def test_actualizar_solo_un_campo(self):
        """Se puede actualizar solo el BMI sin tocar nada más."""
        actualizacion = PatientUpdate(bmi=30.0)
        assert actualizacion.bmi == 30.0
        assert actualizacion.smoking is None  # Los demás siguen en None


###############################################################################
# Test de schemas de prediccion
###############################################################################


class TestRiskPredictionRequest:
    """Comprueba la validación de los datos de entrada para predicción."""

    def test_request_valido(self):
        """Una petición con datos correctos debe funcionar."""
        request = RiskPredictionRequest(
            smoking=7.0,
            alcohol_use=5.0,
            obesity=3.0,
            family_history=1,
            diet_red_meat=6.0,
            diet_salted_processed=4.0,
            fruit_veg_intake=3.0,
            physical_activity=2.0,
            bmi=32.0,
            fobt_resultado=1,
            cea_level=5.5,
        )
        assert request.smoking == 7.0
        assert request.family_history == 1

    def test_cea_negativo_falla(self):
        """El nivel de CEA no puede ser negativo."""
        with pytest.raises(ValidationError):
            RiskPredictionRequest(
                smoking=0,
                alcohol_use=0,
                obesity=0,
                family_history=0,
                diet_red_meat=0,
                diet_salted_processed=0,
                fruit_veg_intake=0,
                physical_activity=0,
                bmi=25.0,
                fobt_resultado=0,
                cea_level=-1.0,  # ← Esto debe fallar
            )


class TestRiskPredictionResponse:
    """Comprueba que la respuesta de predicción se estructura bien."""

    def test_response_valido(self):
        respuesta = RiskPredictionResponse(
            risk_level="High",
            risk_score=0.85,
            probabilities=RiskProbabilities(low=0.05, medium=0.10, high=0.85),
            features_used={"Smoking": 8.0, "BMI": 35.0},
        )
        assert respuesta.risk_level == "High"
        assert respuesta.probabilities.high == 0.85


###############################################################################
# Test de schemas de imagen
###############################################################################


class TestImageAnalysisResponse:
    """Comprueba la respuesta de análisis de imagen."""

    def test_response_colonoscopia(self):
        """Una respuesta de colonoscopia debe incluir is_polyp."""
        respuesta = ColonoscopyAnalysisResponse(
            diagnosis="POLIPO DETECTADO",
            is_positive=True,
            confidence=0.92,
            raw_prediction=0.08,
            recommendation="Se recomienda revisión inmediata.",
            is_polyp=True,
        )
        assert respuesta.is_polyp is True
        assert respuesta.analysis_type == "colonoscopy"

    def test_gradcam_base64_es_opcional(self):
        """El campo gradcam_base64 puede ser None."""
        respuesta = ImageAnalysisResponse(
            diagnosis="TEJIDO SANO",
            is_positive=False,
            confidence=0.95,
            raw_prediction=0.95,
            recommendation="Todo normal.",
            gradcam_base64=None,
        )
        assert respuesta.gradcam_base64 is None
