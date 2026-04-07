"""
Módulo de schemas Pydantic para validación de datos del proyecto Cancer de Colon.

Schemas disponibles:
    Pacientes:
        - PatientCreate, PatientUpdate, PatientResponse, PatientListResponse

    Predicciones:
        - RiskPredictionRequest, RiskPredictionResponse, SHAPResponse

    Imágenes:
        - ImageAnalysisResponse, ColonoscopyAnalysisResponse, BiopsyAnalysisResponse
"""

from src.schemas.patient_schemas import (
    PatientBase,
    PatientCreate,
    PatientUpdate,
    PatientResponse,
    PatientListResponse,
)
from src.schemas.prediction_schemas import (
    RiskPredictionRequest,
    RiskProbabilities,
    RiskPredictionResponse,
    SHAPExplanation,
    SHAPResponse,
)
from src.schemas.image_schemas import (
    ImageAnalysisResponse,
    ColonoscopyAnalysisResponse,
    BiopsyAnalysisResponse,
)

__all__ = [
    # Pacientes
    "PatientBase",
    "PatientCreate",
    "PatientUpdate",
    "PatientResponse",
    "PatientListResponse",
    # Predicciones
    "RiskPredictionRequest",
    "RiskProbabilities",
    "RiskPredictionResponse",
    "SHAPExplanation",
    "SHAPResponse",
    # Imágenes
    "ImageAnalysisResponse",
    "ColonoscopyAnalysisResponse",
    "BiopsyAnalysisResponse",
]
