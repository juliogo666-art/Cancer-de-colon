"""
Pydantic para peticiones y respuestas de predicción ML.

Valida los factores de riesgo que recibe el endpoint de predicción
y estructura la respuesta con probabilidades y nivel de riesgo.
"""

from typing import Optional
from pydantic import BaseModel, Field


class RiskPredictionRequest(BaseModel):
    """
    Datos clínicos de un paciente para predicción de riesgo.
    Estos son los 11 factores que el modelo LightGBM espera recibir.
    """

    smoking: float = Field(..., ge=0, le=10, description="Nivel de tabaquismo (0-10)")
    alcohol_use: float = Field(
        ..., ge=0, le=10, description="Consumo de alcohol (0-10)"
    )
    obesity: float = Field(..., ge=0, le=10, description="Nivel de obesidad (0-10)")
    family_history: int = Field(
        ..., ge=0, le=1, description="Historial familiar (0=No, 1=Sí)"
    )
    diet_red_meat: float = Field(
        ..., ge=0, le=10, description="Consumo de carne roja (0-10)"
    )
    diet_salted_processed: float = Field(
        ..., ge=0, le=10, description="Consumo de sal/procesados (0-10)"
    )
    fruit_veg_intake: float = Field(
        ..., ge=0, le=10, description="Consumo de fruta/verdura (0-10)"
    )
    physical_activity: float = Field(
        ..., ge=0, le=10, description="Actividad física (0-10)"
    )
    bmi: float = Field(..., ge=10, le=60, description="Índice de masa corporal")
    fobt_resultado: int = Field(
        ..., ge=0, le=1, description="FOBT sangre oculta (0=Negativo, 1=Positivo)"
    )
    cea_level: float = Field(
        ..., ge=0, description="Nivel de marcador tumoral CEA (ng/mL)"
    )


class RiskProbabilities(BaseModel):
    """Probabilidades desglosadas por nivel de riesgo."""

    low: float = Field(..., ge=0, le=1, description="Probabilidad de riesgo bajo")
    medium: float = Field(..., ge=0, le=1, description="Probabilidad de riesgo medio")
    high: float = Field(..., ge=0, le=1, description="Probabilidad de riesgo alto")


class RiskPredictionResponse(BaseModel):
    """Respuesta del endpoint de predicción de riesgo."""

    risk_level: str = Field(..., description="Nivel predicho: Low, Medium o High")
    risk_score: float = Field(
        ..., ge=0, le=1, description="Score ponderado de riesgo (0-1)"
    )
    probabilities: RiskProbabilities
    features_used: dict[str, float] = Field(
        ..., description="Features utilizadas en la predicción"
    )


class SHAPExplanation(BaseModel):
    """Explicación SHAP de una predicción individual."""

    feature_name: str
    shap_value: float
    impact_percentage: float = Field(
        ..., description="Porcentaje de impacto sobre el total"
    )
    direction: str = Field(..., description="'Sube riesgo' o 'Baja riesgo'")


class SHAPResponse(BaseModel):
    """Respuesta con explicación SHAP completa."""

    patient_id: Optional[int] = None
    risk_level: str
    explanations: list[SHAPExplanation]
