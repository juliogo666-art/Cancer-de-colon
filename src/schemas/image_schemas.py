"""
Pydantic para respuestas de análisis de imagen.

Estructura las respuestas tanto de colonoscopia (detección de pólipos)
como de biopsias (clasificación benigno/maligno).
"""

from typing import Optional
from pydantic import BaseModel, Field


class ImageAnalysisResponse(BaseModel):
    """
    Respuesta estandarizada para cualquier análisis de imagen médica.
    Aplica tanto a colonoscopia como a biopsias.
    """

    diagnosis: str = Field(
        ...,
        description="Diagnóstico: 'POLIPO DETECTADO', 'TEJIDO SANO', 'MALIGNO', 'BENIGNO'",
    )
    is_positive: bool = Field(
        ..., description="True si se detectó hallazgo (pólipo o malignidad)"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confianza de la predicción (0-1)"
    )
    raw_prediction: float = Field(..., description="Valor crudo de salida del modelo")
    recommendation: str = Field(
        ..., description="Recomendación clínica basada en el resultado"
    )
    gradcam_base64: Optional[str] = Field(
        None, description="Mapa de calor Grad-CAM codificado en base64"
    )


class ColonoscopyAnalysisResponse(ImageAnalysisResponse):
    """Respuesta específica para análisis de colonoscopia."""

    analysis_type: str = Field(default="colonoscopy", description="Tipo de análisis")
    is_polyp: bool = Field(..., description="True si se detectó pólipo")


class BiopsyAnalysisResponse(ImageAnalysisResponse):
    """Respuesta específica para análisis de biopsia."""

    analysis_type: str = Field(default="biopsy", description="Tipo de análisis")
    is_benign: bool = Field(..., description="True si el tejido es benigno")
