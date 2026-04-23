"""
Módulo de pipelines del proyecto de Cáncer de Colón.

Pipelines disponibles:
    - ModelEvaluationPipeline : Evaluación de modelos con métricas clínicas
    - TrainingPipeline        : Entrenamiento unificado (carga → train → eval → save)
    - ImageAnalysisPipeline   : Análisis de imagen (colonoscopia + biopsias)
"""

from src.pipelines.evaluation_pipeline import ModelEvaluationPipeline
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.image_pipeline import ImageAnalysisPipeline, ImageAnalysisResult

__all__ = [
    "ModelEvaluationPipeline",
    "TrainingPipeline",
    "ImageAnalysisPipeline",
    "ImageAnalysisResult",
]
