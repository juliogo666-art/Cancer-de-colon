"""
Módulo de métricas de clasificación clínica para cáncer de colon.

Métricas disponibles:
    - AccuracyMetric     : Exactitud global
    - PrecisionMetric    : Precisión por clase
    - RecallMetric       : Sensibilidad por clase (la más crítica en oncología)
    - FBetaMetric        : F1/F2 Score (F2 recomendado para diagnóstico médico)
    - ConfusionMatrixMetric : Matriz de confusión + classification report
    - ROCAUCMetric       : Área bajo curva ROC (requiere probabilidades)

Protocolo:
    - ClassificationMetricProtocol : Interfaz que todas las métricas cumplen
"""

from src.metrics.protocols import ClassificationMetricProtocol
from src.metrics.accuracy import AccuracyMetric
from src.metrics.precision import PrecisionMetric
from src.metrics.recall import RecallMetric
from src.metrics.f_score import FBetaMetric
from src.metrics.confusion import ConfusionMatrixMetric
from src.metrics.roc_auc import ROCAUCMetric

__all__ = [
    "ClassificationMetricProtocol",
    "AccuracyMetric",
    "PrecisionMetric",
    "RecallMetric",
    "FBetaMetric",
    "ConfusionMatrixMetric",
    "ROCAUCMetric",
]
