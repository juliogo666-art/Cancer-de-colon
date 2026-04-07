"""
Precision — De todas las alertas que el modelo lanzó, ¿cuántas eran reales?

En contexto médico: si el modelo dice que 100 pacientes tienen riesgo alto,
¿cuántos de esos 100 realmente lo tenían?
"""

from typing import Optional
import numpy as np
from sklearn.metrics import precision_score


class PrecisionMetric:
    """
    Calcula Precision por clase usando macro-average por defecto
    (trata todas las clases por igual, importante cuando las clases
    están desbalanceadas como Low vs High).
    """

    def __init__(self, average: str = "macro"):
        self.average = average

    @property
    def name(self) -> str:
        return f"Precision ({self.average})"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> float:
        return float(precision_score(y_true, y_pred, average=self.average, zero_division=0))
