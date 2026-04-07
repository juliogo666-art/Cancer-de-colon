"""Accuracy — Porcentaje de predicciones correctas sobre el total."""

from typing import Optional
import numpy as np
from sklearn.metrics import accuracy_score


class AccuracyMetric:
    """Calcula la exactitud global del modelo."""

    @property
    def name(self) -> str:
        return "Accuracy"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> float:
        return float(accuracy_score(y_true, y_pred))
