"""
Recall — De todos los pacientes que realmente tenían cáncer, ¿cuántos detectó el modelo?

Esta es la métrica MÁS CRÍTICA en diagnóstico médico:
    - Un Recall bajo en la clase "High" significa pacientes enfermos NO detectados (FN).
    - En oncología, un falso negativo puede costar una vida.
    - Por eso priorizamos F2-Score (que da más peso al Recall) sobre F1.
"""

from typing import Optional
import numpy as np
from sklearn.metrics import recall_score


class RecallMetric:
    """
    Calcula Recall (Sensibilidad) por clase.

    Parameters
    ----------
    average : str
        'macro': Media aritmética de recall de cada clase.
        'weighted': Media ponderada por el número de muestras de cada clase.
        None: Devuelve un array con el recall de cada clase individualmente.
    """

    def __init__(self, average: str = "macro"):
        self.average = average

    @property
    def name(self) -> str:
        return f"Recall ({self.average})"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> float:
        return float(recall_score(y_true, y_pred, average=self.average, zero_division=0))
