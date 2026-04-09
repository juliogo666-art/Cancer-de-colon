"""
F-Score — Media harmónica ponderada entre Precision y Recall.

    - F1: Equilibrio 50/50 entre Precision y Recall.
    - F2: Da el DOBLE de importancia al Recall → ideal para diagnóstico médico
           porque preferimos detectar un enfermo ya que este sano (FP) a NO detectar uno
           enfermo de verdad (FN).
"""

from typing import Optional
import numpy as np
from sklearn.metrics import fbeta_score


class FBetaMetric:
    """
    Calcula F-Beta Score con beta configurable.

    Parameters
    ----------
    beta : float
        1.0 → F1-Score (equilibrio)
        2.0 → F2-Score (prioriza Recall, recomendado para cáncer)
    average : str
        'macro', 'weighted', 'micro', o None para desglose por clase.
    """

    def __init__(self, beta: float = 2.0, average: str = "macro"):
        self.beta = beta
        self.average = average

    @property
    def name(self) -> str:
        return f"F{self.beta:.0f}-Score ({self.average})"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> float:
        return float(
            fbeta_score(
                y_true, y_pred, beta=self.beta, average=self.average, zero_division=0
            )
        )
