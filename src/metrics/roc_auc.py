"""
ROC-AUC — Área bajo la curva ROC (Receiver Operating Characteristic).

Mide la capacidad del modelo para discriminar entre clases.
    - AUC = 1.0 → Discriminación perfecta
    - AUC = 0.5 → Equivalente a tirar una moneda (inútil)

Para clasificación multiclase (Low/Medium/High), usamos One-vs-Rest
que calcula una curva ROC por cada clase y promedia.

NOTA: Esta métrica REQUIERE las probabilidades (y_proba), no solo
las predicciones duras (y_pred).
"""

from typing import Optional
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


class ROCAUCMetric:
    """
    Calcula el área bajo la curva ROC.

    Parameters
    ----------
    multi_class : str
        'ovr' (One-vs-Rest) o 'ovo' (One-vs-One).
    average : str
        'macro' o 'weighted'.
    """

    def __init__(self, multi_class: str = "ovr", average: str = "macro"):
        self.multi_class = multi_class
        self.average = average

    @property
    def name(self) -> str:
        return f"ROC-AUC ({self.average})"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> float:
        if y_proba is None:
            # Sin probabilidades no podemos calcular ROC-AUC
            return 0.0

        try:
            n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 2

            if n_classes == 2:
                # Clasificación binaria: usamos la probabilidad de la clase positiva
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                # Multiclase: One-vs-Rest
                return float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class=self.multi_class,
                        average=self.average,
                    )
                )
        except ValueError:
            # Puede fallar si alguna clase no está representada en y_true
            return 0.0
