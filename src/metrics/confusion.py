"""
Confusion Matrix — Desglose completo de TP, TN, FP, FN por clase.

Devuelve un diccionario con la matriz y el desglose numérico, especialmente
relevante para identificar Falsos Negativos en la clase "High" (pacientes
con cáncer que el modelo no detectó).
"""

from typing import Optional
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class ConfusionMatrixMetric:
    """
    Calcula la matriz de confusión completa y un informe de clasificación.

    Parameters
    ----------
    labels : list[str], optional
        Nombres de las clases. Por defecto: ["Low", "Medium", "High"].
    """

    def __init__(self, labels: list[str] | None = None):
        self.labels = labels or ["Low", "Medium", "High"]

    @property
    def name(self) -> str:
        return "Confusion Matrix"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> dict:
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=self.labels[:len(np.unique(np.concatenate([y_true, y_pred])))],
            output_dict=True,
            zero_division=0,
        )

        return {
            "matrix": cm.tolist(),
            "labels": self.labels[:cm.shape[0]],
            "classification_report": report,
        }
