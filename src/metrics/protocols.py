"""
=============================================================================
ARCHIVO: protocols.py (Interfaces / Contratos de OOP)
=============================================================================
Define el contrato (Protocol) para las métricas de clasificación clínica.

Aquí evaluamos modelos de clasificación médica
que predicen el riesgo de cáncer de colon (Low / Medium / High).

Cualquier métrica que implemente este protocolo puede integrarse
automáticamente en el EvaluationPipeline.
=============================================================================
"""

from typing import Optional, Protocol

import numpy as np


class ClassificationMetricProtocol(Protocol):
    """
    Interfaz que deben cumplir todas las métricas de clasificación clínica.

    Parámetros estándar
    -------------------
    y_true : np.ndarray
        Etiquetas reales (ground truth). Ej: [0, 1, 2, 1, 0, ...]
    y_pred : np.ndarray
        Predicciones del modelo. Ej: [0, 1, 1, 1, 0, ...]
    y_proba : np.ndarray, optional
        Probabilidades de cada clase, shape (n_samples, n_classes).
        Necesario para métricas como ROC-AUC.

    Devuelve
    --------
    float | dict
        El resultado de la métrica. Puede ser un valor escalar o un
        diccionario con desglose por clase.
    """

    @property
    def name(self) -> str:
        """Nombre legible de la métrica."""
        ...

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> float | dict:
        """Calcula la métrica a partir de las predicciones."""
        ...
