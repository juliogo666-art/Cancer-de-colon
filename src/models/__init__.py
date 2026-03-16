"""
Módulo de modelos del proyecto de Cáncer de Colón.

Contiene los modelos de clasificación y detección de pólipos.
"""

from src.models.modelo_busca_polipos_Clas import PolypDetector, ColonoscopyDataset
from src.models.testeos import entrenar_random_forest

__all__ = [
    "PolypDetector",
    "ColonoscopyDataset",
    "entrenar_random_forest",
]
