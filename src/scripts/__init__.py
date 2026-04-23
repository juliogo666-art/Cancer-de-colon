"""
Módulo de scripts del proyecto de Cáncer de Colón.

Contiene EDA, limpieza de datos y generación de datos sintéticos.
"""

from src.scripts.eda import eda
from src.scripts.data_cleaning import (
    limpiar_datos_globales,
    limpiar_datos_sinteticos,
    limpiar_datos_kaggle,
    combinar_datos_s_g,
    limpiar_datos_kaggle_finales,
)
from src.scripts.sintetiza_historiales import (
    generar_datos_sinteticos,
    sintetizar_historiales,
    sintetizar_datos_kaggle,
)

__all__ = [
    "eda",
    "limpiar_datos_globales",
    "limpiar_datos_sinteticos",
    "limpiar_datos_kaggle",
    "combinar_datos_s_g",
    "limpiar_datos_kaggle_finales",
    "generar_datos_sinteticos",
    "sintetizar_historiales",
    "sintetizar_datos_kaggle",
]
