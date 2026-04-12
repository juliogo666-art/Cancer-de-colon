"""
dependencies.py — Carga centralizada de modelos de IA para el backend.

Este módulo gestiona el ciclo de vida de los modelos (ML y DL) que se cargan
una sola vez al arrancar la API y se comparten entre todos los endpoints.

Modelos gestionados:
    1. LightGBM clínico  → Predicción de nivel de riesgo (Low/Medium/High)
    2. CNN TensorFlow     → Detección de pólipos en colonoscopia
    3. ResNet18 PyTorch   → Clasificación de biopsias (benigno/maligno)
"""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from contextlib import asynccontextmanager
from fastapi import FastAPI


###############################################################################
# Configuraciones y Rutas Globales (Desde settings.py)
###############################################################################
from src.config.settings import settings

MODEL_ML_PATH = settings.MODEL_ML_PATH
MODEL_CNN_PATH = settings.MODEL_CNN_PATH
MODEL_BIOPSY_PATH = settings.MODEL_BIOPSY_PATH

# Datos de pacientes
CSV_RISK_PATH = settings.CSV_RISK_PATH
CSV_PATIENTS_PATH = settings.CSV_PACIENTES_5000_PATH

# Features que espera el modelo ML (orden exacto)
ML_FEATURE_NAMES = settings.ML_FEATURE_NAMES

# Mapeo de clases para el modelo ML
RISK_LEVEL_MAP = settings.RISK_LEVEL_MAP


###############################################################################
# Arquitectura del modelo de biopsias (PyTorch)
###############################################################################


class BiopsyClassifier(nn.Module):
    """
    Clasificador binario ResNet18 para biopsias de colon.
    Arquitectura idéntica a la usada en el entrenamiento
    (src/networks/dl_biopsia/modelo_biopsia_v0.py).
    """

    def __init__(self):
        super(BiopsyClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)


###############################################################################
# Funciones de carga individual
###############################################################################


def load_ml_model(path: str = MODEL_ML_PATH):
    """Carga el modelo LightGBM clínico desde disco."""
    if not os.path.exists(path):
        print(f"[AVISO] Modelo ML no encontrado en: {path}")
        return None

    try:
        modelo = joblib.load(path)
        print(f"[OK] Modelo ML cargado: {type(modelo).__name__} desde {path}")
        return modelo
    except Exception as e:
        print(f"[ERROR] Fallo al cargar modelo ML: {e}")
        return None


def load_cnn_model(path: str = MODEL_CNN_PATH):
    """
    Carga el modelo CNN de colonoscopia (TensorFlow/Keras).
    Incluye parche de compatibilidad para Keras 3.
    """
    if not os.path.exists(path):
        print(f"[AVISO] Modelo CNN no encontrado en: {path}")
        return None

    try:
        import tensorflow as tf

        # Parche de compatibilidad para Keras 3 (quantization_config)
        original_dense_init = tf.keras.layers.Dense.__init__

        def patched_dense_init(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            return original_dense_init(self, *args, **kwargs)

        tf.keras.layers.Dense.__init__ = patched_dense_init

        modelo = tf.keras.models.load_model(path, compile=False)
        print(f"[OK] Modelo CNN cargado desde {path}")
        return modelo
    except Exception as e:
        print(f"[ERROR] Fallo al cargar modelo CNN: {e}")
        return None


def load_biopsy_model(path: str = MODEL_BIOPSY_PATH):
    """Carga los pesos del clasificador ResNet18 de biopsias (PyTorch)."""
    if not os.path.exists(path):
        print(f"[AVISO] Modelo de biopsias no encontrado en: {path}")
        return None

    try:
        model = BiopsyClassifier()
        state_dict = torch.load(
            path, map_location=torch.device("cpu"), weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        print(f"[OK] Modelo de biopsias cargado desde {path}")
        return model
    except Exception as e:
        print(f"[ERROR] Fallo al cargar modelo de biopsias: {e}")
        return None


###############################################################################
# LIFESPAN — Carga y descarga de modelos al arrancar/parar la API
###############################################################################


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Contexto de vida de la aplicación FastAPI.
    Carga los modelos al arrancar y los libera al parar.
    """
    print("=" * 60)
    print("  ColonAI API — Inicializando modelos...")
    print("=" * 60)

    app.state.modelo_ml = load_ml_model()
    app.state.modelo_cnn = load_cnn_model()
    app.state.modelo_biopsia = load_biopsy_model()

    modelos_ok = sum(
        [
            app.state.modelo_ml is not None,
            app.state.modelo_cnn is not None,
            app.state.modelo_biopsia is not None,
        ]
    )
    print(f"\n[RESUMEN] {modelos_ok}/3 modelos cargados correctamente.")
    print("=" * 60)

    yield  # La API se ejecuta aquí

    # Cleanup al parar
    print("[INFO] Liberando recursos de modelos...")
    app.state.modelo_ml = None
    app.state.modelo_cnn = None
    app.state.modelo_biopsia = None
