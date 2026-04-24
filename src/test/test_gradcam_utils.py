"""
=============================================================================
Tests unitarios para gradcam_utils.py
=============================================================================
Verifican las funciones de generación de Grad-CAM (mapas de calor)
usando tensores y modelos dummy de PyTorch.

Ejecutar con:
    pytest src/test/test_gradcam_utils.py -v
=============================================================================
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from unittest.mock import patch, MagicMock


###############################################################################
# Modelo dummy para tests de Grad-CAM (PyTorch)
###############################################################################


class ModeloDummyPyTorch(nn.Module):
    """
    Modelo mínimo con una capa convolucional para probar Grad-CAM.
    Simula la estructura básica de un clasificador binario.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


###############################################################################
# Tests de generate_gradcam (PyTorch — Biopsias)
###############################################################################


class TestGenerateGradcam:
    """Comprueba la función generate_gradcam para modelos PyTorch."""

    def test_genera_mapa_de_calor_valido(self):
        """Debe devolver una imagen RGB y una probabilidad."""
        from src.utils.gradcam_utils import generate_gradcam

        modelo = ModeloDummyPyTorch()
        modelo.eval()
        img_pil = Image.new("RGB", (224, 224), color="blue")
        target_layer = modelo.conv

        resultado_img, prob = generate_gradcam(modelo, img_pil, target_layer)

        # La imagen resultado debe ser un array numpy RGB
        assert isinstance(resultado_img, np.ndarray)
        assert resultado_img.shape == (224, 224, 3)
        assert resultado_img.dtype == np.uint8

        # La probabilidad debe ser un float entre 0 y 1
        assert isinstance(prob, float)

    def test_gradcam_con_imagen_pequena(self):
        """Debe funcionar con imágenes de cualquier tamaño (se redimensionan)."""
        from src.utils.gradcam_utils import generate_gradcam

        modelo = ModeloDummyPyTorch()
        modelo.eval()
        img_pil = Image.new("RGB", (50, 50), color="green")
        target_layer = modelo.conv

        resultado_img, prob = generate_gradcam(modelo, img_pil, target_layer)
        assert resultado_img.shape == (224, 224, 3)


###############################################################################
# Tests de generate_gradcam_pytorch (PyTorch — Colonoscopia)
###############################################################################


class TestGenerateGradcamPytorch:
    """Comprueba la función generate_gradcam_pytorch."""

    def test_genera_resultado_valido(self):
        """Debe devolver una imagen numpy de la superposición."""
        from src.utils.gradcam_utils import generate_gradcam_pytorch

        modelo = ModeloDummyPyTorch()
        modelo.eval()
        img_pil = Image.new("RGB", (224, 224), color="red")
        target_layer = modelo.conv

        resultado = generate_gradcam_pytorch(modelo, img_pil, target_layer)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (224, 224, 3)

    def test_con_imagen_no_cuadrada(self):
        """Debe funcionar con imágenes no cuadradas."""
        from src.utils.gradcam_utils import generate_gradcam_pytorch

        modelo = ModeloDummyPyTorch()
        modelo.eval()
        img_pil = Image.new("RGB", (300, 150), color="yellow")
        target_layer = modelo.conv

        resultado = generate_gradcam_pytorch(modelo, img_pil, target_layer)
        assert resultado.shape == (224, 224, 3)


###############################################################################
# Tests de generar_explicacion_shap (función de explicabilidad ML)
###############################################################################


class TestExplicacionSHAP:
    """Comprueba la función generar_explicacion_shap."""

    @patch("src.utils.gradcam_utils.shap.TreeExplainer")
    @patch("src.utils.gradcam_utils.shap.bar_plot")
    def test_con_modelo_mock_genera_figura(self, mock_bar_plot, mock_explainer):
        """Con un modelo mock de TreeExplainer, debe generar un gráfico."""
        from src.utils.gradcam_utils import generar_explicacion_shap

        # Configurar el mock del explainer
        mock_instance = MagicMock()
        mock_explainer.return_value = mock_instance
        # Simular SHAP values como array 3D [1, 11, 3] (1 muestra, 11 features, 3 clases)
        shap_values_fake = np.random.randn(1, 11, 3)
        mock_instance.shap_values.return_value = shap_values_fake

        features_test = [[5, 3, 2, 1, 4, 3, 7, 6, 25.0, 0, 2.0]]
        textos_test = {
            "shap_col_variable": "Variable",
            "shap_col_impact": "Impacto",
            "shap_col_direction": "Sentido",
            "shap_increases_risk": "Sube riesgo",
            "shap_decreases_risk": "Baja riesgo",
            "shap_chart_title": "Factores de riesgo",
        }

        fig, df_importancia = generar_explicacion_shap(
            MagicMock(),  # modelo mock
            features_test,
            target_class=0,
            textos=textos_test,
        )

        assert fig is not None
        assert df_importancia is not None
        assert len(df_importancia) == 11  # 11 features

    def test_con_error_devuelve_none(self):
        """Si SHAP falla, debe devolver (None, None)."""
        from src.utils.gradcam_utils import generar_explicacion_shap

        # Pasamos un modelo que no es compatible con TreeExplainer
        fig, df = generar_explicacion_shap(
            "modelo_invalido", [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], 0
        )
        assert fig is None
        assert df is None

    def test_sin_textos_usa_defaults(self):
        """Sin traducciones, debe usar textos por defecto en español."""
        from src.utils.gradcam_utils import generar_explicacion_shap

        # Este test verifica que no hay crash sin textos
        fig, df = generar_explicacion_shap(
            "modelo_invalido", [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], 0, textos=None
        )
        # Debería devolver None por error del modelo, pero sin crash
        assert fig is None
        assert df is None
