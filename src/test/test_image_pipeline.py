"""
=============================================================================
Tests unitarios para image_pipeline.py
=============================================================================
Verifican el pipeline de análisis de imagen médica usando mocks para los
modelos de IA. No requieren los archivos .pth reales.

Ejecutar con:
    pytest src/test/test_image_pipeline.py -v
=============================================================================
"""

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.pipelines.image_pipeline import ImageAnalysisPipeline, ImageAnalysisResult


###############################################################################
# Fixture: Pipeline con modelos mockeados
###############################################################################


@pytest.fixture
def pipeline():
    """Crea un ImageAnalysisPipeline limpio."""
    return ImageAnalysisPipeline()


###############################################################################
# Tests del dataclass ImageAnalysisResult
###############################################################################


class TestImageAnalysisResult:
    """Comprueba la estructura del resultado de análisis."""

    def test_resultado_tiene_campos_obligatorios(self):
        """El resultado debe tener todos los campos definidos."""
        resultado = ImageAnalysisResult(
            diagnosis="TEST",
            is_positive=True,
            confidence=0.95,
            raw_prediction=0.95,
            recommendation="Recomendación de prueba",
        )
        assert resultado.diagnosis == "TEST"
        assert resultado.is_positive is True
        assert resultado.confidence == 0.95
        assert resultado.heatmap is None  # Opcional, None por defecto

    def test_resultado_con_heatmap(self):
        """El resultado puede incluir un mapa de calor."""
        heatmap_fake = np.zeros((224, 224, 3), dtype=np.uint8)
        resultado = ImageAnalysisResult(
            diagnosis="TEST",
            is_positive=False,
            confidence=0.8,
            raw_prediction=0.8,
            recommendation="OK",
            heatmap=heatmap_fake,
        )
        assert resultado.heatmap is not None
        assert resultado.heatmap.shape == (224, 224, 3)


###############################################################################
# Tests de análisis de biopsia
###############################################################################


class TestAnalisisBiopsia:
    """Comprueba el pipeline de análisis de biopsias con mocks."""

    def test_sin_modelo_devuelve_error(self, pipeline):
        """Si el modelo no se carga, debe devolver diagnosis=ERROR."""
        pipeline._modelo_biopsia = None
        with patch.object(pipeline, "_get_biopsy_model", return_value=None):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            resultado = pipeline.analyze_biopsy(img, generate_heatmap=False)
            assert resultado.diagnosis == "ERROR"
            assert resultado.confidence == 0.0

    def test_con_modelo_mock_devuelve_diagnostico(self, pipeline):
        """Con un modelo mock, debe devolver un diagnóstico válido."""
        # Crear modelo mock que devuelve un tensor con valor > 0.5 (benigno)
        modelo_mock = MagicMock()
        modelo_mock.return_value = torch.tensor([[0.8]])

        pipeline._modelo_biopsia = modelo_mock
        with patch.object(pipeline, "_get_biopsy_model", return_value=modelo_mock):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            resultado = pipeline.analyze_biopsy(img, generate_heatmap=False)

            assert "BENIGNO" in resultado.diagnosis
            assert resultado.is_positive is False  # Benigno = no positivo
            assert resultado.confidence > 0.5

    def test_clasificacion_maligna(self, pipeline):
        """Un valor sigmoid < 0.5 del modelo debe clasificar como maligno."""
        modelo_mock = MagicMock()
        # torch.sigmoid(-2.0) ≈ 0.12 → maligno (< 0.5)
        modelo_mock.return_value = torch.tensor([[-2.0]])

        pipeline._modelo_biopsia = modelo_mock
        with patch.object(pipeline, "_get_biopsy_model", return_value=modelo_mock):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            resultado = pipeline.analyze_biopsy(img, generate_heatmap=False)

            assert "MALIGNO" in resultado.diagnosis
            assert resultado.is_positive is True  # Maligno = positivo


###############################################################################
# Tests de análisis de colonoscopia
###############################################################################


class TestAnalisisColonoscopia:
    """Comprueba el pipeline de análisis de colonoscopia con mocks."""

    def test_sin_modelo_devuelve_error(self, pipeline):
        """Si el modelo CNN no se carga, devuelve ERROR."""
        pipeline._modelo_cnn = None
        with patch.object(pipeline, "_get_cnn_model", return_value=None):
            img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
            resultado = pipeline.analyze_colonoscopy(img, generate_heatmap=False)
            assert resultado.diagnosis == "ERROR"

    def test_con_modelo_mock_detecta_polipo(self, pipeline):
        """Un valor < 0.5 del modelo debe clasificar como pólipo."""
        modelo_mock = MagicMock()
        modelo_mock.return_value = torch.tensor([[0.2]])

        pipeline._modelo_cnn = modelo_mock
        with patch.object(pipeline, "_get_cnn_model", return_value=modelo_mock):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            resultado = pipeline.analyze_colonoscopy(img, generate_heatmap=False)

            assert "PÓLIPO" in resultado.diagnosis
            assert resultado.is_positive is True

    def test_con_modelo_mock_tejido_sano(self, pipeline):
        """Un valor >= 0.5 del modelo debe clasificar como sano."""
        modelo_mock = MagicMock()
        modelo_mock.return_value = torch.tensor([[0.85]])

        pipeline._modelo_cnn = modelo_mock
        with patch.object(pipeline, "_get_cnn_model", return_value=modelo_mock):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            resultado = pipeline.analyze_colonoscopy(img, generate_heatmap=False)

            assert "SANO" in resultado.diagnosis
            assert resultado.is_positive is False


###############################################################################
# Tests de inicialización del pipeline
###############################################################################


class TestInicializacionPipeline:
    """Comprueba el estado inicial del pipeline."""

    def test_modelos_none_al_crear(self, pipeline):
        """Al crear un pipeline, no debe tener modelos cargados."""
        assert pipeline._modelo_cnn is None
        assert pipeline._modelo_biopsia is None
