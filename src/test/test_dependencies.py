"""
=============================================================================
Tests unitarios para dependencies.py
=============================================================================
Verifican la carga de modelos usando mocks (no necesitan los archivos .pkl/.pth
reales). Comprueban que cada función de carga maneja correctamente los casos
de archivo encontrado, archivo no encontrado y errores de carga.

Ejecutar con:
    pytest src/test/test_dependencies.py -v
=============================================================================
"""

import pytest
from unittest.mock import patch, MagicMock


###############################################################################
# Tests de carga de modelo ML (LightGBM)
###############################################################################


class TestCargaModeloML:
    """Comprueba la función load_ml_model()."""

    def test_devuelve_none_si_archivo_no_existe(self, tmp_path):
        """Si el archivo .pkl no existe, debe devolver None."""
        from src.api.dependencies import load_ml_model

        resultado = load_ml_model(path=str(tmp_path / "no_existe.pkl"))
        assert resultado is None

    @patch("src.api.dependencies.joblib.load")
    @patch("src.api.dependencies.os.path.exists", return_value=True)
    def test_carga_modelo_exitosamente(self, mock_exists, mock_joblib):
        """Si el archivo existe y joblib carga bien, devuelve el modelo."""
        from src.api.dependencies import load_ml_model

        modelo_fake = MagicMock()
        mock_joblib.return_value = modelo_fake

        resultado = load_ml_model(path="modelo_fake.pkl")
        assert resultado is modelo_fake
        mock_joblib.assert_called_once_with("modelo_fake.pkl")

    @patch("src.api.dependencies.joblib.load", side_effect=Exception("Error de prueba"))
    @patch("src.api.dependencies.os.path.exists", return_value=True)
    def test_devuelve_none_si_carga_falla(self, mock_exists, mock_joblib):
        """Si joblib.load lanza una excepción, devuelve None."""
        from src.api.dependencies import load_ml_model

        resultado = load_ml_model(path="modelo_corrupto.pkl")
        assert resultado is None


###############################################################################
# Tests de carga de modelo ML final (Ensemble)
###############################################################################


class TestCargaModeloMLFinal:
    """Comprueba la función load_ml_final_model()."""

    def test_devuelve_none_si_archivo_no_existe(self, tmp_path):
        """Si el archivo no existe, devuelve None."""
        from src.api.dependencies import load_ml_final_model

        resultado = load_ml_final_model(path=str(tmp_path / "no_existe.pkl"))
        assert resultado is None

    @patch("src.api.dependencies.joblib.load")
    @patch("src.api.dependencies.os.path.exists", return_value=True)
    def test_carga_modelo_exitosamente(self, mock_exists, mock_joblib):
        """Si existe, carga y devuelve el modelo."""
        from src.api.dependencies import load_ml_final_model

        modelo_fake = MagicMock()
        mock_joblib.return_value = modelo_fake

        resultado = load_ml_final_model(path="ensemble.pkl")
        assert resultado is modelo_fake

    @patch("src.api.dependencies.joblib.load", side_effect=Exception("Fallo"))
    @patch("src.api.dependencies.os.path.exists", return_value=True)
    def test_devuelve_none_si_carga_falla(self, mock_exists, mock_joblib):
        """Error en carga devuelve None."""
        from src.api.dependencies import load_ml_final_model

        assert load_ml_final_model(path="corrupto.pkl") is None


###############################################################################
# Tests de carga de modelo Triaje
###############################################################################


class TestCargaModeloTriaje:
    """Comprueba la función load_triage_model()."""

    def test_devuelve_none_si_archivo_no_existe(self, tmp_path):
        """Si el archivo no existe, devuelve None."""
        from src.api.dependencies import load_triage_model

        resultado = load_triage_model(path=str(tmp_path / "no_existe.pkl"))
        assert resultado is None

    @patch("src.api.dependencies.joblib.load")
    @patch("src.api.dependencies.os.path.exists", return_value=True)
    def test_carga_modelo_exitosamente(self, mock_exists, mock_joblib):
        """Si existe, carga y devuelve el modelo."""
        from src.api.dependencies import load_triage_model

        modelo_fake = MagicMock()
        mock_joblib.return_value = modelo_fake

        resultado = load_triage_model(path="triage.pkl")
        assert resultado is modelo_fake

    @patch("src.api.dependencies.joblib.load", side_effect=Exception("Fallo"))
    @patch("src.api.dependencies.os.path.exists", return_value=True)
    def test_devuelve_none_si_carga_falla(self, mock_exists, mock_joblib):
        """Error en carga devuelve None."""
        from src.api.dependencies import load_triage_model

        assert load_triage_model(path="corrupto.pkl") is None


###############################################################################
# Tests de carga de modelo CNN (Colonoscopia — PyTorch)
###############################################################################


class TestCargaModeloCNN:
    """Comprueba la función load_cnn_model()."""

    def test_devuelve_none_si_archivo_no_existe(self, tmp_path):
        """Si el .pth no existe, devuelve None."""
        from src.api.dependencies import load_cnn_model

        resultado = load_cnn_model(path=str(tmp_path / "no_existe.pth"))
        assert resultado is None

    @patch("src.api.dependencies.torch.load")
    @patch("src.api.dependencies.os.path.exists", return_value=True)
    def test_devuelve_none_si_carga_falla(self, mock_exists, mock_torch_load):
        """Si torch.load falla, devuelve None."""
        from src.api.dependencies import load_cnn_model

        mock_torch_load.side_effect = Exception("Pesos incompatibles")
        resultado = load_cnn_model(path="modelo_roto.pth")
        assert resultado is None


###############################################################################
# Tests de carga de modelo de Biopsias (DenseNet121 — PyTorch)
###############################################################################


class TestCargaModeloBiopsia:
    """Comprueba la función load_biopsy_model()."""

    def test_devuelve_none_si_archivo_no_existe(self, tmp_path):
        """Si el .pth no existe, devuelve None."""
        from src.api.dependencies import load_biopsy_model

        resultado = load_biopsy_model(path=str(tmp_path / "no_existe.pth"))
        assert resultado is None

    @patch("src.api.dependencies.torch.load")
    @patch("src.api.dependencies.os.path.exists", return_value=True)
    def test_devuelve_none_si_carga_falla(self, mock_exists, mock_torch_load):
        """Si torch.load falla, devuelve None."""
        from src.api.dependencies import load_biopsy_model

        mock_torch_load.side_effect = Exception("Pesos incompatibles")
        resultado = load_biopsy_model(path="modelo_roto.pth")
        assert resultado is None


###############################################################################
# Tests de las clases clasificadoras (arquitectura)
###############################################################################


class TestArquitecturaClasificadores:
    """Comprueba que las clases del modelo tienen la estructura correcta."""

    def test_colonoscopy_classifier_tiene_forward(self):
        """ColonoscopyClassifier debe tener método forward."""
        from src.api.dependencies import ColonoscopyClassifier

        modelo = ColonoscopyClassifier()
        assert hasattr(modelo, "forward")
        assert hasattr(modelo, "features")
        assert hasattr(modelo, "classifier")

    def test_biopsy_classifier_tiene_forward(self):
        """BiopsyClassifier debe tener método forward."""
        from src.api.dependencies import BiopsyClassifier

        modelo = BiopsyClassifier()
        assert hasattr(modelo, "forward")
        assert hasattr(modelo, "model")

    def test_colonoscopy_classifier_acepta_tensor(self):
        """El modelo debe poder procesar un tensor de entrada sin error."""
        import torch
        from src.api.dependencies import ColonoscopyClassifier

        modelo = ColonoscopyClassifier()
        modelo.eval()
        tensor_dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            salida = modelo(tensor_dummy)
        # Salida debe ser un tensor con 1 valor (binario)
        assert salida.shape == (1, 1)

    def test_biopsy_classifier_acepta_tensor(self):
        """El modelo de biopsia debe procesar un tensor sin error."""
        import torch
        from src.api.dependencies import BiopsyClassifier

        modelo = BiopsyClassifier()
        modelo.eval()
        tensor_dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            salida = modelo(tensor_dummy)
        assert salida.shape == (1, 1)
