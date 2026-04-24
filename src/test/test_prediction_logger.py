"""
=============================================================================
Tests unitarios para prediction_logger.py
=============================================================================
Verifican que el logger de predicciones escribe correctamente al CSV
y que las funciones de consulta (get_history, get_stats) funcionan.

Ejecutar con:
    pytest src/test/test_prediction_logger.py -v
=============================================================================
"""

import csv
import os
import pytest

from src.tracking.prediction_logger import PredictionLogger, CSV_HEADERS


###############################################################################
# Fixture: Logger temporal que escribe en un CSV de prueba
###############################################################################


@pytest.fixture
def logger_temporal(tmp_path):
    """Crea un PredictionLogger que escribe en un directorio temporal."""
    ruta_csv = str(tmp_path / "predictions_test.csv")
    return PredictionLogger(log_path=ruta_csv)


###############################################################################
# Tests de creación e inicialización
###############################################################################


class TestInicializacion:
    """Comprueba que el logger crea el CSV con las cabeceras correctas."""

    def test_crea_archivo_csv_al_inicializar(self, logger_temporal):
        """El archivo CSV debe existir tras crear el logger."""
        assert os.path.exists(logger_temporal.log_path)

    def test_archivo_csv_tiene_cabeceras_correctas(self, logger_temporal):
        """La primera fila del CSV debe contener las cabeceras esperadas."""
        with open(logger_temporal.log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            cabeceras = next(reader)
        assert cabeceras == CSV_HEADERS

    def test_no_sobreescribe_archivo_existente(self, logger_temporal):
        """Si el CSV ya existe con datos, no debe sobreescribirlo."""
        logger_temporal.log_risk_prediction(
            patient_id=1, risk_level="Low", risk_score=0.1
        )
        # Crear otro logger apuntando al mismo archivo
        logger_2 = PredictionLogger(log_path=logger_temporal.log_path)
        historial = logger_2.get_history()
        assert len(historial) == 1  # El registro previo sigue ahí


###############################################################################
# Tests de log de predicción de riesgo
###############################################################################


class TestLogRiesgo:
    """Comprueba el registro de predicciones ML de riesgo."""

    def test_registra_prediccion_de_riesgo_basica(self, logger_temporal):
        """Debe añadir una fila con los datos correctos al CSV."""
        logger_temporal.log_risk_prediction(
            patient_id=42,
            risk_level="High",
            risk_score=0.85,
            confidence=0.92,
            features={"Smoking": 8, "BMI": 32.5},
        )

        historial = logger_temporal.get_history()
        assert len(historial) == 1
        registro = historial[0]
        assert registro["prediction_type"] == "risk_ml"
        assert registro["patient_id"] == "42"
        assert registro["risk_level"] == "High"

    def test_registra_multiples_predicciones(self, logger_temporal):
        """Múltiples predicciones deben añadirse secuencialmente."""
        for i in range(5):
            logger_temporal.log_risk_prediction(
                patient_id=i, risk_level="Low", risk_score=0.1
            )

        historial = logger_temporal.get_history()
        assert len(historial) == 5

    def test_prediccion_sin_patient_id(self, logger_temporal):
        """El patient_id debe ser vacío si no se proporciona."""
        logger_temporal.log_risk_prediction(risk_level="Medium", risk_score=0.5)
        historial = logger_temporal.get_history()
        assert historial[0]["patient_id"] == ""

    def test_redondeo_de_confianza(self, logger_temporal):
        """La confianza debe redondearse a 4 decimales."""
        logger_temporal.log_risk_prediction(
            risk_level="High", risk_score=0.123456789, confidence=0.987654321
        )
        historial = logger_temporal.get_history()
        assert historial[0]["confidence"] == "0.9877"


###############################################################################
# Tests de log de predicción de imagen
###############################################################################


class TestLogImagen:
    """Comprueba el registro de predicciones de imagen (colonoscopia/biopsia)."""

    def test_registra_prediccion_de_colonoscopia(self, logger_temporal):
        """Debe registrar una predicción de colonoscopia correctamente."""
        logger_temporal.log_image_prediction(
            analysis_type="colonoscopy",
            diagnosis="POLIPO DETECTADO",
            confidence=0.92,
            patient_id=10,
            image_name="colon_001.jpg",
        )

        historial = logger_temporal.get_history()
        assert len(historial) == 1
        assert historial[0]["prediction_type"] == "image_colonoscopy"
        assert historial[0]["diagnosis"] == "POLIPO DETECTADO"
        assert "colon_001.jpg" in historial[0]["details"]

    def test_registra_prediccion_de_biopsia(self, logger_temporal):
        """Debe registrar una predicción de biopsia correctamente."""
        logger_temporal.log_image_prediction(
            analysis_type="biopsy",
            diagnosis="BENIGNO",
            confidence=0.78,
        )

        historial = logger_temporal.get_history()
        assert historial[0]["prediction_type"] == "image_biopsy"

    def test_imagen_sin_nombre_usa_upload(self, logger_temporal):
        """Si no se da nombre de imagen, debe usar 'upload'."""
        logger_temporal.log_image_prediction(
            analysis_type="biopsy", diagnosis="MALIGNO", confidence=0.65
        )
        historial = logger_temporal.get_history()
        assert "upload" in historial[0]["details"]


###############################################################################
# Tests de consulta (get_history, get_stats)
###############################################################################


class TestConsultas:
    """Comprueba las funciones de consulta del historial."""

    def test_get_history_con_limite(self, logger_temporal):
        """get_history con limit debe devolver solo las últimas N."""
        for i in range(10):
            logger_temporal.log_risk_prediction(
                patient_id=i, risk_level="Low", risk_score=0.1
            )

        ultimas_3 = logger_temporal.get_history(limit=3)
        assert len(ultimas_3) == 3
        # Las últimas deben ser patient_id 7, 8, 9
        assert ultimas_3[0]["patient_id"] == "7"

    def test_get_history_archivo_vacio(self, tmp_path):
        """get_history sobre archivo inexistente devuelve lista vacía."""
        logger = PredictionLogger(log_path=str(tmp_path / "no_existe.csv"))
        # Borramos el archivo creado automáticamente para simular inexistencia
        os.remove(logger.log_path)
        assert logger.get_history() == []

    def test_get_stats_basicas(self, logger_temporal):
        """get_stats debe devolver conteo correcto por tipo y diagnóstico."""
        logger_temporal.log_risk_prediction(risk_level="High", risk_score=0.9)
        logger_temporal.log_risk_prediction(risk_level="Low", risk_score=0.1)
        logger_temporal.log_image_prediction(
            analysis_type="biopsy", diagnosis="BENIGNO", confidence=0.8
        )

        stats = logger_temporal.get_stats()
        assert stats["total"] == 3
        assert stats["by_type"]["risk_ml"] == 2
        assert stats["by_type"]["image_biopsy"] == 1

    def test_get_stats_sin_datos(self, tmp_path):
        """get_stats sin predicciones devuelve total=0."""
        logger = PredictionLogger(log_path=str(tmp_path / "vacio.csv"))
        stats = logger.get_stats()
        assert stats["total"] == 0
