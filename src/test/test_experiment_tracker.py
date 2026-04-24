"""
=============================================================================
Tests unitarios para experiment_tracker.py
=============================================================================
Verifican que el tracker de experimentos registra, consulta y resume
correctamente los experimentos de entrenamiento en JSON.

Ejecutar con:
    pytest src/test/test_experiment_tracker.py -v
=============================================================================
"""

import json
import os
import pytest

from src.tracking.experiment_tracker import ExperimentTracker


###############################################################################
# Fixture: Tracker temporal
###############################################################################


@pytest.fixture
def tracker_temporal(tmp_path):
    """Crea un ExperimentTracker que escribe en un directorio temporal."""
    ruta_json = str(tmp_path / "experiments_test.json")
    return ExperimentTracker(tracker_path=ruta_json)


###############################################################################
# Tests de inicialización
###############################################################################


class TestInicializacion:
    """Comprueba que el tracker crea el archivo JSON correctamente."""

    def test_crea_archivo_json_al_inicializar(self, tracker_temporal):
        """El archivo JSON debe existir tras crear el tracker."""
        assert os.path.exists(tracker_temporal.tracker_path)

    def test_archivo_json_empieza_vacio(self, tracker_temporal):
        """El JSON inicial debe ser un diccionario vacío."""
        with open(tracker_temporal.tracker_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
        assert datos == {}


###############################################################################
# Tests de registro de experimentos
###############################################################################


class TestRegistro:
    """Comprueba que log_experiment registra los datos correctamente."""

    def test_registra_un_experimento_basico(self, tracker_temporal):
        """Debe registrar un experimento y devolver un ID."""
        exp_id = tracker_temporal.log_experiment(
            model_name="LightGBM",
            metrics={"Accuracy": 0.85, "Recall": 0.91},
        )

        assert isinstance(exp_id, str)
        assert len(exp_id) == 8  # UUID truncado a 8 chars

    def test_experimento_contiene_campos_completos(self, tracker_temporal):
        """El registro debe contener todos los campos especificados."""
        exp_id = tracker_temporal.log_experiment(
            model_name="XGBoost",
            metrics={"Accuracy": 0.88},
            hyperparameters={"n_estimators": 300, "learning_rate": 0.02},
            features=["Smoking", "BMI"],
            dataset_path="datos_test.csv",
            model_path="modelo_test.pkl",
            train_size=4000,
            test_size=1000,
            notes="Test unitario",
        )

        exp = tracker_temporal.get_experiment(exp_id)
        assert exp["model_name"] == "XGBoost"
        assert exp["metrics"]["Accuracy"] == 0.88
        assert exp["hyperparameters"]["n_estimators"] == 300
        assert exp["features"] == ["Smoking", "BMI"]
        assert exp["dataset_path"] == "datos_test.csv"
        assert exp["train_size"] == 4000
        assert exp["notes"] == "Test unitario"
        assert "timestamp" in exp

    def test_registra_multiples_experimentos(self, tracker_temporal):
        """Múltiples experimentos deben acumularse en el JSON."""
        for i in range(5):
            tracker_temporal.log_experiment(
                model_name=f"Modelo_{i}",
                metrics={"Accuracy": 0.5 + i * 0.1},
            )

        todos = tracker_temporal.get_all_experiments()
        assert len(todos) == 5

    def test_metricas_se_redondean_a_6_decimales(self, tracker_temporal):
        """Las métricas float deben redondearse a 6 decimales."""
        exp_id = tracker_temporal.log_experiment(
            model_name="Test",
            metrics={"Accuracy": 0.123456789012345},
        )
        exp = tracker_temporal.get_experiment(exp_id)
        assert exp["metrics"]["Accuracy"] == 0.123457

    def test_metricas_dict_se_preservan(self, tracker_temporal):
        """Las métricas tipo dict (confusion matrix) deben preservarse."""
        exp_id = tracker_temporal.log_experiment(
            model_name="Test",
            metrics={
                "Accuracy": 0.85,
                "Confusion Matrix": {"matrix": [[10, 2], [3, 15]]},
            },
        )
        exp = tracker_temporal.get_experiment(exp_id)
        assert "Confusion Matrix" in exp["metrics"]
        assert exp["metrics"]["Confusion Matrix"]["matrix"] == [[10, 2], [3, 15]]


###############################################################################
# Tests de consulta
###############################################################################


class TestConsultas:
    """Comprueba las funciones de búsqueda y resumen."""

    def test_get_experiment_inexistente_devuelve_none(self, tracker_temporal):
        """Consultar un ID que no existe debe devolver None."""
        resultado = tracker_temporal.get_experiment("id_falso")
        assert resultado is None

    def test_get_all_experiments_vacio(self, tracker_temporal):
        """Sin experimentos, debe devolver dict vacío."""
        todos = tracker_temporal.get_all_experiments()
        assert todos == {}

    def test_get_best_experiment_por_accuracy(self, tracker_temporal):
        """Debe devolver el experimento con mayor Accuracy."""
        tracker_temporal.log_experiment("Malo", metrics={"Accuracy": 0.50})
        tracker_temporal.log_experiment("Bueno", metrics={"Accuracy": 0.95})
        tracker_temporal.log_experiment("Medio", metrics={"Accuracy": 0.75})

        resultado = tracker_temporal.get_best_experiment("Accuracy")
        assert resultado is not None
        exp_id, exp_data = resultado
        assert exp_data["model_name"] == "Bueno"
        assert exp_data["metrics"]["Accuracy"] == 0.95

    def test_get_best_experiment_sin_datos_devuelve_none(self, tracker_temporal):
        """Sin experimentos, get_best_experiment devuelve None."""
        resultado = tracker_temporal.get_best_experiment("Accuracy")
        assert resultado is None

    def test_get_best_experiment_por_recall(self, tracker_temporal):
        """Debe funcionar con cualquier métrica, no solo Accuracy."""
        tracker_temporal.log_experiment("A", metrics={"Recall": 0.70})
        tracker_temporal.log_experiment("B", metrics={"Recall": 0.99})

        resultado = tracker_temporal.get_best_experiment("Recall")
        _, exp = resultado
        assert exp["model_name"] == "B"


###############################################################################
# Tests de resumen textual
###############################################################################


class TestResumen:
    """Comprueba la función summary()."""

    def test_summary_sin_datos(self, tracker_temporal):
        """Sin experimentos, summary devuelve mensaje apropiado."""
        texto = tracker_temporal.summary()
        assert "No hay experimentos" in texto

    def test_summary_con_datos(self, tracker_temporal):
        """Con experimentos, summary incluye nombres y métricas."""
        tracker_temporal.log_experiment("LightGBM", metrics={"Accuracy": 0.85})
        tracker_temporal.log_experiment("XGBoost", metrics={"Accuracy": 0.88})

        texto = tracker_temporal.summary()
        assert "LightGBM" in texto
        assert "XGBoost" in texto
        assert "2 registros" in texto
