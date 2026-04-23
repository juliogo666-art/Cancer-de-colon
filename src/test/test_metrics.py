"""
=============================================================================
Tests reales para las métricas clínicas
=============================================================================
Estos tests comprueban que cada métrica de src/metrics/ calcula
correctamente los valores que debería para datos conocidos.

Usamos datos inventados pero con resultados que podemos calcular a mano
para verificar que las métricas funcionan bien.

Ejecutar con:
    pytest src/test/test_metrics.py -v
=============================================================================
"""

import numpy as np
import pytest

from src.metrics import (
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    FBetaMetric,
    ConfusionMatrixMetric,
    ROCAUCMetric,
)


###############################################################################
# Datos de prueba
###############################################################################

# Inventamos predicciones y etiquetas reales que podemos verificar a mano.
# Imagina 10 pacientes con su nivel de riesgo real y lo que el modelo predijo:

# Etiquetas reales (lo que de verdad tiene cada paciente)
# 0 = Low, 1 = Medium, 2 = High
ETIQUETAS_REALES = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])

# Lo que el modelo predijo para cada paciente
PREDICCIONES_DEL_MODELO = np.array([0, 0, 1, 0, 2, 1, 0, 1, 2, 1])
# Aciertos: pacientes 0,1,2,4,6,7,8 → 7 de 10 → Accuracy = 0.70

# Probabilidades que el modelo asignó a cada clase (para ROC-AUC)
# Cada fila es un paciente, cada columna es [Low, Medium, High]
PROBABILIDADES_DEL_MODELO = np.array(
    [
        [0.9, 0.08, 0.02],  # Paciente 0: claramente Low
        [0.85, 0.1, 0.05],  # Paciente 1: claramente Low
        [0.1, 0.8, 0.1],  # Paciente 2: claramente Medium
        [0.5, 0.3, 0.2],  # Paciente 3: el modelo dudó (real: Medium, pred: Low)
        [0.05, 0.1, 0.85],  # Paciente 4: claramente High
        [0.1, 0.6, 0.3],  # Paciente 5: dudó (real: High, pred: Medium)
        [0.8, 0.15, 0.05],  # Paciente 6: claramente Low
        [0.15, 0.7, 0.15],  # Paciente 7: Medium
        [0.05, 0.15, 0.8],  # Paciente 8: High
        [0.3, 0.5, 0.2],  # Paciente 9: dudó (real: Low, pred: Medium)
    ]
)


###############################################################################
# Test de cada métrica
###############################################################################


class TestAccuracyMetric:
    """Comprueba que Accuracy funciona correctamente."""

    def test_nombre_de_la_metrica(self):
        """El nombre debe ser 'Accuracy'."""
        metrica = AccuracyMetric()
        assert metrica.name == "Accuracy"

    def test_calculo_con_datos_conocidos(self):
        """Con 7 aciertos de 10, Accuracy debe ser 0.7."""
        metrica = AccuracyMetric()
        resultado = metrica.compute(ETIQUETAS_REALES, PREDICCIONES_DEL_MODELO)
        assert resultado == pytest.approx(0.7, abs=0.01)

    def test_prediccion_perfecta(self):
        """Si predices todo bien, Accuracy = 1.0."""
        metrica = AccuracyMetric()
        datos = np.array([0, 1, 2])
        resultado = metrica.compute(datos, datos)  # Predicción = Realidad
        assert resultado == 1.0

    def test_prediccion_todo_mal(self):
        """Si predices todo al revés, Accuracy = 0.0."""
        metrica = AccuracyMetric()
        reales = np.array([0, 0, 0])
        predichos = np.array([1, 1, 1])  # Todo mal
        resultado = metrica.compute(reales, predichos)
        assert resultado == 0.0


class TestPrecisionMetric:
    """Comprueba que Precision funciona correctamente."""

    def test_nombre_de_la_metrica(self):
        metrica = PrecisionMetric(average="macro")
        assert metrica.name == "Precision (macro)"

    def test_precision_es_un_float(self):
        """El resultado debe ser un número flotante entre 0 y 1."""
        metrica = PrecisionMetric()
        resultado = metrica.compute(ETIQUETAS_REALES, PREDICCIONES_DEL_MODELO)
        assert isinstance(resultado, float)
        assert 0.0 <= resultado <= 1.0


class TestRecallMetric:
    """Comprueba que Recall funciona correctamente."""

    def test_nombre_de_la_metrica(self):
        metrica = RecallMetric(average="macro")
        assert metrica.name == "Recall (macro)"

    def test_recall_es_un_float(self):
        """El resultado debe ser un número flotante entre 0 y 1."""
        metrica = RecallMetric()
        resultado = metrica.compute(ETIQUETAS_REALES, PREDICCIONES_DEL_MODELO)
        assert isinstance(resultado, float)
        assert 0.0 <= resultado <= 1.0

    def test_recall_perfecto(self):
        """Si detectas todos los enfermos, Recall = 1.0."""
        metrica = RecallMetric(average="macro")
        datos = np.array([0, 1, 2])
        resultado = metrica.compute(datos, datos)
        assert resultado == 1.0


class TestFBetaMetric:
    """Comprueba que F-Beta Score funciona correctamente."""

    def test_nombre_f1(self):
        metrica = FBetaMetric(beta=1.0, average="macro")
        assert metrica.name == "F1-Score (macro)"

    def test_nombre_f2(self):
        metrica = FBetaMetric(beta=2.0, average="macro")
        assert metrica.name == "F2-Score (macro)"

    def test_f2_mayor_o_igual_que_cero(self):
        """F2 Score debe ser un número válido."""
        metrica = FBetaMetric(beta=2.0)
        resultado = metrica.compute(ETIQUETAS_REALES, PREDICCIONES_DEL_MODELO)
        assert isinstance(resultado, float)
        assert 0.0 <= resultado <= 1.0


class TestConfusionMatrixMetric:
    """Comprueba que la Confusion Matrix funciona correctamente."""

    def test_nombre_de_la_metrica(self):
        metrica = ConfusionMatrixMetric()
        assert metrica.name == "Confusion Matrix"

    def test_devuelve_diccionario(self):
        """La confusion matrix debe devolver un dict con 'matrix' y 'labels'."""
        metrica = ConfusionMatrixMetric()
        resultado = metrica.compute(ETIQUETAS_REALES, PREDICCIONES_DEL_MODELO)
        assert isinstance(resultado, dict)
        assert "matrix" in resultado
        assert "labels" in resultado
        assert "classification_report" in resultado

    def test_la_matriz_tiene_3x3_clases(self):
        """Para 3 clases (Low/Medium/High) la matriz debe ser 3x3."""
        metrica = ConfusionMatrixMetric()
        resultado = metrica.compute(ETIQUETAS_REALES, PREDICCIONES_DEL_MODELO)
        matriz = resultado["matrix"]
        assert len(matriz) == 3  # 3 filas
        assert len(matriz[0]) == 3  # 3 columnas


class TestROCAUCMetric:
    """Comprueba que ROC-AUC funciona correctamente."""

    def test_nombre_de_la_metrica(self):
        metrica = ROCAUCMetric()
        assert metrica.name == "ROC-AUC (macro)"

    def test_sin_probabilidades_devuelve_cero(self):
        """Si no pasas probabilidades, ROC-AUC no se puede calcular → devuelve 0."""
        metrica = ROCAUCMetric()
        resultado = metrica.compute(
            ETIQUETAS_REALES, PREDICCIONES_DEL_MODELO, y_proba=None
        )
        assert resultado == 0.0

    def test_con_probabilidades_devuelve_float(self):
        """Con probabilidades válidas, debe devolver un float entre 0 y 1."""
        metrica = ROCAUCMetric()
        resultado = metrica.compute(
            ETIQUETAS_REALES,
            PREDICCIONES_DEL_MODELO,
            y_proba=PROBABILIDADES_DEL_MODELO,
        )
        assert isinstance(resultado, float)
        assert 0.0 <= resultado <= 1.0
