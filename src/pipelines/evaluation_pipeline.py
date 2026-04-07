"""
=============================================================================
Pipeline de evaluación de modelos clínicos
=============================================================================
Orquesta todas las métricas de src/metrics/ para evaluar modelos de
clasificación de riesgo de cáncer de colon.

Evalúa y_true vs y_pred de modelos de clasificación médica.

Uso:
    from src.metrics import AccuracyMetric, RecallMetric, FBetaMetric
    from src.pipelines.evaluation_pipeline import ModelEvaluationPipeline

    pipeline = ModelEvaluationPipeline(metrics=[
        AccuracyMetric(),
        RecallMetric(),
        FBetaMetric(beta=2),
    ])

    results = pipeline.evaluate_model("LightGBM", y_test, y_pred, y_proba)
    df_summary = pipeline.get_summary_dataframe()
=============================================================================
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.metrics.protocols import ClassificationMetricProtocol


class ModelEvaluationPipeline:
    """
    Pipeline de evaluación que ejecuta una lista de métricas clínicas
    sobre las predicciones de uno o varios modelos.
    """

    def __init__(self, metrics: list[ClassificationMetricProtocol]):
        """
        Parameters
        ----------
        metrics : list
            Lista de métricas que implementan ClassificationMetricProtocol.
        """
        self.metrics = metrics
        self.results: dict[str, dict[str, float | dict]] = {}

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> dict[str, float | dict]:
        """
        Evalúa un modelo con todas las métricas cargadas.

        Parameters
        ----------
        model_name : str
            Nombre identificativo del modelo (ej: "LightGBM", "XGBoost").
        y_true : np.ndarray
            Etiquetas reales (ground truth).
        y_pred : np.ndarray
            Predicciones del modelo.
        y_proba : np.ndarray, optional
            Probabilidades por clase (necesario para ROC-AUC).

        Returns
        -------
        dict[str, float | dict]
            Diccionario con el resultado de cada métrica.
        """
        model_results = {}

        for metric in self.metrics:
            name = metric.name
            try:
                score = metric.compute(y_true, y_pred, y_proba)
                model_results[name] = score
            except Exception as e:
                print(f"[ALERTA] Fallo al computar {name} en {model_name}: {e}")
                model_results[name] = 0.0

        self.results[model_name] = model_results
        return model_results

    def evaluate_multiple(
        self,
        models: dict[str, dict],
    ) -> pd.DataFrame:
        """
        Evalúa múltiples modelos de una vez.

        Parameters
        ----------
        models : dict
            Diccionario con estructura:
            {
                "LightGBM": {"y_pred": array, "y_proba": array},
                "XGBoost": {"y_pred": array, "y_proba": array},
            }
            Los y_true se pasan por separado ya que son compartidos.

        Returns
        -------
        pd.DataFrame con todas las métricas de todos los modelos.
        """
        for model_name, preds in models.items():
            self.evaluate_model(
                model_name=model_name,
                y_true=preds["y_true"],
                y_pred=preds["y_pred"],
                y_proba=preds.get("y_proba"),
            )

        return self.get_summary_dataframe()

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Exporta todos los resultados en un DataFrame limpio.

        Filtra los valores que son diccionarios (como la confusion matrix)
        para que el DataFrame solo contenga valores escalares.
        """
        records = []
        for model_name, metrics_dict in self.results.items():
            record = {"Modelo": model_name}
            for metric_name, value in metrics_dict.items():
                # Solo incluimos valores escalares en la tabla resumen
                if isinstance(value, (int, float)):
                    record[metric_name] = round(value, 4)
            records.append(record)

        return pd.DataFrame(records)

    def print_report(self, model_name: str | None = None):
        """Imprime un informe formateado de las métricas."""
        targets = {model_name: self.results[model_name]} if model_name else self.results

        for name, metrics in targets.items():
            print(f"\n{'=' * 60}")
            print(f"  EVALUACIÓN: {name}")
            print(f"{'=' * 60}")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name:<30} : {value:.4f}")
                elif isinstance(value, dict) and "matrix" in value:
                    print(f"  {metric_name:<30} :")
                    for row in value["matrix"]:
                        print(f"    {row}")
            print(f"{'=' * 60}")
