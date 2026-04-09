"""
=============================================================================
Pipeline unificado de entrenamiento
=============================================================================
Encapsula el flujo completo de entrenamiento de modelos ML para predicción
de riesgo de cáncer de colon:

    1. Carga de datos desde CSV
    2. Selección de features y target
    3. División train/test con stratify
    4. Entrenamiento de uno o varios modelos
    5. Evaluación con las métricas clínicas
    6. Guardado del modelo + métricas + metadatos

Esto unifica la lógica que antes estaba dispersa en ml_v0.py, ml_v1.py,
ml_v2.py y ml_v3.py en un único flujo reutilizable.

Uso:
    from src.pipelines.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline(
        csv_path="src/data/clean/cancer_risk_final.csv",
        target_col="Risk_Level_n",
    )
    pipeline.load_and_prepare()
    results = pipeline.train_and_evaluate(modelo, "LightGBM")
    pipeline.save_model(modelo, "lgbm_clinico")
=============================================================================
"""

import os
import json
from datetime import datetime
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.metrics import (
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    FBetaMetric,
    ConfusionMatrixMetric,
    ROCAUCMetric,
)
from src.pipelines.evaluation_pipeline import ModelEvaluationPipeline


# Features clínicas estándar del proyecto
DEFAULT_FEATURES = [
    "Smoking",
    "Alcohol_Use",
    "Obesity",
    "Family_History",
    "Diet_Red_Meat",
    "Diet_Salted_Processed",
    "Fruit_Veg_Intake",
    "Physical_Activity",
    "BMI",
    "FOBT_Resultado_n",
    "CEA_Level_ng_mL",
]

# Directorio por defecto para guardar modelos
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, "src", "models", "ml")


class TrainingPipeline:
    """
    Pipeline de entrenamiento unificado para modelos de clasificación
    de riesgo de cáncer de colon.
    """

    def __init__(
        self,
        csv_path: str,
        target_col: str = "Risk_Level_n",
        features: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        csv_path : str
            Ruta al CSV con los datos de pacientes.
        target_col : str
            Columna objetivo (por defecto: Risk_Level_n → 0=Low, 1=Medium, 2=High).
        features : list[str], optional
            Lista de features a usar. Si es None, usa DEFAULT_FEATURES.
        test_size : float
            Proporción de datos para test (0.2 = 20%).
        random_state : int
            Semilla para reproducibilidad.
        """
        self.csv_path = csv_path
        self.target_col = target_col
        self.features = features or DEFAULT_FEATURES
        self.test_size = test_size
        self.random_state = random_state

        # Se rellenan al llamar load_and_prepare()
        self.df: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Pipeline de evaluación con todas las métricas clínicas
        self.eval_pipeline = ModelEvaluationPipeline(
            metrics=[
                AccuracyMetric(),
                PrecisionMetric(average="macro"),
                RecallMetric(average="macro"),
                FBetaMetric(beta=1.0, average="macro"),
                FBetaMetric(beta=2.0, average="macro"),
                ConfusionMatrixMetric(),
                ROCAUCMetric(),
            ]
        )

    def load_and_prepare(self) -> "TrainingPipeline":
        """
        Carga el CSV, selecciona features y divide en train/test.

        Returns
        -------
        self (para encadenamiento)
        """
        print(f"[INFO] Cargando datos desde: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)

        # Verificar que las columnas existen
        missing = [f for f in self.features if f not in self.df.columns]
        if missing:
            raise ValueError(f"Columnas no encontradas en el CSV: {missing}")

        if self.target_col not in self.df.columns:
            raise ValueError(f"Columna target '{self.target_col}' no encontrada.")

        X = self.df[self.features].values
        y = self.df[self.target_col].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"[INFO] Datos cargados: {len(self.df)} registros")
        print(f"[INFO] Split: {len(self.X_train)} train / {len(self.X_test)} test")
        print(f"[INFO] Features: {self.features}")
        print(f"[INFO] Target: {self.target_col}")

        return self

    def train_and_evaluate(
        self,
        model: Any,
        model_name: str,
    ) -> dict:
        """
        Entrena el modelo y lo evalúa con todas las métricas clínicas.

        Parameters
        ----------
        model : sklearn-compatible estimator
            Modelo con .fit(), .predict() y opcionalmente .predict_proba().
        model_name : str
            Nombre identificativo para el informe.

        Returns
        -------
        dict con métricas y metadatos del entrenamiento.
        """
        if self.X_train is None:
            raise RuntimeError("Ejecuta load_and_prepare() primero.")

        # Entrenamiento
        print(f"\n[ENTRENANDO] {model_name}...")
        model.fit(self.X_train, self.y_train)

        # Predicción
        y_pred = model.predict(self.X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(self.X_test)

        # Evaluación con todas las métricas
        metrics = self.eval_pipeline.evaluate_model(
            model_name=model_name,
            y_true=self.y_test,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        # Imprimir informe
        self.eval_pipeline.print_report(model_name)

        return {
            "model_name": model_name,
            "metrics": metrics,
            "model": model,
            "timestamp": datetime.now().isoformat(),
        }

    def save_model(
        self,
        model: Any,
        filename: str,
        output_dir: str | None = None,
        save_metrics: bool = True,
    ) -> str:
        """
        Guarda el modelo entrenado y opcionalmente sus métricas.

        Parameters
        ----------
        model : trained model
        filename : str
            Nombre base del archivo (sin extensión).
        output_dir : str, optional
            Directorio de salida. Por defecto: src/models/ml/
        save_metrics : bool
            Si True, guarda también un JSON con las métricas.

        Returns
        -------
        str : Ruta al modelo guardado.
        """
        output_dir = output_dir or DEFAULT_MODEL_DIR
        os.makedirs(output_dir, exist_ok=True)

        # Guardar modelo
        model_path = os.path.join(output_dir, f"{filename}.pkl")
        joblib.dump(model, model_path)
        print(f"[GUARDADO] Modelo → {model_path}")

        # Guardar métricas
        if save_metrics and self.eval_pipeline.results:
            metrics_path = os.path.join(output_dir, f"{filename}_metrics.json")
            # Filtrar solo valores serializables
            serializable = {}
            for model_name, metrics_dict in self.eval_pipeline.results.items():
                serializable[model_name] = {}
                for k, v in metrics_dict.items():
                    if isinstance(v, (int, float)):
                        serializable[model_name][k] = round(v, 4)
                    elif isinstance(v, dict):
                        serializable[model_name][k] = v

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
            print(f"[GUARDADO] Métricas → {metrics_path}")

        return model_path

    def get_summary(self) -> pd.DataFrame:
        """Devuelve un DataFrame resumen de todos los modelos evaluados."""
        return self.eval_pipeline.get_summary_dataframe()
