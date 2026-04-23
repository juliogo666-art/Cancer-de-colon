"""
=============================================================================
Registro de experimentos de entrenamiento
=============================================================================
Guarda los resultados de cada sesión de entrenamiento en un archivo JSON
centralizado, creando un historial completo de los modelos entrenados.

Cada registro incluye:
    - Nombre del modelo y tipo de algoritmo
    - Hiperparámetros utilizados
    - Métricas de evaluación (Accuracy, Recall, F1, F2, ROC-AUC, etc.)
    - Features utilizadas
    - Fecha y duración del entrenamiento
    - Ruta al modelo guardado

Esto sustituye la práctica actual de "imprimir las métricas por consola
y después perderlas", proporcionando un registro auditable de todos
los experimentos realizados.

Uso:
    from src.tracking.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker()

    experiment_id = tracker.log_experiment(
        model_name="LightGBM",
        hyperparameters={"n_estimators": 300, "learning_rate": 0.02},
        metrics={"Accuracy": 0.85, "Recall": 0.91, "F2-Score": 0.89},
        features=["Smoking", "Alcohol_Use", ...],
        model_path="src/models/ml/lgbm_clinico.pkl",
        notes="Entrenado con cancer_risk_final.csv, 5000 registros",
    )
=============================================================================
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Optional
import matplotlib.pyplot as plt

# Directorio por defecto para guardar los logs de experimentos
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_TRACKER_PATH = os.path.join(
    _PROJECT_ROOT, "src", "tracking", "experiments.json"
)


class ExperimentTracker:
    """
    Registra y consulta el historial de experimentos de entrenamiento.

    Los experimentos se almacenan en un archivo JSON con formato:
    {
        "experiment_id": {
            "model_name": "LightGBM",
            "timestamp": "2026-04-07T22:30:00",
            "metrics": {...},
            ...
        }
    }
    """

    def __init__(self, tracker_path: str = DEFAULT_TRACKER_PATH):
        """
        Parameters
        ----------
        tracker_path : str
            Ruta al archivo JSON donde se almacenan los experimentos.
        """
        self.tracker_path = tracker_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Crea el archivo y directorio si no existen."""
        os.makedirs(os.path.dirname(self.tracker_path), exist_ok=True)
        if not os.path.exists(self.tracker_path):
            with open(self.tracker_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load(self) -> dict:
        """Carga todos los experimentos del archivo."""
        with open(self.tracker_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict):
        """Guarda todos los experimentos al archivo."""
        with open(self.tracker_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def log_experiment(
        self,
        model_name: str,
        metrics: dict[str, float | dict],
        hyperparameters: Optional[dict[str, Any]] = None,
        features: Optional[list[str]] = None,
        dataset_path: Optional[str] = None,
        model_path: Optional[str] = None,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        notes: Optional[str] = None,
        training_history: Optional[dict[str, list[float]]] = None,
    ) -> str:
        """
        Registra un nuevo experimento de entrenamiento.

        Parameters
        ----------
        model_name : str
            Nombre del modelo (ej: "LightGBM", "XGBoost").
        metrics : dict
            Métricas de evaluación obtenidas.
        hyperparameters : dict, optional
            Hiperparámetros usados en el entrenamiento.
        features : list[str], optional
            Lista de features utilizadas.
        dataset_path : str, optional
            Ruta al dataset usado.
        model_path : str, optional
            Ruta donde se guardó el modelo.
        train_size : int, optional
            Número de muestras de entrenamiento.
        test_size : int, optional
            Número de muestras de test.
        duration_seconds : float, optional
            Duración del entrenamiento en segundos.
        notes : str, optional
            Notas adicionales del experimentador.
        training_history : dict, optional
            Historial de entrenamiento por época (ej. train_loss vs val_loss).

        Returns
        -------
        str : ID único del experimento registrado.
        """
        experiment_id = str(uuid.uuid4())[:8]

        # Filtrar métricas para serialización (solo escalares y dicts)
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                clean_metrics[k] = round(v, 6)
            elif isinstance(v, dict):
                clean_metrics[k] = v

        record = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": clean_metrics,
            "hyperparameters": hyperparameters or {},
            "features": features or [],
            "dataset_path": dataset_path,
            "model_path": model_path,
            "train_size": train_size,
            "test_size": test_size,
            "duration_seconds": duration_seconds,
            "training_history": training_history,
            "notes": notes,
        }

        data = self._load()
        data[experiment_id] = record
        self._save(data)
        
        if training_history and "train_loss" in training_history and "val_loss" in training_history:
            self._plot_training_history(model_name, training_history)

        print(f"[TRACKER] Experimento '{experiment_id}' registrado para {model_name}")
        return experiment_id

    def _plot_training_history(self, model_name: str, history: dict):
        """Genera y guarda gráficos de entrenamiento para detectar overfitting."""
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(10, 4))
        
        # Gráfico de Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], 'b-', label='Train Loss')
        plt.plot(epochs, history["val_loss"], 'r-', label='Val Loss')
        plt.title(f'{model_name} - Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Gráfico de Métricas (si existen)
        if "val_accuracy" in history and history["val_accuracy"]:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, history["val_accuracy"], 'g-', label='Val Accuracy')
            if "val_f1" in history and history["val_f1"]:
                plt.plot(epochs, history["val_f1"], 'm-', label='Val F1')
            plt.title(f'{model_name} - Metrics')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            
        plt.tight_layout()
        artifacts_dir = os.path.join(_PROJECT_ROOT, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        safe_name = model_name.replace(" ", "_").lower()
        plt.savefig(os.path.join(artifacts_dir, f"training_history_{safe_name}.png"))
        plt.close()

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        """Devuelve un experimento por su ID."""
        data = self._load()
        return data.get(experiment_id)

    def get_all_experiments(self) -> dict:
        """Devuelve todos los experimentos registrados."""
        return self._load()

    def get_best_experiment(
        self, metric_name: str = "Accuracy"
    ) -> Optional[tuple[str, dict]]:
        """
        Devuelve el experimento con el mejor valor para una métrica dada.

        Parameters
        ----------
        metric_name : str
            Nombre de la métrica a maximizar.

        Returns
        -------
        tuple[str, dict] | None
            (experiment_id, experiment_data) o None si no hay datos.
        """
        data = self._load()
        if not data:
            return None

        best_id = None
        best_score = -float("inf")

        for exp_id, exp_data in data.items():
            score = exp_data.get("metrics", {}).get(metric_name, -float("inf"))
            if isinstance(score, (int, float)) and score > best_score:
                best_score = score
                best_id = exp_id

        if best_id:
            return best_id, data[best_id]
        return None

    def summary(self) -> str:
        """Devuelve un resumen legible de todos los experimentos."""
        data = self._load()
        if not data:
            return "No hay experimentos registrados."

        lines = [
            f"{'=' * 60}",
            f"  HISTORIAL DE EXPERIMENTOS ({len(data)} registros)",
            f"{'=' * 60}",
        ]

        for exp_id, exp in data.items():
            metrics_str = " | ".join(
                f"{k}: {v:.4f}"
                for k, v in exp.get("metrics", {}).items()
                if isinstance(v, (int, float))
            )
            lines.append(
                f"  [{exp_id}] {exp['model_name']} ({exp['timestamp'][:16]})"
                f"\n    {metrics_str}"
            )

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)
