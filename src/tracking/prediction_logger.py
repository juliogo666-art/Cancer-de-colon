"""
=============================================================================
Logger de predicciones en producción
=============================================================================
Registra cada predicción realizada por la API o el frontend Streamlit
en un archivo CSV centralizado para auditoría clínica.

Cada registro incluye:
    - Timestamp de la predicción
    - Tipo de predicción (riesgo ML, colonoscopia, biopsia)
    - Patient_ID (si aplica)
    - Resultado del diagnóstico
    - Confianza del modelo
    - Inputs utilizados (features o nombre de imagen)

Esto es esencial en un contexto médico donde cada predicción debe ser
trazable y reproducible para validación clínica.

Uso:
    from src.tracking.prediction_logger import PredictionLogger

    logger = PredictionLogger()

    logger.log_risk_prediction(
        patient_id=42,
        risk_level="High",
        risk_score=0.85,
        features={"Smoking": 8, "BMI": 32.5, ...}
    )

    logger.log_image_prediction(
        analysis_type="colonoscopy",
        diagnosis="POLIPO DETECTADO",
        confidence=0.92,
    )
=============================================================================
"""

import csv
import os
from datetime import datetime
from typing import Any, Optional


# Directorio por defecto para los logs de predicciones
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_LOG_PATH = os.path.join(_PROJECT_ROOT, "src", "tracking", "predictions.csv")

# Cabeceras del CSV de predicciones
CSV_HEADERS = [
    "timestamp",
    "prediction_type",
    "patient_id",
    "diagnosis",
    "confidence",
    "risk_score",
    "risk_level",
    "details",
]


class PredictionLogger:
    """
    Registra predicciones en un CSV para auditoría clínica.

    El CSV se crea automáticamente con cabeceras si no existe.
    Los registros se añaden de forma incremental (append).
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        """
        Parameters
        ----------
        log_path : str
            Ruta al archivo CSV donde se almacenan los logs.
        """
        self.log_path = log_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Crea el archivo CSV con cabeceras si no existe."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

    def _append_row(self, row: list):
        """Añade una fila al CSV."""
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_risk_prediction(
        self,
        patient_id: Optional[int] = None,
        risk_level: str = "",
        risk_score: float = 0.0,
        confidence: float = 0.0,
        features: Optional[dict[str, Any]] = None,
    ):
        """
        Registra una predicción de riesgo clínico (ML).

        Parameters
        ----------
        patient_id : int, optional
            ID del paciente analizado.
        risk_level : str
            Nivel de riesgo predicho (Low, Medium, High).
        risk_score : float
            Score ponderado de riesgo (0-1).
        confidence : float
            Confianza de la predicción.
        features : dict, optional
            Features clínicas utilizadas en la predicción.
        """
        self._append_row(
            [
                datetime.now().isoformat(),
                "risk_ml",
                patient_id or "",
                f"Riesgo: {risk_level}",
                round(confidence, 4),
                round(risk_score, 4),
                risk_level,
                str(features or {}),
            ]
        )

    def log_image_prediction(
        self,
        analysis_type: str,
        diagnosis: str,
        confidence: float,
        patient_id: Optional[int] = None,
        image_name: Optional[str] = None,
    ):
        """
        Registra una predicción de análisis de imagen (colonoscopia o biopsia).

        Parameters
        ----------
        analysis_type : str
            Tipo: "colonoscopy" o "biopsy".
        diagnosis : str
            Diagnóstico del modelo (ej: "POLIPO DETECTADO").
        confidence : float
            Confianza de la predicción (0-1).
        patient_id : int, optional
            ID del paciente si se conoce.
        image_name : str, optional
            Nombre del archivo de imagen analizado.
        """
        self._append_row(
            [
                datetime.now().isoformat(),
                f"image_{analysis_type}",
                patient_id or "",
                diagnosis,
                round(confidence, 4),
                "",
                "",
                f"image: {image_name or 'upload'}",
            ]
        )

    def get_history(self, limit: int = 100) -> list[dict]:
        """
        Lee las últimas N predicciones del log.

        Parameters
        ----------
        limit : int
            Número máximo de registros a devolver (las más recientes).

        Returns
        -------
        list[dict] : Lista de predicciones como diccionarios.
        """
        if not os.path.exists(self.log_path):
            return []

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Devolver las últimas N filas (las más recientes)
            return rows[-limit:]
        except Exception:
            return []

    def get_stats(self) -> dict:
        """
        Devuelve estadísticas resumidas de todas las predicciones registradas.

        Returns
        -------
        dict con total de predicciones, desglose por tipo y conteo de diagnósticos.
        """
        history = self.get_history(limit=10000)
        if not history:
            return {"total": 0}

        stats = {
            "total": len(history),
            "by_type": {},
            "by_diagnosis": {},
        }

        for row in history:
            pred_type = row.get("prediction_type", "unknown")
            diagnosis = row.get("diagnosis", "unknown")

            stats["by_type"][pred_type] = stats["by_type"].get(pred_type, 0) + 1
            stats["by_diagnosis"][diagnosis] = (
                stats["by_diagnosis"].get(diagnosis, 0) + 1
            )

        return stats
