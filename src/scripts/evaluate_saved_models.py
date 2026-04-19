"""
Script para cargar modelos preentrenados y evaluarlos utilizando
toda la arquitectura OOP del proyecto (metrics + evaluation_pipeline).
Genera un informe final en CSV con todas las métricas requeridas.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import settings
from src.metrics.accuracy import AccuracyMetric
from src.metrics.precision import PrecisionMetric
from src.metrics.recall import RecallMetric
from src.metrics.f_score import FBetaMetric
from src.metrics.confusion import ConfusionMatrixMetric
from src.metrics.roc_auc import ROCAUCMetric
from src.pipelines.evaluation_pipeline import ModelEvaluationPipeline

def main():
    print("Iniciando Pipeline de Evaluación Maestro...")
    
    # 1. Cargar datos
    df = pd.read_csv(settings.CSV_MASTER_PATH)
    target = "Risk_Level_n"
    
    # Features
    features_clinico = settings.ML_FEATURE_NAMES
    features_triage = [f for f in features_clinico if f not in ["FOBT_Resultado_n", "CEA_Level_ng_mL"]]

    # Separar para asegurar que el evaluador solo prueba con un split justo (Test set)
    # Usamos la misma semilla que en el entrenamiento para que el test sea idéntico
    X_clin = df[features_clinico]
    y_clin = df[target]
    _, X_test_c, _, y_test_c = train_test_split(X_clin, y_clin, test_size=0.2, random_state=42, stratify=y_clin)
    
    X_tri = df[features_triage]
    y_tri = df[target]
    _, X_test_t, _, y_test_t = train_test_split(X_tri, y_tri, test_size=0.2, random_state=42, stratify=y_tri)

    # 2. Cargar Modelos guardados
    path_clinico = os.path.join("artifacts", "weights", "lgbm_clinico.pkl")
    path_triage = os.path.join("artifacts", "weights", "lgbm_triage.pkl")
    
    if not os.path.exists(path_clinico) or not os.path.exists(path_triage):
        print("Error: Los modelos no están entrenados. Ejecuta ml_v4_dual.py primero.")
        return

    model_clinico = joblib.load(path_clinico)
    model_triage = joblib.load(path_triage)

    # 3. Predicciones
    preds_c = model_clinico.predict(X_test_c)
    proba_c = model_clinico.predict_proba(X_test_c)
    
    preds_t = model_triage.predict(X_test_t)
    proba_t = model_triage.predict_proba(X_test_t)

    # 4. Instanciar Pipeline con TODAS las métricas de la carpeta 'metrics'
    eval_pipeline = ModelEvaluationPipeline(metrics=[
        AccuracyMetric(),
        PrecisionMetric(),
        RecallMetric(),
        FBetaMetric(beta=1), # F1 Score
        ConfusionMatrixMetric(),
        ROCAUCMetric()
    ])

    # 5. Evaluar
    eval_pipeline.evaluate_model("LightGBM_Clinico", y_test_c, preds_c, proba_c)
    eval_pipeline.evaluate_model("LightGBM_Triaje", y_test_t, preds_t, proba_t)
    
    # 6. Imprimir y guardar
    eval_pipeline.print_report()
    
    df_summary = eval_pipeline.get_summary_dataframe()
    artifacts_dir = os.path.join("artifacts", "reports")
    os.makedirs(artifacts_dir, exist_ok=True)
    report_path = os.path.join(artifacts_dir, "model_evaluation_report.csv")
    
    df_summary.to_csv(report_path, index=False)
    print(f"\n[OK] Reporte completo guardado en: {report_path}")

if __name__ == "__main__":
    main()
