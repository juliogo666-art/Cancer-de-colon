# Walkthrough — Componentes 1-5 Implementados

## Resumen

Se han implementado **17 archivos** nuevos/reescritos y eliminado **4 archivos** de streaming en una sola sesión. Todos los imports verificados con `python -c`.

---

## Componente 1: Backend API (`src/api/`)

| Archivo | Descripción |
|---|---|
| [dependencies.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/api/dependencies.py) | Carga centralizada de 3 modelos (ML, CNN, Biopsias) con lifespan pattern |
| [main_api.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/api/main_api.py) | FastAPI con 8 endpoints en 3 grupos |

**Endpoints creados:**
- `POST /api/v1/predict/risk` — Predicción de riesgo con 11 factores clínicos
- `POST /api/v1/analyze/colonoscopy` — Análisis de imagen + Grad-CAM
- `POST /api/v1/analyze/biopsy` — Análisis de biopsia + Grad-CAM
- `GET/POST/PUT /api/v1/patients/` — CRUD de pacientes sobre CSV

---

## Componente 2: Métricas Clínicas (`src/metrics/`)

**Eliminados:** `ndcg.py`, `hitrate.py`, `coverage.py` (métricas de ranking de películas)

**Creados/Reescritos:**

| Archivo | Métrica | Relevancia médica |
|---|---|---|
| [protocols.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/metrics/protocols.py) | `ClassificationMetricProtocol` | Contrato OOP para `(y_true, y_pred, y_proba)` |
| [accuracy.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/metrics/accuracy.py) | Accuracy | Exactitud global |
| [precision.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/metrics/precision.py) | Precision | ¿Cuántas alertas del modelo eran reales? |
| [recall.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/metrics/recall.py) | Recall | **La más crítica**: ¿cuántos enfermos detectó? |
| [f_score.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/metrics/f_score.py) | F-Beta | F2 por defecto (prioriza Recall en oncología) |
| [confusion.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/metrics/confusion.py) | Confusion Matrix | Desglose TP/TN/FP/FN + classification_report |
| [roc_auc.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/metrics/roc_auc.py) | ROC-AUC | Discriminación multiclase (One-vs-Rest) |

---

## Componente 3: Pipelines (`src/pipelines/`)

**Eliminado:** Pipeline de evaluación de recomendación de películas

**Creados:**

| Archivo | Pipeline | Qué hace |
|---|---|---|
| [evaluation_pipeline.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/pipelines/evaluation_pipeline.py) | `ModelEvaluationPipeline` | Ejecuta N métricas sobre predicciones, genera DataFrame resumen |
| [training_pipeline.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/pipelines/training_pipeline.py) | `TrainingPipeline` | Flujo completo: CSV → split → train → eval → save modelo + métricas JSON |
| [image_pipeline.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/pipelines/image_pipeline.py) | `ImageAnalysisPipeline` | Inferencia de colonoscopia + biopsias con Grad-CAM, desacoplado del frontend |

---

## Componente 4: Schemas (`src/schemas/`)

**Eliminado:** `schemas.py` (LoginRequest, RegisterRequest de streaming)

**Creados:**

| Archivo | Schemas |
|---|---|
| [patient_schemas.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/schemas/patient_schemas.py) | `PatientCreate`, `PatientUpdate`, `PatientResponse`, `PatientListResponse` |
| [prediction_schemas.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/schemas/prediction_schemas.py) | `RiskPredictionRequest`, `RiskPredictionResponse`, `SHAPExplanation`, `SHAPResponse` |
| [image_schemas.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/schemas/image_schemas.py) | `ImageAnalysisResponse`, `ColonoscopyAnalysisResponse`, `BiopsyAnalysisResponse` |

---

## Componente 5: Tracking (`src/tracking/`)

| Archivo | Componente | Formato |
|---|---|---|
| [experiment_tracker.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/tracking/experiment_tracker.py) | `ExperimentTracker` — Log de entrenamientos con métricas, hiperparámetros y búsqueda del mejor modelo | JSON |
| [prediction_logger.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/tracking/prediction_logger.py) | `PredictionLogger` — Audit trail de predicciones en producción | CSV |

---

## Verificación

```
✅ python -c "from src.metrics import ..."      → OK
✅ python -c "from src.pipelines import ..."     → OK
✅ python -c "from src.schemas import ..."       → OK
✅ python -c "from src.tracking import ..."      → OK
✅ python -c "from src.api.dependencies import ..." → OK
```

## Pendiente para mañana (Componentes 6-7)

- Fusionar los dos frontends Streamlit
- Limpieza de archivos legacy
- Configuración centralizada (`settings.py`)
- Actualizar `main.py` con comandos `api` y `frontend`
- Mover modelos ML legacy a `_legacy/`
- Añadir dependencias FastAPI al `pyproject.toml`
- Tests reales con `pytest`
