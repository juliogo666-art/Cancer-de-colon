# Plan de Implementación — Proyecto 2: Cáncer de Colón

## Resumen del Estado Actual

Tras revisar **todos** los archivos del proyecto, la estructura es la siguiente:

```
├── main.py                          # CLI con argparse (eda, train-polyps, train-rf, generate-data)
├── src/
│   ├── api/                         # VACÍO (solo __init__.py vacío)
│   ├── config/                      # rules.yaml VACÍO
│   ├── data/
│   │   ├── raw/                     # Datasets reales + sintéticos (CSV, imágenes)
│   │   ├── clean/                   # VACÍO
│   │   ├── add_Data.py              # Script legacy con rutas hardcodeadas
│   │   └── api_call_img.py          # Descarga de datasets desde Kaggle/HF
│   ├── frontend/
│   │   ├── app_s.py                 # Streamlit principal (Tab datos + Tab visión)
│   │   └── app_s_busqueda.py        # Streamlit alternativo con búsqueda DNI/NUSS
│   ├── models/
│   │   ├── modelo_busca_polipos_Clas.py   # ResNet18 clasificador (PyTorch)
│   │   ├── modelo_busca_polipos_Segment.py # Solo TODO
│   │   ├── polyp_resnet18.pth       # Pesos entrenados (~44MB)
│   │   └── ml/                      # 4 versiones de ML (v0-v3) + modelos .pkl
│   ├── networks/
│   │   ├── dl/                      # 6 versiones CNN TensorFlow (v0-v5) + modelo .keras
│   │   └── dl_biopsia/              # ResNet18 biopsias (PyTorch) + pesos .pth
│   ├── pipelines/                   # ⚠️ TRAÍDO DE STREAMING
│   │   └── evaluation_pipeline.py   # Orquestador de métricas de RECOMENDACIÓN
│   ├── metrics/                     # ⚠️ TRAÍDO DE STREAMING
│   │   ├── protocols.py             # MetricProtocol para recomendaciones (userId, tmdb_id)
│   │   ├── precision.py, recall.py  # Precision@K, Recall@K para recomendaciones
│   │   ├── ndcg.py, hitrate.py      # NDCG@K, HitRate@K para recomendaciones
│   │   └── coverage.py             # Coverage@K de catálogo
│   ├── schemas/                     # ⚠️ TRAÍDO DE STREAMING
│   │   └── schemas.py              # LoginRequest, RegisterRequest (streaming)
│   ├── tracking/                    # ⚠️ TRAÍDO DE STREAMING — COMPLETAMENTE VACÍO
│   ├── scripts/
│   │   ├── eda.py                   # App Streamlit EDA (6 secciones)
│   │   ├── data_cleaning.py         # Funciones de limpieza de datos
│   │   ├── sintetiza_historiales.py  # Generación de datos sintéticos
│   │   └── aumento_historiales_factores_riesgo.py # Augmentation de factores
│   ├── test/
│   │   ├── test_dp_v2.py            # Test de predicción DL (rutas legacy)
│   │   └── test_ml_v0.py            # Entrenamiento RF (NO es un test real)
│   └── utils/
│       ├── cargar_modelos_s.py      # Carga modelos CNN+Biopsia+ML + SHAP (Streamlit)
│       ├── data_load_s.py           # Carga datos clínicos para Streamlit
│       ├── data_clean.py            # Script legacy con rutas absolutas hardcodeadas
│       ├── eda_visualization.py     # 712 líneas de gráficos Streamlit
│       ├── gradcam_utils.py         # Grad-CAM para PyTorch y TensorFlow
│       └── ClientsData.R            # Script R huérfano
```

---

## 1. ¿Necesitas un `main_api.py` con FastAPI?

> [!IMPORTANT]
> **SÍ, lo necesitas.** El proyecto actualmente funciona como una app monolítica Streamlit donde la lógica de negocio (predicción ML, análisis de imagen, acceso a datos) está acoplada directamente al frontend. Esto es un problema para:
> - **Escalabilidad**: No puedes servir la IA a otro cliente (app móvil, web, otro equipo).
> - **Separación de responsabilidades**: El frontend Streamlit no debería cargar modelos PyTorch/TensorFlow directamente.
> - **Evaluación académica**: Un backend API demuestra arquitectura profesional.

### Cómo lo abordaría

El `main_api.py` con FastAPI tendría **3 grupos de endpoints**:

#### Grupo 1: Predicción de Riesgo Clínico (ML)
| Endpoint | Método | Función |
|---|---|---|
| `POST /api/v1/predict/risk` | POST | Recibe factores de riesgo → devuelve probabilidad y nivel |
| `GET /api/v1/predict/explain/{patient_id}` | GET | Devuelve la explicación SHAP del último análisis |

#### Grupo 2: Análisis de Imagen (DL)
| Endpoint | Método | Función |
|---|---|---|
| `POST /api/v1/analyze/colonoscopy` | POST | Recibe imagen → devuelve probabilidad pólipo + heatmap Grad-CAM |
| `POST /api/v1/analyze/biopsy` | POST | Recibe imagen → devuelve probabilidad maligno + heatmap Grad-CAM |

#### Grupo 3: Gestión de Pacientes (CRUD)
| Endpoint | Método | Función |
|---|---|---|
| `GET /api/v1/patients` | GET | Lista pacientes (paginación) |
| `GET /api/v1/patients/{id}` | GET | Obtiene datos de un paciente por ID |
| `POST /api/v1/patients` | POST | Crea nuevo paciente |
| `PUT /api/v1/patients/{id}` | PUT | Actualiza datos de un paciente |

**Carga de modelos al iniciar** (`lifespan`):
```python
# Al arrancar la API, cargamos los modelos una sola vez en memoria
app.state.modelo_ml = joblib.load("src/models/ml/lgbm_clinico.pkl")
app.state.modelo_colonoscopia = tf.keras.models.load_model("src/networks/dl/modelo_pro_agresivo.keras")
app.state.modelo_biopsia = cargar_pesos_pytorch("src/networks/dl_biopsia/biopsia_resnet18_best.pth")
```

**Base de datos**: Para este proyecto, CSV es suficiente (ya lo estáis usando). No necesitáis SQLite/PostgreSQL a menos que el profesor lo pida. El API simplemente leerá/escribirá los CSV existentes.

---

## 2. Carpetas traídas del Proyecto 4 (Streaming) — Diagnóstico completo

### 📁 `src/metrics/` — ❌ BORRAR TODO y REESCRIBIR

| Archivo | Diagnóstico | Razonamiento |
|---|---|---|
| `protocols.py` | ❌ **BORRAR** | Define `MetricProtocol` con parámetros `recommendations: dict[userId → list[tmdb_id]]`. Esto es 100% para recomendación de películas. No aplica a clasificación clínica. |
| `precision.py` | ❌ **BORRAR** | `PrecisionAtK` calcula hit-rate en listas de `tmdb_id`. Nada que ver con precision clínica (TP/FP). |
| `recall.py` | ❌ **BORRAR** | `RecallAtK` para listas de recomendaciones. No es recall de clasificación médica. |
| `ndcg.py` | ❌ **BORRAR** | `NDCGAtK` evalúa ranking de películas. No aplica. |
| `hitrate.py` | ❌ **BORRAR** | Métrica binaria de recomendación. No aplica. |
| `coverage.py` | ❌ **BORRAR** | Mide cobertura de catálogo de películas. No aplica. |

**Qué CREAR en su lugar** — Métricas para clasificación clínica:

| Archivo nuevo | Contenido |
|---|---|
| `protocols.py` | `ClassificationMetricProtocol` con `compute(y_true, y_pred, y_proba) → dict` |
| `accuracy.py` | `AccuracyMetric` — Accuracy global |
| `precision.py` | `PrecisionMetric` — Precision por clase (especialmente clase "High") |
| `recall.py` | `RecallMetric` — Recall por clase (importantísimo en cáncer: minimizar FN) |
| `f_score.py` | `FBetaMetric` — F1, F2 (F2 da más peso al recall, vital en diagnóstico médico) |
| `confusion.py` | `ConfusionMatrixMetric` — Devuelve la matriz TP/TN/FP/FN |
| `roc_auc.py` | `ROCAUCMetric` — Área bajo la curva ROC |

### 📁 `src/pipelines/` — ❌ BORRAR y REESCRIBIR

| Archivo | Diagnóstico |
|---|---|
| `evaluation_pipeline.py` | ❌ **BORRAR** — Itera sobre `dict[userId → list[tmdb_id]]` evaluando métricas de ranking. Para cáncer necesitamos un pipeline que evalúe `y_true` vs `y_pred` de modelos ML/DL. |

**Qué CREAR en su lugar**:

| Archivo nuevo | Contenido |
|---|---|
| `evaluation_pipeline.py` | `ModelEvaluationPipeline` — Recibe modelo, X_test, y_test → calcula todas las métricas clínicas en una sola pasada |
| `training_pipeline.py` | `TrainingPipeline` — Encapsula el flujo completo: cargar datos → split → entrenar → evaluar → guardar modelo + métricas (unifica ml_v0..v3) |
| `image_pipeline.py` | `ImageAnalysisPipeline` — Encapsula carga de imagen → preprocesado → inferencia → Grad-CAM → resultado |

### 📁 `src/schemas/` — ❌ BORRAR y REESCRIBIR

| Archivo | Diagnóstico |
|---|---|
| `schemas.py` | ❌ **BORRAR** — Contiene `LoginRequest` y `RegisterRequest` para sistema de login de streaming. No aplica en absoluto a este proyecto. |

**Qué CREAR en su lugar**:

| Archivo nuevo | Contenido |
|---|---|
| `patient_schemas.py` | `PatientInput`, `PatientUpdate` — Validación Pydantic de datos de pacientes |
| `prediction_schemas.py` | `RiskPredictionRequest`, `RiskPredictionResponse` — Entrada/salida de predicción ML |
| `image_schemas.py` | `ImageAnalysisResponse` — Resultado del análisis de imagen (probabilidad, tipo, recomendación) |

### 📁 `src/tracking/` — ⚠️ MANTENER VACÍO → LLENAR

| Estado actual | Acción |
|---|---|
| Solo `__init__.py` vacío | ✅ MANTENER estructura, CREAR contenido |

**Qué CREAR**:

| Archivo nuevo | Contenido |
|---|---|
| `experiment_tracker.py` | Clase que guarda métricas de cada entrenamiento en un JSON/CSV (fecha, modelo, accuracy, recall, params, etc.). Sustituye la práctica actual de "imprimir por consola y olvidar". |
| `prediction_logger.py` | Registra cada predicción hecha (patient_id, timestamp, resultado, confianza) para auditoría clínica. |

---

## 3. Disonancias entre programas — Mapa de conflictos

> [!WARNING]
> Se han encontrado **múltiples disonancias** que requieren reestructuración.

### 3.1 Dos frontends Streamlit incompatibles

| `app_s.py` | `app_s_busqueda.py` |
|---|---|
| Busca por `Patient_ID` numérico | Busca por `DNI` / `NUSS` |
| Usa `cancer_risk_final.csv` + `nuevos_pacientes_5000.csv` | Lee `csv_path` global al importar (side-effect) |
| Carga modelo ML `lgbm_clinico.pkl` | No usa modelo ML |
| Tab2: Colonoscopia + Biopsias | Tab2: Solo Colonoscopia |
| Usa `predecir()` con SHAP | No tiene predicción ML |

**Resolución**: Fusionar ambos en un solo `app.py` que combine lo mejor de los dos:
- Búsqueda por DNI/NUSS (de `app_s_busqueda`)
- Predicción ML con SHAP (de `app_s`)
- Ambos análisis de imagen: Colonoscopia + Biopsias (de `app_s`)
- Sin side-effects al importar

### 3.2 Cuatro versiones de modelos ML (ml_v0..v3) sin consolidar

| Versión | Dataset | Target | Features | Problema |
|---|---|---|---|---|
| `ml_v0.py` | `datos_finales_Kaggle.csv` | `Survival_Prediction` | 12 features | Rutas absolutas `C:/Users/Ana-L/...` |
| `ml_v1.py` | `datos_finales_Kaggle.csv` | `Survival_Prediction` | 12 features | Rutas absolutas + SMOTE + GPU |
| `ml_v2.py` | `datos_finales_Kaggle.csv` | `Survival_Prediction` | 13 features | Rutas relativas, umbral ajustable |
| `ml_v3.py` | `cancer_risk_final.csv` | `Risk_Level_n` | 11 features clínicas | **Este es el actual**, usa ensemble |

**Resolución**: Solo `ml_v3.py` es relevante. Los anteriores son iteraciones históricas. Se marcarán como `_legacy/` o se eliminan.

### 3.3 `test/` no contiene tests reales

| Archivo | Qué es realmente |
|---|---|
| `test_ml_v0.py` | Script de **entrenamiento** de Random Forest (no un test) |
| `test_dp_v2.py` | Script de **inferencia** DL con rutas absolutas legacy |

**Resolución**: Mover contenido útil a `scripts/` y crear tests unitarios reales con `pytest`.

### 3.4 `data_clean.py` (utils) vs `data_cleaning.py` (scripts)

| `src/utils/data_clean.py` | `src/scripts/data_cleaning.py` |
|---|---|
| Hardcoded: `'/cancer de colon/prueba/...'` | Rutas relativas, funciones limpias |
| Ejecuta al importar (side-effect) | Funciones invocables |
| Clasifica imágenes polipo/sano | Limpia CSVs de pacientes |

**Resolución**: `data_clean.py` es legacy y debe eliminarse o refactorizarse.

### 3.5 `main.py` no hace referencia al frontend `app_s.py`

El `main.py` llama a `src/scripts/eda.py` (EDA antiguo), pero el frontend principal está en `src/frontend/app_s.py`. Son aplicaciones Streamlit completamente diferentes.

**Resolución**: `main.py` debería tener un comando `frontend` que lance `app_s.py`.

### 3.6 Modelo ML que carga el frontend no existe

`app_s.py` busca `lgbm_clinico.pkl` pero en el repo solo existen:
- `lgbm_sensible.pkl` (de ml_v2)
- `best_rf_model.pkl` (de ml_v1)
- `xgb_sensible.pkl` (de ml_v2)

**Resolución**: Entrenar con `ml_v3.py` genera `lgbm_clinico.pkl`, `xgb_clinico.pkl`, `rf_clinico.pkl`. Verificar que se ejecutó correctamente.

---

## 4. Qué más modificaría y/o añadiría

### 4.1 Archivos a ELIMINAR

| Archivo | Razón |
|---|---|
| `src/utils/data_clean.py` | Script legacy con rutas absolutas hardcodeadas, ejecuta side-effects al importar |
| `src/utils/ClientsData.R` | Script R sin conexión con el proyecto Python |
| `src/data/add_Data.py` | Script legacy con rutas absolutas hardcodeadas |
| `src/data/raw/historial_pacientes/Crear pacientes sinteticos.R` | Script R, la funcionalidad ya existe en Python |
| `src/data/raw/historial_pacientes/Crear user.R` | Script R sin uso |
| `src/data/raw/historial_pacientes/.RData` | Datos de sesión R |
| `src/data/raw/historial_pacientes/.Rhistory` | Historial R |
| `src/utils/.Rhistory` | Historial R |
| `src/config/rules.yaml` | Archivo vacío |
| `src/models/implementation_plan.md` | Plan ya ejecutado, mover a `docs/` |
| `src/models/implementation_plan_mbp_clas.md` | Plan ya ejecutado, mover a `docs/` |

### 4.2 Crear estructura `docs/`

```
docs/
├── architecture.md          # Diagrama de arquitectura del sistema
├── models_catalog.md        # Catálogo: qué modelo hace qué, métricas, fecha
├── api_reference.md         # Documentación de endpoints FastAPI
└── deployment_guide.md      # Cómo arrancar el proyecto
```

### 4.3 Crear `src/config/settings.py` — Configuración centralizada

Actualmente las rutas están hardcodeadas en cada archivo. Centralizar con Pydantic Settings:
```python
class Settings(BaseSettings):
    MODEL_ML_PATH: str = "src/models/ml/lgbm_clinico.pkl"
    MODEL_CNN_PATH: str = "src/networks/dl/modelo_pro_agresivo.keras"
    MODEL_BIOPSY_PATH: str = "src/networks/dl_biopsia/biopsia_resnet18_best.pth"
    CSV_RISK_PATH: str = "src/data/raw/historial_pacientes/cancer_risk_final.csv"
    CSV_PATIENTS_PATH: str = "src/data/raw/historial_pacientes/nuevos_pacientes_5000.csv"
```

### 4.4 Marcar versiones legacy

Crear subcarpeta `src/models/ml/_legacy/` y mover `ml_v0.py`, `ml_v1.py`, `ml_v2.py` ahí. Solo `ml_v3.py` queda activo.

### 4.5 Tests reales con `pytest`

```
src/test/
├── test_api.py              # Tests de endpoints FastAPI (httpx + TestClient)
├── test_ml_prediction.py    # Test que carga modelo ML y predice un caso conocido
├── test_image_pipeline.py   # Test que carga modelo DL y clasifica imagen de prueba  
└── test_data_loading.py     # Test que verifica que los CSV se cargan sin error
```

### 4.6 Añadir dependencias al `pyproject.toml`

Faltan en las dependencias actuales:
- `fastapi` — Para el backend API
- `uvicorn` — Servidor ASGI
- `python-multipart` — Subida de archivos en FastAPI
- `pydantic-settings` — Configuración tipada
- `pytest` — Testing
- `httpx` — Cliente HTTP para tests

### 4.7 Mover datos limpios a `data/clean/`

Actualmente `data/clean/` está vacío y los CSV limpios (`cancer_risk_clean.csv`, etc.) están mezclados en `raw/`. Separar correctamente.

### 4.8 `.gitignore` — Añadir modelos pesados

Verificar que `.gitignore` excluye:
```
*.pth
*.pkl
*.joblib
*.keras
*.h5
src/data/raw/
```

---

## Proposed Changes — Resumen por componente

### Componente 1: Backend API
#### [NEW] `src/api/main_api.py`
FastAPI con endpoints de predicción ML, análisis de imagen y CRUD de pacientes.

#### [NEW] `src/api/dependencies.py`
Carga de modelos compartida (lifespan pattern).

---

### Componente 2: Métricas Clínicas (sustituye métricas de streaming)
#### [DELETE] `src/metrics/protocols.py`, `precision.py`, `recall.py`, `ndcg.py`, `hitrate.py`, `coverage.py`
#### [NEW] `src/metrics/protocols.py` — Protocolo para métricas de clasificación
#### [NEW] `src/metrics/accuracy.py`, `precision.py`, `recall.py`, `f_score.py`, `confusion.py`, `roc_auc.py`

---

### Componente 3: Pipelines (sustituye pipeline de streaming)
#### [DELETE] `src/pipelines/evaluation_pipeline.py`
#### [NEW] `src/pipelines/evaluation_pipeline.py` — Pipeline de evaluación para clasificación
#### [NEW] `src/pipelines/training_pipeline.py` — Pipeline de entrenamiento unificado
#### [NEW] `src/pipelines/image_pipeline.py` — Pipeline de análisis de imagen

---

### Componente 4: Schemas (sustituye schemas de streaming)
#### [DELETE] `src/schemas/schemas.py`
#### [NEW] `src/schemas/patient_schemas.py`
#### [NEW] `src/schemas/prediction_schemas.py`
#### [NEW] `src/schemas/image_schemas.py`

---

### Componente 5: Tracking (llenar carpeta vacía)
#### [NEW] `src/tracking/experiment_tracker.py`
#### [NEW] `src/tracking/prediction_logger.py`

---

### Componente 6: Consolidación del Frontend
#### [MODIFY] `src/frontend/app_s.py` — Fusionar capacidades de `app_s_busqueda.py`
#### [DELETE] `src/frontend/app_s_busqueda.py` — Funcionalidad absorbida

---

### Componente 7: Limpieza general
#### [DELETE] Archivos legacy listados en sección 4.1
#### [MODIFY] `main.py` — Añadir comando `api` y `frontend`
#### [NEW] `src/config/settings.py` — Configuración centralizada
#### [MOVE] `ml_v0.py`, `ml_v1.py`, `ml_v2.py` → `src/models/ml/_legacy/`
#### [MODIFY] `pyproject.toml` — Añadir dependencias FastAPI, pytest, etc.

---

## Open Questions

> [!IMPORTANT]
> **Pregunta 1**: ¿Quieres que el frontend Streamlit llame al backend FastAPI (arquitectura desacoplada) o prefieres que siga cargando los modelos directamente (monolito)? La opción desacoplada es más profesional pero requiere arrancar dos procesos.

> [!IMPORTANT]
> **Pregunta 2**: ¿El `app_s_busqueda.py` tiene funcionalidad que quieres conservar aparte del `app_s.py`, o puedo fusionar todo en uno solo?

> [!IMPORTANT]
> **Pregunta 3**: ¿Quieres eliminar las versiones legacy de ML (`ml_v0`, `ml_v1`, `ml_v2`) completamente o prefieres moverlas a una carpeta `_legacy/` para mantener el historial?

> [!IMPORTANT]
> **Pregunta 4**: ¿Tenéis el modelo `lgbm_clinico.pkl` generado correctamente desde `ml_v3.py`? Si no, necesitamos ejecutar ese entrenamiento primero porque el frontend `app_s.py` lo necesita.

> [!IMPORTANT]
> **Pregunta 5**: Los archivos R (`.R`, `.RData`, `.Rhistory`) en el repo — ¿son de otro compañero del equipo que trabaja en R, o son artefactos antiguos que se pueden eliminar?

---

## Verification Plan

### Automated Tests
1. `pytest src/test/` — Todos los tests unitarios pasan
2. `uvicorn src.api.main_api:app` — El API arranca sin error
3. `curl POST /api/v1/predict/risk` — Devuelve predicción válida
4. `curl POST /api/v1/analyze/colonoscopy` — Devuelve análisis de imagen
5. `python -c "from src.metrics.protocols import ClassificationMetricProtocol; print('OK')"` — Imports limpios

### Manual Verification
1. `streamlit run src/frontend/app_s.py` — La app funciona completa
2. Verificar que la predicción ML + SHAP funciona end-to-end
3. Verificar análisis de colonoscopia con Grad-CAM
4. Verificar análisis de biopsias con Grad-CAM
