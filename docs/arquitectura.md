# Arquitectura del Proyecto Cancer Colon

## Visión General

Cancer de Colon es un sistema de apoyo al diagnóstico de cáncer de colon que combina:
- **Machine Learning** (LightGBM) para evaluar riesgo clínico a partir de 11 factores
- **Deep Learning** (CNN TensorFlow + ResNet18 PyTorch) para análisis de imágenes médicas
- **Grad-CAM** para explicar visualmente qué zonas de la imagen mira el modelo

## Estructura del Proyecto

```
Proyecto 2 - Cancer colon/
├── main.py                        # Punto de entrada principal
├── pyproject.toml                 # Dependencias del proyecto
│
├── src/
│   ├── api/                       # Backend HTTP (FastAPI)
│   │   ├── main_api.py            # Endpoints de la API
│   │   └── dependencies.py        # Carga de modelos de IA
│   │
│   ├── frontend/                  # Frontend visual (Streamlit)
│   │   ├── app.py                 # App unificada (la principal)
│   │   ├── app_s.py               # (Legacy) Frontend original
│   │   └── app_s_busqueda.py      # (Legacy) Frontend con búsqueda
│   │
│   ├── config/                    # Configuración centralizada
│   │   └── settings.py            # Todas las rutas y constantes
│   │
│   ├── models/                    # Modelos ML (scikit-learn, LightGBM)
│   │   └── ml/
│   │       ├── ml_v3.py           # Entrenamiento del modelo activo
│   │       ├── lgbm_clinico.pkl   # Modelo entrenado (binario)
│   │       └── anteriores/        # Versiones antiguas
│   │
│   ├── networks/                  # Modelos DL (TensorFlow, PyTorch)
│   │   ├── dl/                    # CNN para colonoscopia
│   │   └── dl_biopsia/            # ResNet18 para biopsias
│   │
│   ├── metrics/                   # Métricas de evaluación clínica
│   │   ├── protocols.py           # Interfaz/contrato que todas cumplen
│   │   ├── accuracy.py            # Exactitud global
│   │   ├── recall.py              # Sensibilidad (la más crítica)
│   │   └── ...                    # precision, f_score, confusion, roc_auc
│   │
│   ├── pipelines/                 # Flujos automatizados
│   │   ├── evaluation_pipeline.py # Evaluar modelos con todas las métricas
│   │   ├── training_pipeline.py   # Entrenamiento unificado (CSV → modelo)
│   │   └── image_pipeline.py      # Inferencia de imágenes desacoplada
│   │
│   ├── schemas/                   # Validación de datos (Pydantic)
│   │   ├── patient_schemas.py     # Formularios de pacientes
│   │   ├── prediction_schemas.py  # Entrada/salida de predicciones
│   │   └── image_schemas.py       # Respuestas de análisis de imagen
│   │
│   ├── tracking/                  # Registro de experimentos y predicciones
│   │   ├── experiment_tracker.py  # Historial de entrenamientos (JSON)
│   │   └── prediction_logger.py   # Log de predicciones en producción (CSV)
│   │
│   ├── test/                      # Tests reales con pytest
│   │   ├── test_metrics.py        # Tests de las métricas clínicas
│   │   ├── test_schemas.py        # Tests de validación Pydantic
│   │   └── test_api.py            # Tests de integración de la API
│   │
│   ├── utils/                     # Utilidades compartidas
│   │   ├── cargar_modelos_s.py    # Carga de modelos para Streamlit
│   │   ├── gradcam_utils.py       # Generación de mapas de calor
│   │   └── eda_visualization.py   # Visualizaciones EDA
│   │
│   └── data/                      # Datos del proyecto
│       ├── raw/                   # Datos originales sin modificar
│       └── clean/                 # Datos limpios/procesados
│
├── docs/                          # Documentación
│   ├── arquitectura.md            # Este archivo
│   ├── modelos.md                 # Documentación de los modelos
│   └── api_endpoints.md           # Documentación de la API
│
└── info/                          # PDFs del enunciado y rúbricas
```

## Flujo de Datos

```
Paciente → Streamlit (app.py) → API (main_api.py) → Modelo de IA → Resultado
                                                         ↓
                                            prediction_logger.py (CSV)
```

## Dos Mundos Separados

1. **Producción**: API + Streamlit + image_pipeline → predicciones en tiempo real
2. **Entrenamiento**: training_pipeline + evaluation_pipeline + métricas → entrenar modelos nuevos
