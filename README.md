# Cáncer de Colón — Sistema de Análisis y Predicción

El clinico necesita desarrollar un programa que al entregarle los datos del usuario analiza la información y predice si tienes o no cancer de colon.

Hasta ahora !!!
Sistema de análisis y predicción de cáncer de colón que combina análisis exploratorio de datos (EDA), modelos de machine learning (Random Forest) y deep learning (ResNet18) para la detección de pólipos en imágenes de colonoscopia.

## Requisitos

- Python >= 3.11
- GPU (opcional, recomendada para entrenamiento del modelo de pólipos)
- Token de HuggingFace (para descargar el dataset de colonoscopia)

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/juliogo666-art/Cancer-de-colon.git
cd Cancer-de-colon

# Instalar dependencias con uv
uv sync

# O con pip
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.sample .env
# Editar .env con tu token de HuggingFace
```

## Uso

### App de Análisis Exploratorio (EDA)

```bash
streamlit run main.py
```

La app incluye 5 secciones:

1. **Datos globales** — Análisis de dataset de cáncer global
2. **Datos simulados** — Análisis de pacientes sintéticos generados
3. **Datos combinados** — Análisis de datos globales + sintéticos
4. **Datos Kaggle** — Análisis del dataset colorectal de Kaggle
5. **Datos finales** — Análisis de Kaggle + datos sintéticos añadidos

### Generar datos sintéticos

```bash
python main.py generate-data
```

### Entrenar modelos

```bash
# Modelo de detección de pólipos (ResNet18)
python main.py train-polyps

# Modelo Random Forest (datos de pacientes)
python main.py train-rf
```

## Estructura del Proyecto

```
├── main.py                     # Punto de entrada principal
├── src/
│   ├── config/                 # Configuración
│   ├── data/
│   │   ├── raw/                # Datos originales (no subir a Git)
│   │   ├── clean/              # Datos limpios
│   │   └── api_call_img.py     # Descarga de datasets de imágenes
│   ├── models/
│   │   ├── modelo_busca_polipos_Clas.py    # ResNet18 clasificador de pólipos
│   │   ├── modelo_busca_polipos_Segment.py # Segmentación (en desarrollo)
│   │   └── testeos.py                      # Random Forest para diagnóstico
│   ├── scripts/
│   │   ├── eda.py                    # App Streamlit de EDA
│   │   ├── data_cleaning.py          # Limpieza de datos
│   │   └── sintetiza_historiales.py  # Generación de datos sintéticos
│   ├── utils/
│   │   └── eda_visualization.py      # Funciones de visualización
│   └── networks/                     # Red neuronal (futuro)
├── .env                        # Variables de entorno (no subir a Git)
├── pyproject.toml              # Configuración del proyecto
└── requirements.txt            # Dependencias
```

## Qué tenemos

1. **Datos del paciente**
   - Historial médico (analítica general, electrocardiograma, tensión, etc.)
   - Video/imágenes de colonoscopia

2. **Pruebas y resultados de diagnóstico**
   - Imágenes de pólipos cancerígenos y sin
   - Historial médico ficticio del paciente: datos vitales e información hereditaria
   - Resultados de test de diagnóstico general y tumoral

## Flujo del sistema

1. **INPUTS**: Datos del paciente + Imágenes/video colonoscopia
2. **Base de datos**: Consulta datos de pólipos y relacionados
3. **Vectorización** de la información
4. **Inferencia** del modelo
