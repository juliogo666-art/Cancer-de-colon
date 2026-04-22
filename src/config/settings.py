"""
=============================================================================
Configuración centralizada
=============================================================================
ESTE ARCHIVO ES EL "MANDO A DISTANCIA" DEL PROYECTO.

Aquí se definen TODAS las rutas a archivos, modelos y datos del proyecto.
En lugar de tener rutas escritas a mano (hardcodeadas) desperdigadas por
todo el código, cualquier programa que necesite una ruta la importa de aquí.

Ventajas:
    - Si mueves un archivo, solo cambias la ruta aquí y todo sigue funcionando.
    - No hay rutas absolutas de Windows (como C:\\Users\\...) en el código.
    - Se pueden cambiar rutas mediante variables de entorno (.env) sin tocar código.

Uso en cualquier archivo del proyecto:
    from src.config.settings import settings

    # Acceder a una ruta:
    ruta_csv = settings.CSV_RISK_PATH
    ruta_modelo = settings.MODEL_ML_PATH
=============================================================================
"""

import os

###############################################################################
# PASO 1: Calcular la raíz del proyecto automáticamente
###############################################################################
# Este archivo está en: src/config/settings.py
# La raíz del proyecto está 2 niveles arriba: ../../
# Así que subimos dos carpetas para llegar a la raíz del proyecto
RAIZ_DEL_PROYECTO = os.path.dirname(  # -> src/
    os.path.dirname(  # -> Proyecto 2 - Cancer colon/
        os.path.dirname(os.path.abspath(__file__))  # -> src/config/
    )
)


class Settings:
    """
    Clase que contiene todas las rutas y configuraciones del proyecto.

    Cada ruta se construye de forma relativa desde la raíz del proyecto,
    así funciona en cualquier ordenador (Windows, Mac, Linux) sin cambiar nada.
    """

    def __init__(self):
        ##########################################################################
        # Rutas a datos de pacientes (CSV)
        ##########################################################################

        # Carpeta donde están los historiales de pacientes (datos crudos/originales)
        self.CARPETA_HISTORIALES = os.path.join(
            RAIZ_DEL_PROYECTO, "src", "data", "raw", "historial_pacientes"
        )

        # Carpeta para datos limpios/procesados (resultados de limpieza)
        self.CARPETA_DATOS_LIMPIOS = os.path.join(
            RAIZ_DEL_PROYECTO, "src", "data", "clean"
        )

        # CSV Maestro con TODA la información consolidada (pacientes + historial clínico)
        self.CSV_MASTER_PATH = os.path.join(
            RAIZ_DEL_PROYECTO, "src", "data", "ready", "pacientes_master.csv"
        )

        # CSV de factores de riesgo limpio para entrenamiento del modelo ML
        # Contiene las 11 features + Risk_Level_n como target
        self.CSV_RISK_PATH = os.path.join(
            RAIZ_DEL_PROYECTO, "src", "data", "clean", "cancer_risk_final.csv"
        )

        ##########################################################################
        # Rutas a modelos de IA
        ##########################################################################

        # Modelo de Machine Learning (LightGBM) para predecir riesgo clínico
        # Entrada: 11 factores de riesgo → Salida: Low / Medium / High
        self.MODEL_ML_PATH = os.path.join(
            RAIZ_DEL_PROYECTO, "artifacts", "weights", "lgbm_clinico.pkl"
        )

        self.MODEL_ML_FINAL_PATH = os.path.join(
            RAIZ_DEL_PROYECTO, "artifacts", "weights", "modelo_ensemble.pkl"
        )

        # Modelo de Machine Learning (LightGBM) para triaje rápido (sin analíticas)
        # Entrada: 9 factores de riesgo (sin FOBT ni CEA) → Salida: Low / Medium / High
        self.MODEL_ML_TRIAGE_PATH = os.path.join(
            RAIZ_DEL_PROYECTO, "artifacts", "weights", "lgbm_triage.pkl"
        )

        # Modelo de Deep Learning (CNN TensorFlow) para detectar pólipos en colonoscopia
        # Entrada: Imagen 150x150 → Salida: Pólipo / Sano
        self.MODEL_CNN_PATH = os.path.join(
            RAIZ_DEL_PROYECTO,
            "artifacts",
            "weights",
            "mejor_modelo_anti_overfit.pth",
        )

        # Modelo de Deep Learning (DenseNet121 PyTorch) para clasificar biopsias
        # Entrada: Imagen 224x224 → Salida: Benigno / Maligno
        self.MODEL_BIOPSY_PATH = os.path.join(
            RAIZ_DEL_PROYECTO,
            "artifacts",
            "checkpoints",
            "biopsia_densenet121_best.pth",
        )

        ##########################################################################
        # Rutas a datos de imagen
        ##########################################################################

        # Carpeta con imágenes de colonoscopia procesadas
        self.CARPETA_IMAGENES_COLONOSCOPIA = os.path.join(
            RAIZ_DEL_PROYECTO, "src", "data", "raw", "Colonoscopic_processed"
        )

        # Carpeta con conjuntos de imágenes de colon
        self.CARPETA_IMAGENES_COLON = os.path.join(
            RAIZ_DEL_PROYECTO, "src", "data", "raw", "colon_image_sets"
        )

        # Carpeta con imágenes decorativas del frontend
        self.CARPETA_IMAGENES_UI = os.path.join(RAIZ_DEL_PROYECTO, "static")

        ##########################################################################
        # Rutas de tracking (registros de experimentos y predicciones)
        ##########################################################################

        # Archivo JSON donde se guardan los experimentos de entrenamiento
        self.TRACKER_EXPERIMENTS_PATH = os.path.join(
            RAIZ_DEL_PROYECTO, "src", "tracking", "experiments.json"
        )

        # Archivo CSV donde se registran las predicciones realizadas
        self.TRACKER_PREDICTIONS_PATH = os.path.join(
            RAIZ_DEL_PROYECTO, "src", "tracking", "predictions.csv"
        )

        ##########################################################################
        # Las 11 features que espera el modelo ML (orden exacto)
        ##########################################################################
        # IMPORTANTE: Si cambias el orden, el modelo dará resultados incorrectos.
        # Este es el mismo orden que se usó durante el entrenamiento en ml_v3.py
        self.ML_FEATURE_NAMES = [
            "Smoking",  # Nivel de tabaquismo (0-10)
            "Alcohol_Use",  # Consumo de alcohol (0-10)
            "Obesity",  # Nivel de obesidad (0-10)
            "Family_History",  # ¿Tiene antecedentes familiares? (0 = No, 1 = Sí)
            "Diet_Red_Meat",  # Consumo de carne roja (0-10)
            "Diet_Salted_Processed",  # Consumo de alimentos salados/procesados (0-10)
            "Fruit_Veg_Intake",  # Consumo de frutas y verduras (0-10)
            "Physical_Activity",  # Nivel de actividad física (0-10)
            "BMI",  # Índice de Masa Corporal (10-60 kg/m²)
            "FOBT_Resultado_n",  # Sangre oculta en heces (0 = Negativo, 1 = Positivo)
            "CEA_Level_ng_mL",  # Marcador tumoral CEA en nanogramos por mililitro
        ]

        ##########################################################################
        # Mapeo de clases del modelo ML
        ##########################################################################
        # El modelo devuelve un número (0, 1, 2) y esto lo traduce a texto
        self.RISK_LEVEL_MAP = {
            0: "Low",  # Riesgo bajo
            1: "Medium",  # Riesgo medio
            2: "High",  # Riesgo alto
        }

        ##########################################################################
        # Configuración de la API
        ##########################################################################
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))


##################################################################################
# Instancia global — Importar así: from src.config.settings import settings
##################################################################################
settings = Settings()
