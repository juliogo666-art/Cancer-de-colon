"""
=============================================================================
Entrenamiento de Modelos Dual (Clínico y Triaje)
=============================================================================
Este script entrena dos modelos LightGBM en paralelo:
1. Modelo Clínico Completo (11 features): Incluye análisis de sangre (FOBT, CEA).
2. Modelo de Triaje (9 features): Basado puramente en estilo de vida y antecedentes.

Estos modelos se despliegan en la API de FastAPI. El frontend usará el 
modelo Clínico si hay analíticas disponibles, y el de Triaje si no las hay.
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from src.tracking.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()

warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN ---
# Usamos el master nuevo que está unificado y limpio, conteniendo tanto el
# historial de pacientes antiguos como los nuevos generados.
RUTA_DATA = r"src\data\ready\pacientes_master.csv"
SAVE_DIR = r"artifacts\weights"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Cargando dataset maestro para entrenamiento ML Dual...")
df = pd.read_csv(RUTA_DATA)

# --- 2. MODELO CLÍNICO (COMPLETO - 11 FEATURES) ---
# Este modelo es el "gold standard" diagnostico de riesgo. Utiliza 
# hábitos, antecedentes y analíticas (Sangre oculta en heces y marcador CEA).
print("\n--- FASE 1: MODELO CLÍNICO COMPLETO ---")
features_clinico = [
    "Smoking", "Alcohol_Use", "Obesity", "Family_History", "Diet_Red_Meat", 
    "Diet_Salted_Processed", "Fruit_Veg_Intake", "Physical_Activity", "BMI", 
    "FOBT_Resultado_n", "CEA_Level_ng_mL"
]
target = "Risk_Level_n"

X_clin = df[features_clinico]
y_clin = df[target]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clin, y_clin, test_size=0.2, random_state=42, stratify=y_clin
)

lgbm_clinico = LGBMClassifier(n_estimators=300, learning_rate=0.02, verbosity=-1, random_state=42)
lgbm_clinico.fit(X_train_c, y_train_c)

y_pred_c = lgbm_clinico.predict(X_test_c)
acc_c = accuracy_score(y_test_c, y_pred_c)
rec_c = recall_score(y_test_c, y_pred_c, average='macro')
f1_c = f1_score(y_test_c, y_pred_c, average='macro')
print(f"Evaluación Clínico: Accuracy = {acc_c:.4f}, Recall = {rec_c:.4f}")

tracker.log_experiment(
    model_name="LightGBM_Clinico",
    metrics={"Accuracy": acc_c, "Recall_Macro": rec_c, "F1_Macro": f1_c},
    hyperparameters={"n_estimators": 300, "learning_rate": 0.02},
    features=features_clinico,
    dataset_path=RUTA_DATA,
    model_path=os.path.join(SAVE_DIR, "lgbm_clinico.pkl")
)

# --- 3. MODELO DE TRIAJE (ESTILO DE VIDA - 9 FEATURES) ---
# Este modelo solo utiliza los 9 factores base. 
# Su objetivo no es diagnosticar definitivamente, sino estratificar a las personas 
# en la sala de espera para priorizar quién necesita pruebas urgentes.
print("\n--- FASE 2: MODELO TRIAJE (SIN ANALÍTICAS) ---")
features_triage = [
    "Smoking", "Alcohol_Use", "Obesity", "Family_History", "Diet_Red_Meat", 
    "Diet_Salted_Processed", "Fruit_Veg_Intake", "Physical_Activity", "BMI"
]

X_tri = df[features_triage]
y_tri = df[target]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_tri, y_tri, test_size=0.2, random_state=42, stratify=y_tri
)

lgbm_triage = LGBMClassifier(n_estimators=300, learning_rate=0.02, verbosity=-1, random_state=42)
lgbm_triage.fit(X_train_t, y_train_t)

y_pred_t = lgbm_triage.predict(X_test_t)
acc_t = accuracy_score(y_test_t, y_pred_t)
rec_t = recall_score(y_test_t, y_pred_t, average='macro')
f1_t = f1_score(y_test_t, y_pred_t, average='macro')
print(f"Evaluación Triaje: Accuracy = {acc_t:.4f}, Recall = {rec_t:.4f}")

tracker.log_experiment(
    model_name="LightGBM_Triaje",
    metrics={"Accuracy": acc_t, "Recall_Macro": rec_t, "F1_Macro": f1_t},
    hyperparameters={"n_estimators": 300, "learning_rate": 0.02},
    features=features_triage,
    dataset_path=RUTA_DATA,
    model_path=os.path.join(SAVE_DIR, "lgbm_triage.pkl")
)

# --- 4. GUARDADO DE PESOS Y MAPPING ---
print("\n--- FASE 3: GUARDADO ---")
joblib.dump(lgbm_clinico, os.path.join(SAVE_DIR, "lgbm_clinico.pkl"))
joblib.dump(lgbm_triage, os.path.join(SAVE_DIR, "lgbm_triage.pkl"))

os.makedirs(r"artifacts\mappings", exist_ok=True)
joblib.dump(features_clinico, os.path.join(r"artifacts\mappings", "features_clinico.pkl"))
joblib.dump(features_triage, os.path.join(r"artifacts\mappings", "features_triage.pkl"))

print(f"Bases de IA exportadas con éxito. Modelos listos para servidor FastAPI.")