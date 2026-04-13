import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN ---
# Usamos el master nuevo que está unificado y limpio
RUTA_DATA = r"src\data\ready\pacientes_master.csv"
SAVE_DIR = r"artifacts\weights"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Cargando dataset maestro para entrenamiento ML Dual...")
df = pd.read_csv(RUTA_DATA)

# --- 2. MODELO CLÍNICO (COMPLETO - 11 FEATURES) ---
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

print("Evaluación Clínico: Precisión =", accuracy_score(y_test_c, lgbm_clinico.predict(X_test_c)))

# --- 3. MODELO DE TRIAJE (ESTILO DE VIDA - 9 FEATURES) ---
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

print("Evaluación Triaje: Precisión =", accuracy_score(y_test_t, lgbm_triage.predict(X_test_t)))

# --- 4. GUARDADO DE PESOS Y MAPPING ---
print("\n--- FASE 3: GUARDADO ---")
joblib.dump(lgbm_clinico, os.path.join(SAVE_DIR, "lgbm_clinico.pkl"))
joblib.dump(lgbm_triage, os.path.join(SAVE_DIR, "lgbm_triage.pkl"))

os.makedirs(r"artifacts\mappings", exist_ok=True)
joblib.dump(features_clinico, os.path.join(r"artifacts\mappings", "features_clinico.pkl"))
joblib.dump(features_triage, os.path.join(r"artifacts\mappings", "features_triage.pkl"))

print(f"Bases de IA exportadas con éxito. Modelos listos para servidor FastAPI.")