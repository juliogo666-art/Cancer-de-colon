#Este es el bueno/final

import pandas as pd
import numpy as np
import os
import joblib
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, fbeta_score

# --- 1. CONFIGURACIÓN DE RUTAS ---
directorio_actual = os.path.dirname(os.path.abspath(__file__))
# src/models/ml -> src/data/raw/historial_pacientes
directorio_src = os.path.dirname(os.path.dirname(directorio_actual))
file_path_data = os.path.join(directorio_src, 'data', 'raw', 'historial_pacientes')
file_modelos = directorio_actual
os.makedirs(file_modelos, exist_ok=True)

# --- 2. CARGA Y SELECCIÓN DE VARIABLES ---
print("⏳ Cargando datos...")
df = pd.read_csv(os.path.join(file_path_data, 'datos_finales_Kaggle.csv'))

# He incluido Cancer_Stage y Tumor_Size porque son las que dan capacidad predictiva real
features = [
    'Age', 'Gender', 'Cancer_Stage', 'Tumor_Size_mm', 'Family_History', 
    'Smoking_History', 'Alcohol_Consumption', 'Obesity_BMI', 
    'Inflammatory_Bowel_Disease', 'FOBT_Resultado_n', 
    'CEA_Level_ng_mL..Marcador.Tumoral.', 'altura_cm', 'peso_kg'
]
target = 'Survival_Prediction'

X = df[features].values 
y = df[target].values

# --- 3. DIVISIÓN DE DATOS (SIN SMOTE) ---
# Usamos stratify para mantener la proporción 60/40 en ambos sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 4. ENTRENAMIENTO DE MODELOS ---

# MODELO 1: RANDOM FOREST
print("🌲 Entrenando Random Forest (CPU)...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# MODELO 2: XGBOOST (GPU)
print("🚀 Entrenando XGBoost (GPU)...")
xgb_model = XGBClassifier(
    tree_method='hist',
    device='cuda',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)

# MODELO 3: LIGHTGBM (GPU)
print("⚡ Entrenando LightGBM (GPU)...")
lgbm_model = LGBMClassifier(
    device='gpu',
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    verbosity=-1,
    random_state=42
)
lgbm_model.fit(X_train, y_train)

# --- 5. FUNCIÓN DE EVALUACIÓN MÉDICA (AJUSTE DE UMBRAL) ---

def evaluar_modelo_sensible(model, X_t, y_t, name, umbral=0.25):
    """
    Evaluación enfocada en reducir Falsos Negativos (FN) 
    ajustando el umbral de probabilidad.
    """
    # Obtenemos la probabilidad de la clase 1
    probs = model.predict_proba(X_t)[:, 1]
    
    # Si la probabilidad es mayor al umbral, lo marcamos como Positivo (1)
    preds = (probs >= umbral).astype(int)
    
    acc = accuracy_score(y_t, preds)
    prec = precision_score(y_t, preds)
    rec = recall_score(y_t, preds)
    f2 = fbeta_score(y_t, preds, beta=2) # El F2 da más peso al Recall
    tn, fp, fn, tp = confusion_matrix(y_t, preds).ravel()
    
    print(f"\n" + "="*60)
    print(f"MÉTRICAS: {name} (Umbral de decisión: {umbral})")
    print("="*60)
    print(f"{'Accuracy':<15} | {acc:.4f}")
    print(f"{'Precision':<15} | {prec:.4f} (Cuántos de los avisados están enfermos)")
    print(f"{'RECALL':<15} | {rec:.4f} (Capacidad de detectar enfermos)")
    print(f"{'F2-Score':<15} | {f2:.4f}")
    print("-" * 30)
    print(f"CONFUSIÓN: TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"NOTA: Hay {fn} pacientes enfermos que el modelo NO detectó.")
    print("="*60)

# --- 6. RESULTADOS ---

evaluar_modelo_sensible(rf_model, X_test, y_test, "Random Forest")
# evaluar_modelo_sensible(xgb_model, X_test, y_test, "XGBoost GPU")
# Prueba con umbrales más altos para ver si el modelo "separa" a los pacientes
evaluar_modelo_sensible(xgb_model, X_test, y_test, "XGBoost GPU", umbral=0.50)
evaluar_modelo_sensible(xgb_model, X_test, y_test, "XGBoost GPU", umbral=0.70)
evaluar_modelo_sensible(lgbm_model, X_test, y_test, "LightGBM GPU")

# Guardar modelos
joblib.dump(rf_model, os.path.join(file_modelos, 'best_rf_model.pkl'))
joblib.dump(xgb_model, os.path.join(file_modelos, 'xgb_sensible.pkl'))
joblib.dump(lgbm_model, os.path.join(file_modelos, 'lgbm_sensible.pkl'))

print("\n✅ Proceso completado. Modelos guardados en:", file_modelos)