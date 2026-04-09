#Este es las primera pruebas y modelos

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, fbeta_score, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib

# 1. Cargar datos
print("⏳ Cargando registros...")
file_path = 'C:/Users/Ana-L/Desktop/cosas de Juan/Programacion/cancer de colon/prueba'
file_modelos = file_path + '/modelos/ml'
df = pd.read_csv(file_path + '/datos_finales_Kaggle.csv')

# features = [
#     'Age', 'Gender', 'Cancer_Stage', 'Tumor_Size_mm', 'Family_History', 
#     'Smoking_History', 'Genetic_Mutation', 'Dieta_rica_en_grasas_animales',
#     'Alcohol_Consumption', 'Obesity_BMI', 'Diet_Risk', 
#     'Physical_Activity', 'Diabetes', 'Inflammatory_Bowel_Disease',
#     'FOBT_Resultado_n', 'CEA_Level_ng_mL..Marcador.Tumoral.',
#     'Sedentarismo', 'Diabetes_tipo_2', 'Componente_Hereditario',
#     'Antecedentes_Familiares', 'Componente_Hereditario', 'Sindromes_Predisponentes',
#     'Enfermedad_Inflamatoria_Intestinal', 
#     'altura_cm', 'peso_kg'
# ]

features = [
    'Age', 'Gender', 'Family_History', 'Smoking_History', 
    'Alcohol_Consumption', 'Obesity_BMI', 'Diet_Risk', 
    'Physical_Activity', 'Diabetes', 'Inflammatory_Bowel_Disease',
    'FOBT_Resultado_n', 'CEA_Level_ng_mL..Marcador.Tumoral.'
]

target = 'Survival_Prediction'

X = df[features].values 
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ENTRENAMIENTO RÁPIDO ---
print("🌲 Entrenando Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)

print("🚀 Entrenando XGBoost...")
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6).fit(X_train, y_train)

print("⚡ Entrenando LightGBM...")
lgb_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, verbosity=-1).fit(X_train, y_train)

# --- FUNCIÓN DE EVALUACIÓN TABULAR ---
def evaluar_tabla(model, name):
    preds = model.predict(X_test)
    
    # Cálculo de métricas
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    f2 = fbeta_score(y_test, preds, beta=2)
    
    # Matriz de confusión desglosada
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    print(f"\n" + "="*60)
    print(f"📊 MÉTRICAS DE RENDIMIENTO: {name}")
    print("="*60)
    
    # Mostrar métricas principales
    print(f"{'Métrica':<15} | {'Valor':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<15} | {acc:.4f}")
    print(f"{'Precision':<15} | {prec:.4f}")
    print(f"{'Recall':<15} | {rec:.4f}")
    print(f"{'F1-Score':<15} | {f1:.4f}")
    print(f"{'F2-Score':<15} | {f2:.4f}")
    
    print("\n" + "-"*30)
    print(f"📌 DESGLOSE DE LA MATRIZ (Valores Reales)")
    print("-" * 30)
    print(f"True Positives (TP):  {tp:>8}")
    print(f"True Negatives (TN):  {tn:>8}")
    print(f"False Positives (FP): {fp:>8}")
    print(f"False Negativos (FN): {fn:>8}  <-- ¡Vigilar este!")
    print("="*60)

# Ejecutar tablas
evaluar_tabla(rf_model, "Random Forest")
evaluar_tabla(xgb_model, "XGBoost")
evaluar_tabla(lgb_model, "LightGBM")

# Guardar el que consideres mejor (normalmente el de mayor F2 o menor FN)
joblib.dump(xgb_model, file_modelos + '/modelo_historial_final.pkl')
print("\n✅ Evaluación terminada. Archivo 'modelo_historial_final.pkl' generado.")