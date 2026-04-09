#Este es las segunda pruebas y modelos

import pandas as pd
import numpy as np
import os
import joblib
import warnings

# --- SILENCIAR WARNINGS MOLESTOS ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, fbeta_score, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)

# 1. Configuración de rutas
file_path = 'C:/Users/Ana-L/Desktop/cosas de Juan/Programacion/cancer de colon/prueba'
file_modelos = file_path + '/modelos/ml'
if not os.path.exists(file_modelos):
    os.makedirs(file_modelos)

# 2. Cargar datos
print("Cargando 146k registros...")
df = pd.read_csv(file_path + '/datos_finales_Kaggle.csv')

features = [
    'Age', 'Gender', 'Family_History', 'Smoking_History', 
    'Alcohol_Consumption', 'Obesity_BMI', 'Diet_Risk', 
    'Physical_Activity', 'Diabetes', 'Inflammatory_Bowel_Disease',
    'FOBT_Resultado_n', 'CEA_Level_ng_mL..Marcador.Tumoral.'
]
# features = [
#     'Age', 'Gender', 
#     # 'Cancer_Stage', 'Tumor_Size_mm', 'Genetic_Mutation',
#     'Family_History', 'Smoking_History',
#     'Dieta_rica_en_grasas_animales',
#     'Alcohol_Consumption', 'Obesity_BMI', 'Diet_Risk', 
#     'Physical_Activity', 'Diabetes', 'Inflammatory_Bowel_Disease',
#     'FOBT_Resultado_n', 'CEA_Level_ng_mL..Marcador.Tumoral.',
#     'Sedentarismo', 'Componente_Hereditario',
#     'Sindromes_Predisponentes',
#     'Enfermedad_Inflamatoria_Intestinal', 
#     'altura_cm', 'peso_kg'
# ]
target = 'Survival_Prediction'

X = df[features].values 
y = df[target].values

# 1. DIVIDIR PRIMERO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. APLICAR SMOTE SOLO AL TRAIN
print("⚖️ Aplicando SMOTE solo al entrenamiento...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # 3. Balanceo de datos con SMOTE
# print("Aplicando SMOTE para balancear clases...")
# smote = SMOTE(random_state=42)
# X_res, y_res = smote.fit_resample(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 4. Configuración de Validación Cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- MODELO 1: RANDOM FOREST + Tuning ---
print("Optimizando Random Forest...")
param_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), 
                               param_rf, cv=cv, n_iter=5, scoring='average_precision', n_jobs=-1)
rf_search.fit(X_train_res, y_train_res)
rf_best = rf_search.best_estimator_
joblib.dump(rf_best, file_modelos + '/best_rf_model.pkl')

# --- MODELO 2: XGBOOST + Tuning ---
print("Optimizando XGBoost...")
param_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    'scale_pos_weight': [10, 20]
}
# xgb_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 
#                                 param_xgb, cv=cv, n_iter=5, scoring='recall', n_jobs=-1)
# xgb_search.fit(X_train, y_train)
# xgb_best = xgb_search.best_estimator_
# joblib.dump(xgb_best, file_modelos + '/best_xgb_model.pkl')
xgb_gpu = XGBClassifier(
    tree_method='hist', 
    device='cuda', 
    eval_metric='logloss',
    random_state=42
)
xgb_search = RandomizedSearchCV(xgb_gpu, param_xgb, cv=cv, n_iter=5, scoring='average_precision')
xgb_search.fit(X_train_res, y_train_res)
xgb_best = xgb_search.best_estimator_
joblib.dump(xgb_best, file_modelos + '/best_xgb_model.pkl')

# --- MODELO 3: LIGHTGBM + Tuning ---
print("Optimizando LightGBM...")
param_lgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [31, 50, 70],
    'boosting_type': ['gbdt', 'dart']
}
# lgb_search = RandomizedSearchCV(LGBMClassifier(verbosity=-1), device='gpu',
#                                 param_lgb, cv=cv, n_iter=5, scoring='recall', n_jobs=-1)
# lgb_search.fit(X_train, y_train)
# lgb_best = lgb_search.best_estimator_
# joblib.dump(lgb_best, file_modelos + '/best_lgbm_model.pkl')

# print("\nTodos los modelos optimizados y guardados en:", file_modelos)
# Configuramos device='gpu'
lgb_gpu = LGBMClassifier(device='gpu', verbosity=-1, random_state=42)
lgb_search = RandomizedSearchCV(lgb_gpu, param_lgb, cv=cv, n_iter=5, scoring='average_precision')
lgb_search.fit(X_train_res, y_train_res)
lgb_best = lgb_search.best_estimator_
joblib.dump(lgb_best, file_modelos + '/best_lgbm_model.pkl')

print("\n✅ ¡Entrenamiento en GPU/CPU finalizado!")

# --- FUNCIÓN DE EVALUACIÓN ---
def evaluar_tabla(model, name, X_t, y_t):
    preds = model.predict(X_t)
    acc = accuracy_score(y_t, preds)
    prec = precision_score(y_t, preds)
    rec = recall_score(y_t, preds)
    f1 = f1_score(y_t, preds)
    f2 = fbeta_score(y_t, preds, beta=2)
    tn, fp, fn, tp = confusion_matrix(y_t, preds).ravel()
    
    print(f"\n" + "="*60)
    print(f"MÉTRICAS FINALES (Tras Tuning & SMOTE): {name}")
    print("="*60)
    print(f"{'Métrica':<15} | {'Valor':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<15} | {acc:.4f}")
    print(f"{'Precision':<15} | {prec:.4f}")
    print(f"{'Recall':<15} | {rec:.4f}")
    print(f"{'F1-Score':<15} | {f1:.4f}")
    print(f"{'F2-Score':<15} | {f2:.4f}")
    print("\nMATRIZ DE CONFUSIÓN:")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print("="*60)

# Evaluar los mejores modelos
evaluar_tabla(rf_best, "Random Forest Optimizado", X_test, y_test)
evaluar_tabla(xgb_best, "XGBoost Optimizado", X_test, y_test)
evaluar_tabla(lgb_best, "LightGBM Optimizado", X_test, y_test)