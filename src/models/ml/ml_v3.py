import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN ---
RUTA_DATA = r"src\data\clean\cancer_risk_final.csv"
SAVE_DIR = r"artifacts\weights"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. CARGA Y SELECCIÓN (ACTUALIZADO) ---
df = pd.read_csv(RUTA_DATA)

# IMPORTANTE: Hemos añadido las variables clínicas que pediste
# NOTA: FOBT_Resultado_n y CEA_Level_ng_mL no existen en cancer_risk_clean.csv
# Si necesitas esas features, usa datos_combinados_global_extendido_3.csv
features = [
    "Smoking",
    "Alcohol_Use",
    "Obesity",
    "Family_History",
    "Diet_Red_Meat",
    "Diet_Salted_Processed",
    "Fruit_Veg_Intake",
    "Physical_Activity",
    "BMI",
    "FOBT_Resultado_n",
    "CEA_Level_ng_mL"
]
target = "Risk_Level_n"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. ENTRENAMIENTO ---
print("Entrenando Random Forest Clínico...")
rf_model = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42
)
rf_model.fit(X_train, y_train)

print("Entrenando XGBoost Clínico...")
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.02,
    objective="multi:softprob",
    num_class=3,
    random_state=42,
)
xgb_model.fit(X_train, y_train)

print("Entrenando LightGBM Clínico...")
lgbm_model = LGBMClassifier(
    n_estimators=300, learning_rate=0.02, verbosity=-1, random_state=42
)
lgbm_model.fit(X_train, y_train)


# --- 4. FUNCIÓN DE EVALUACIÓN ---
def evaluar_sistema(y_real, y_pred, nombre):
    acc = accuracy_score(y_real, y_pred)
    rec_high = recall_score(
        y_real, y_pred, labels=[2], average="macro"
    )  # Recall específico de clase High

    print(f"\nANÁLISIS: {nombre}")
    print("-" * 45)
    print(f"Accuracy General: {acc:.4f}")
    print(f"RECALL CLASE ALTA: {rec_high:.4f} <-- Objetivo")
    print("\nInforme detallado:")
    print(classification_report(y_real, y_pred, target_names=["Low", "Medium", "High"]))

    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGn_r",
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
    )
    plt.title(f"Matriz de Confusión: {nombre}")
    plt.ylabel("Realidad")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig(f"artifacts/confusion_matrix_{nombre.replace(' ', '_')}.png")


# --- 5. LÓGICA DE ENSAMBLE CLÍNICO (Súper Sensible) ---
def obtener_preds_ensamble(modelos, X_input, umbral_alto=0.15):
    """
    Bajamos el umbral a 0.15. Si hay marcadores clínicos positivos,
    queremos que el modelo salte de inmediato a Alerta Alta.
    """
    probs_rf = modelos[0].predict_proba(X_input)
    probs_xgb = modelos[1].predict_proba(X_input)
    probs_lgbm = modelos[2].predict_proba(X_input)

    avg_probs = (probs_rf + probs_xgb + probs_lgbm) / 3
    preds = []
    for p in avg_probs:
        if p[2] >= umbral_alto:
            preds.append(2)
        else:
            preds.append(np.argmax(p))
    return np.array(preds)


# --- 6. EJECUCIÓN ---
evaluar_sistema(y_test, rf_model.predict(X_test), "Random Forest")
evaluar_sistema(y_test, xgb_model.predict(X_test), "XGBoost")
evaluar_sistema(y_test, lgbm_model.predict(X_test), "LightGBM")

print("\nCALCULANDO ENSAMBLE CON MARCADORES CLÍNICOS...")
y_pred_ensamble = obtener_preds_ensamble([rf_model, xgb_model, lgbm_model], X_test)
evaluar_sistema(y_test, y_pred_ensamble, "Ensamble Final (Máxima Seguridad)")

# --- 7. GUARDADO ---
# EN LUGAR DE GUARDAR LAS PREDICCIONES, GUARDA UNA LISTA CON LOS 3 MODELOS
modelos_lista = [rf_model, xgb_model, lgbm_model]
joblib.dump(modelos_lista, os.path.join(SAVE_DIR, "modelo_ensemble.pkl"))

# Además guardamos el lgbm_clinico.pkl individualmente para la API
joblib.dump(lgbm_model, os.path.join(SAVE_DIR, "lgbm_clinico.pkl"))

os.makedirs(r"artifacts\mappings", exist_ok=True)
joblib.dump(features, os.path.join(r"artifacts\mappings", "features_list.pkl"))

print(f"\nProceso finalizado. Modelos con variables clínicas guardados.")