import pandas as pd
import numpy as np
from pathlib import Path

# Obtener rutas absolutas basadas en la ubicación de este script
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

# Rutas de entrada y salida
data_dir = (
    project_root
    / "src"
    / "data"
    / "raw"
    / "historial_pacientes"
    / "historiales_factor_riesgo"
)
input_path = data_dir / "cancer-risk-factors.csv"
output_path = data_dir / "cancer_risk_factors_augmented.csv"

# 1. Cargar datos
print(f"Cargando datos de: {input_path}")
df_old = pd.read_csv(input_path)
df_colon = df_old[df_old["Cancer_Type"] == "Colon"].copy()
print(f"Numero de registros originales de colon: {len(df_colon)}")

# Variables que usaremos para predecir el riesgo
features = [
    "Age",
    "Gender",
    "Smoking",
    "Alcohol_Use",
    "Obesity",
    "Family_History",
    "Diet_Red_Meat",
    "Diet_Salted_Processed",
    "Fruit_Veg_Intake",
    "Physical_Activity",
    "BMI",
    "Overall_Risk_Score",
    "Risk_Level",
]
df_colon = df_colon[features]

# 2. Generar datos sintéticos hasta 5000
n_synthetic = 5000 - len(df_colon)
synthetic_data = []
np.random.seed(42)

for i in range(n_synthetic):
    base_idx = np.random.randint(0, len(df_colon))
    base_row = df_colon.iloc[base_idx].copy()

    # Añadimos ruido estadístico para crear pacientes nuevos pero coherentes
    base_row["Age"] = max(20, min(90, int(base_row["Age"] + np.random.normal(0, 5))))
    base_row["Smoking"] = max(
        0, min(10, int(base_row["Smoking"] + np.random.normal(0, 1)))
    )
    base_row["Alcohol_Use"] = max(
        0, min(10, int(base_row["Alcohol_Use"] + np.random.normal(0, 1)))
    )
    base_row["Obesity"] = max(
        0, min(10, int(base_row["Obesity"] + np.random.normal(0, 1)))
    )
    base_row["Diet_Red_Meat"] = max(
        0, min(10, int(base_row["Diet_Red_Meat"] + np.random.normal(0, 1)))
    )
    base_row["Diet_Salted_Processed"] = max(
        0, min(10, int(base_row["Diet_Salted_Processed"] + np.random.normal(0, 1)))
    )
    base_row["Fruit_Veg_Intake"] = max(
        0, min(10, int(base_row["Fruit_Veg_Intake"] + np.random.normal(0, 1)))
    )
    base_row["Physical_Activity"] = max(
        0, min(10, int(base_row["Physical_Activity"] + np.random.normal(0, 1)))
    )
    base_row["BMI"] = max(
        18.0, min(45.0, round(base_row["BMI"] + np.random.normal(0, 2), 1))
    )

    # 3. Recalcular el Riesgo basado en literatura médica
    # Incorporamos los factores médicos agregados para que los sintéticos sigan a los reales
    # Ver archivo factores_de_riesgo.md

    score = (
        (base_row["Age"] / 100) * 0.20  # NCI: Factor principal (>50 años)
        + (base_row["Smoking"] / 10) * 0.15  # Surgeon General: Mutágenos directos
        + (base_row["Alcohol_Use"] / 10) * 0.10  # IARC/OMS: Factor carcinógeno
        + (base_row["Diet_Red_Meat"] / 10)
        * 0.17  # OMS (Monografía 114): +17% de riesgo
        + (base_row["Diet_Salted_Processed"] / 10)
        * 0.18  # OMS (Monografía 114): +18% de riesgo
        + (base_row["Obesity"] / 10) * 0.10  # WCRF: Inflamación metabólica
        + (base_row["Family_History"]) * 0.15  # ACG: Genética (Multiplica el riesgo)
        - (base_row["Fruit_Veg_Intake"] / 10)
        * 0.15  # EPIC: Efecto protector fuerte de la fibra
        - (base_row["Physical_Activity"] / 10)
        * 0.10  # WCRF: Motilidad intestinal acelerada
    )

    # Normalizamos para asegurarnos de que el score final NUNCA se pase de 1.0 ni baje de 0.0
    # Añadimos un 0.1 de "riesgo base poblacional"
    base_row["Overall_Risk_Score"] = max(0.0, min(1.0, score + 0.1))

    # 4. Asignar la etiqueta objetivo (Target) para tu modelo
    if base_row["Overall_Risk_Score"] > 0.65:
        base_row["Risk_Level"] = "High"
    elif base_row["Overall_Risk_Score"] > 0.45:
        base_row["Risk_Level"] = "Medium"
    else:
        base_row["Risk_Level"] = "Low"

    synthetic_data.append(base_row)

df_synthetic = pd.DataFrame(synthetic_data)
df_final = pd.concat([df_colon, df_synthetic], ignore_index=True)
df_final.to_csv(output_path, index=False)
print(f"\nGenerados {n_synthetic} datos sinteticos.")
print(
    f"Total datos guardados (originales + sinteticos = {len(df_final)}) exitosamente en: {output_path}"
)
