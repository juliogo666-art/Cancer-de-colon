import pandas as pd
import numpy as np
import os  # Importamos esto para manejar rutas

RUTA_DATA = r'C:\Users\juanp\OneDrive\Documentos\stucom\2 Trimestre\proyecto2\Cancer-de-colon\src\data\clean\cancer_risk_clean.csv'
SAVE_DIR = r'C:\Users\juanp\OneDrive\Documentos\stucom\2 Trimestre\proyecto2\Cancer-de-colon\src\data\clean'

# 1. Cargar el dataset
df = pd.read_csv(RUTA_DATA)

# 2. Eliminar columnas innecesarias
df = df.drop(columns=['Age', 'Gender'])

# 3. Función para determinar FOBT_Resultado
def determinar_fobt(riesgo):
    if riesgo == 'High':
        return 'Positivo'
    elif riesgo == 'Medium':
        return np.random.choice(['Positivo', 'Negativo'], p=[0.5, 0.5])
    else:
        return 'Negativo'

df['FOBT_Resultado'] = df['Risk_Level'].apply(determinar_fobt)

# 4. Crear columna binaria FOBT_Resultado_n
df['FOBT_Resultado_n'] = df['FOBT_Resultado'].map({'Positivo': 1, 'Negativo': 0})

# 5. Función para el Marcador Tumoral (CEA)
def calcular_cea(row):
    if row['Risk_Level'] in ['Medium', 'High']:
        return round(np.random.uniform(2.5, 10.0), 2)
    return 0.0

df['CEA_Level_ng_mL'] = df.apply(calcular_cea, axis=1)

# --- EL CAMBIO ESTÁ AQUÍ ---
# Creamos la ruta completa correctamente
ruta_salida = os.path.join(SAVE_DIR, 'cancer_risk_final.csv')

# Guardar el archivo
df.to_csv(ruta_salida, index=False)

print(f"¡Archivo guardado con éxito en: {ruta_salida}")