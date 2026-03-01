import pandas as pd
import os
import streamlit as st
import numpy as np

def limpiar_datos_globales(df, file_path_sin_d):
    df = df.copy()
    
    # 1. Eliminar columnas que realmente no sirven
    df = df.drop(columns=['Cancer_Type'], errors='ignore')
    df = df.dropna()

    # 2. CREAR NUEVAS COLUMNAS (Sin tocar las originales)
    
    # Género numérico -> Gender_n
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    df['Gender_n'] = df['Gender'].map(gender_map)

    # País numérico -> Country_n
    paises = df['Country_Region'].unique()
    country_map = {pais: i+1 for i, pais in enumerate(paises)}
    df['Country_n'] = df['Country_Region'].map(country_map)

    # Etapa numérica -> Stage_n
    stage_map = {
        'Stage 0': 0, 
        'Stage I': 1, 
        'Stage II': 2, 
        'Stage III': 3, 
        'Stage IV': 4
    }
    df['Stage_n'] = df['Cancer_Stage'].map(stage_map)

    # 3. Limpieza de tipos para compatibilidad con Streamlit (LargeUtf8 fix)
    # Aplicamos la limpieza a todas las columnas para que no rompa el visualizador
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Mantenemos el texto original pero limpiamos metadatos de Arrow
            df[col] = pd.Series(df[col].astype(str).tolist(), index=df.index)
        elif 'int' in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        elif 'float' in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)

    # 4. Guardado del CSV con las nuevas columnas incluidas
    directorio_destino = os.path.dirname(file_path_sin_d)
    output_path = os.path.join(directorio_destino, 'ej_global_cancer_limpio.csv')
    
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)
        
    df.to_csv(output_path, index=False)
    
    return df

def limpiar_datos_sinteticos(df, file_path_con_d):
    df = df.copy()
    
    # 1. Eliminar duplicados y nulos
    df = df.drop_duplicates()
    df = df.dropna()

    # 2. Eliminar IDs (No sirven para cálculos numéricos)
    if 'Paciente_ID' in df.columns:
        df = df.drop(columns=['Paciente_ID'])

    # 3. CREAR NUEVAS COLUMNAS NUMÉRICAS
    
    # Género (Asegúrate de que en el CSV sea 'Genero')
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    # Usamos .get para evitar errores si aparece un valor inesperado
    df['Gender_n'] = df['Genero'].map(gender_map)

    # FOBT Resultado
    result_fobt_map = {'Negative': 0, 'Positive': 1}
    # Nota: El nombre exacto en tu CSV es 'FOBT_Resultado (Sangre en heces)'
    col_fobt = 'FOBT_Resultado (Sangre en heces)'
    if col_fobt in df.columns:
        df['FOBT_Resultado_n'] = df[col_fobt].map(result_fobt_map)

    # 4. Asegurar que TODO sea numérico (para las columnas que ya eran 0 y 1)
    # Algunas columnas como 'Diagnostico' o 'Antecedentes' ya son 0/1, 
    # pero a veces Pandas las lee como objetos.
    for col in df.columns:
        if 'int' in str(df[col].dtype) or 'float' in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 5. Limpieza final de metadatos para Streamlit
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = pd.Series(df[col].astype(str).tolist(), index=df.index)

    # 6. Guardado
    directorio_destino = os.path.dirname(file_path_con_d)
    # Cambiamos el nombre para diferenciarlo de los datos globales
    output_path = os.path.join(directorio_destino, 'ej_pacientes_simulador_limpio.csv')
    
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)
        
    try:
        df.to_csv(output_path, index=False)
    except PermissionError:
        st.error(f"Error de permisos: Cierra el archivo {output_path} si está abierto.")
        
    return df

# def combinar_datos(df_global, df_simulado, output_dir):
#     """
#     Une el dataset Global y el Simulado, estandariza columnas y limpia datos.
#     """
#     # 1. ESTANDARIZACIÓN DE COLUMNAS (Mapeo de nombres)
#     # Convertimos nombres del simulado para que coincidan con el global donde sea posible
#     df_sim = df_simulado.copy()
#     df_glob = df_global.copy()

#     # Mapeo: {'Columna_Simulado': 'Columna_Objetivo'}
#     mapeo_columnas = {
#         'Edad': 'Age',
#         'Genero': 'Gender',
#         'Fumador': 'Smoking',
#         'Consume_Alcohol': 'Alcohol_Use',
#         'Obesidad': 'Obesity_Level',
#         'Componente_Hereditario': 'Genetic_Risk',
#         'Diagnostico': 'Target_Severity_Score' # Usamos el diagnóstico como score de severidad
#     }
#     df_sim = df_sim.rename(columns=mapeo_columnas)

#     # 2. CONCATENACIÓN (Unión de los dos)
#     # Al concatenar, las columnas que no coinciden se llenarán con NaN
#     df_combinado = pd.concat([df_glob, df_sim], axis=0, ignore_index=True)

#     # 3. LIMPIEZA DE DUPLICADOS Y NULOS
#     df_combinado = df_combinado.drop_duplicates()
    
#     # IDs no sirven para el análisis numérico
#     columnas_id = ['Patient_ID', 'Paciente_ID', 'Patient_ID']
#     df_combinado = df_combinado.drop(columns=[c for c in columnas_id if c in df_combinado.columns])

#     # 4. CONVERSIÓN NUMÉRICA Y MAPEOS
#     # Mapeo de Género
#     gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
#     if df_combinado['Gender'].dtype == 'object':
#         df_combinado['Gender'] = df_combinado['Gender'].map(gender_map)

#     # Mapeo de FOBT (Sangre en heces)
#     fobt_map = {'Negative': 0, 'Positive': 1}
#     col_fobt = 'FOBT_Resultado (Sangre en heces)'
#     if col_fobt in df_combinado.columns:
#         df_combinado['FOBT_n'] = df_combinado[col_fobt].map(fobt_map)

#     # 5. TRATAMIENTO DE NULOS RESTANTES (Imputación)
#     # Como al juntar se crean muchos NaN, rellenamos con 0 para que todo sea numérico
#     # Solo en columnas numéricas
#     num_cols = df_combinado.select_dtypes(include=['number']).columns
#     df_combinado[num_cols] = df_combinado[num_cols].fillna(0)

#     df_combinado = combinar_y_sintetizar_faltantes(df_global, df_simulado)

#     # 6. ELIMINAR COLUMNAS DE TEXTO RESTANTES (Opcional, para ML puro)
#     # Si quieres que el resultado sea 100% numérico para entrenar modelos:
#     # df_combinado = df_combinado.select_dtypes(include=['number'])

#     # 7. GUARDADO
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     path_final = os.path.join(output_dir, 'datos_combinados_limpios.csv')
#     df_combinado.to_csv(path_final, index=False)
    
#     return df_combinado

# import pandas as pd
# import numpy as np

# def combinar_y_sintetizar_faltantes(df_global, df_simulado):
#     # 1. Estandarizar nombres de columnas en el simulado
#     # Mapeamos lo que ya tenemos para que encaje con el Global
#     mapeo = {
#         'Edad': 'Age',
#         'Genero': 'Gender',
#         'Fumador': 'Smoking',
#         'Consume_Alcohol': 'Alcohol_Use',
#         'Obesidad': 'Obesity_Level',
#         'Componente_Hereditario': 'Genetic_Risk',
#         'Diagnostico': 'Target_Severity_Score' 
#     }
#     df_sim = df_simulado.rename(columns=mapeo).copy()

#     # 2. Identificar qué columnas le faltan al simulado que el global sí tiene
#     cols_faltantes = [c for c in df_global.columns if c not in df_sim.columns]

#     # 3. SINTETIZAR los datos para esas columnas solo para los registros del simulador
#     n_sim = len(df_sim)
    
#     for col in cols_faltantes:
#         if col == 'Country_Region':
#             # Copiamos la distribución de países del dataset global
#             paises_probs = df_global['Country_Region'].value_counts(normalize=True)
#             df_sim[col] = np.random.choice(paises_probs.index, size=n_sim, p=paises_probs.values)
        
#         elif col == 'Year':
#             df_sim[col] = np.random.choice(df_global['Year'].unique(), size=n_sim)
            
#         elif col == 'Cancer_Type':
#             df_sim[col] = 'Colon' # En nuestro simulador siempre es Colon
            
#         elif col == 'Cancer_Stage':
#             # Lógica: Si el score es alto (enfermo), asignar etapa. Si no, None.
#             etapas_reales = [e for e in df_global['Cancer_Stage'].unique() if e != 'None']
#             df_sim[col] = df_sim['Target_Severity_Score'].apply(
#                 lambda x: np.random.choice(etapas_reales) if x > 0 else 'None'
#             )
            
#         elif col in ['Air_Pollution', 'Treatment_Cost_USD', 'Survival_Years']:
#             # Rellenamos con valores aleatorios dentro del rango real (Media +/- Desviación)
#             mu, sigma = df_global[col].mean(), df_global[col].std()
#             df_sim[col] = np.random.normal(mu, sigma, n_sim).round(2)
#             # Aseguramos que no haya valores negativos
#             df_sim[col] = df_sim[col].clip(lower=df_global[col].min())

#     # 4. Unir ambos datasets
#     df_final = pd.concat([df_global, df_sim], axis=0, ignore_index=True)

#     # 5. Limpieza final (Duplicados y tipos)
#     df_final = df_final.drop_duplicates()
    
#     return df_final

##################################################

# --- 3. FUNCIÓN DE SINTETIZACIÓN DE RELLENO ---
def combinar_y_sintetizar_faltantes(df_global, df_simulado):
    # 1. Preparar df_simulado con nombres compatibles
    mapeo = {
        'Edad': 'Age', 
        'Genero': 'Gender', 
        'Fumador': 'Smoking',
        'Consume_Alcohol': 'Alcohol_Use', 
        'Obesidad': 'Obesity_Level',
        'Componente_Hereditario': 'Genetic_Risk', 
        'Diagnostico': 'Target_Severity_Score',
        'FOBT_Resultado (Sangre en heces)': 'FOBT_n' # Estandarizamos nombre
    }
    df_sim = df_simulado.rename(columns=mapeo).copy()
    
    # Convertimos el FOBT original del simulador a numérico si es string
    if df_sim['FOBT_n'].dtype == 'object':
        df_sim['FOBT_n'] = df_sim['FOBT_n'].map({'Positive': 1, 'Negative': 0}).fillna(0)

    # --- A. RELLENAR DATOS GLOBALES PARA LOS 5.000 REGISTROS SIMULADOS ---
    cols_global_faltantes = [c for c in df_global.columns if c not in df_sim.columns]
    n_sim = len(df_sim)
    
    for col in cols_global_faltantes:
        if col == 'Country_Region':
            paises_probs = df_global['Country_Region'].value_counts(normalize=True)
            df_sim[col] = np.random.choice(paises_probs.index, size=n_sim, p=paises_probs.values)
        elif col == 'Year':
            df_sim[col] = np.random.choice(df_global['Year'].unique(), size=n_sim)
        elif col == 'Cancer_Type':
            df_sim[col] = 'Colon'
        elif col == 'Cancer_Stage':
            etapas_reales = [e for e in df_global['Cancer_Stage'].unique() if e != 'None']
            df_sim[col] = df_sim['Target_Severity_Score'].apply(
                lambda x: np.random.choice(etapas_reales) if x > 0 else 'None'
            )
        elif col in ['Air_Pollution', 'Treatment_Cost_USD', 'Survival_Years']:
            mu, sigma = df_global[col].mean(), df_global[col].std()
            df_sim[col] = np.random.normal(mu, sigma, n_sim).round(2)

    # --- B. RELLENAR DATOS MÉDICOS PARA LOS 50.000 REGISTROS GLOBALES ---
    df_glob_copy = df_global.copy()
    n_glob = len(df_glob_copy)
    
    # Definimos qué columnas del simulador queremos "inventar" para el Global
    # Usamos nombres limpios y numéricos
    nuevas_cols_medicas = {
        'FOBT_n': 'binario',
        'CEA_Level_ng_mL (Marcador Tumoral)': 'float',
        'Sedentarismo': 'binario',
        'Diabetes_tipo_2': 'binario',
        'Antecedentes_Familiares': 'binario',
        'Dieta_rica_en_grasas_animales': 'binario'
    }
    
    for col, tipo in nuevas_cols_medicas.items():
        if col == 'FOBT_n':
            # 80% prob de positivo si severidad > 5
            df_glob_copy[col] = df_glob_copy['Target_Severity_Score'].apply(
                lambda x: np.random.binomial(n=1, p=0.8) if x > 5 else np.random.binomial(n=1, p=0.05)
            )
        elif 'CEA_Level' in col:
            df_glob_copy[col] = df_glob_copy['Target_Severity_Score'].apply(
                lambda x: np.random.normal(12.0, 5.0) if x > 5 else np.random.normal(2.0, 1.0)
            ).round(2).clip(lower=0.1)
        else:
            # Factores de riesgo generales (30% de probabilidad base)
            df_glob_copy[col] = np.random.binomial(n=1, p=0.3, size=n_glob)

    # --- C. UNIÓN FINAL ---
    df_final = pd.concat([df_glob_copy, df_sim], axis=0, ignore_index=True)
    
    # Asegurar que no queden NaNs en las columnas nuevas por diferencias de nombres
    df_final = df_final.fillna(0)
    
    return df_final

# --- 4. FUNCIÓN PRINCIPAL DE COMBINACIÓN ---
def combinar_datos(df_global, df_simulado, output_dir):
    # Ahora esta función devuelve 55.000 filas con datos en casi todas las columnas
    df_combinado = combinar_y_sintetizar_faltantes(df_global, df_simulado)

    # Identificador de origen
    df_combinado['Is_Simulated'] = 0
    # Los últimos n_sim registros son los simulados
    df_combinado.iloc[-len(df_simulado):, df_combinado.columns.get_loc('Is_Simulated')] = 1

    # Limpieza final de seguridad
    df_combinado = df_combinado.fillna(0)
    
    # Guardar...
    df_combinado.to_csv(os.path.join(output_dir, 'datos_combinados_limpios.csv'), index=False)
    return df_combinado