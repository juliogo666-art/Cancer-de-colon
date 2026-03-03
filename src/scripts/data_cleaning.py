from sys import path

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

def limpiar_datos_globales_combinado(df):
    # Asegurar tipos de datos básicos y eliminar nulos si existieran
    df = df.dropna(subset=['Patient_ID'])
    # Convertir variables categóricas a numéricas si es necesario para el ML futuro

    if 'FOBT_Resultado_n' in df.columns and 'FOBT_Resultado (Sangre en heces)' not in df.columns:
        df['FOBT_Resultado (Sangre en heces)'] = df['FOBT_Resultado_n'].map({1: 'Positive', 0: 'Negative'})
    return df

def limpiar_datos_sinteticos_com(df):
    # En el simulador, 'Diagnostico' parece ser el target (0 o 1)
    # Estandarizamos el resultado de sangre en heces a numérico
    if 'FOBT_Resultado (Sangre en heces)' in df.columns:
        df['FOBT_Resultado_n'] = df['FOBT_Resultado (Sangre en heces)'].map({'Positive': 1, 'Negative': 0}).fillna(0).astype(int)
    return df

def combinar_datos_y_sintetizarlos(df_global, df_simulado, output_dir):
    df_global = df_global.copy()
    df_simulado = df_simulado.copy()

    df_glob_ext = limpiar_datos_globales_combinado(df_global)
    df_sim_raw = limpiar_datos_sinteticos_com(df_simulado)

    mapeo = {
        'Edad': 'Age', 'Genero': 'Gender', 'Fumador': 'Smoking',
        'Consume_Alcohol': 'Alcohol_Use', 'Obesidad': 'Obesity_Level',
        'Componente_Hereditario': 'Genetic_Risk', 'Diagnostico': 'Target_Severity_Score'
    }
    df_sim = df_sim_raw.rename(columns=mapeo).copy()

    # --- A. RELLENAR COLUMNAS DEL GLOBAL EN EL SIMULADOR ---
    n_sim = len(df_sim)
    cols_faltantes_en_sim = [c for c in df_glob_ext.columns if c not in df_sim.columns]

    for col in cols_faltantes_en_sim:
        if col == 'Country_Region':
            probs = df_glob_ext['Country_Region'].value_counts(normalize=True)
            df_sim[col] = np.random.choice(probs.index, size=n_sim, p=probs.values)
        elif col == 'Year':
            df_sim[col] = np.random.choice(df_glob_ext['Year'].unique(), size=n_sim)
        elif col in ['Treatment_Cost_USD', 'Survival_Years', 'Air_Pollution']:
            mu, sigma = df_glob_ext[col].mean(), df_glob_ext[col].std()
            df_sim[col] = np.random.normal(mu, sigma, n_sim).round(2).clip(min=0)
        elif col == 'Cancer_Stage':
            etapas = ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
            df_sim[col] = df_sim['Target_Severity_Score'].apply(
                lambda x: np.random.choice(etapas[2:]) if x > 5 else np.random.choice(etapas[:2])
            )
        else:
            df_sim[col] = 0

    # --- B. RELLENAR COLUMNAS DEL SIMULADOR EN EL GLOBAL ---
    n_glob = len(df_glob_ext)
    cols_que_vienen_del_sim = [c for c in df_sim.columns if c not in df_glob_ext.columns]

    for col in cols_que_vienen_del_sim:
        if 'CEA_Level' in col:
            df_glob_ext[col] = df_glob_ext['Target_Severity_Score'].apply(
                lambda x: np.random.normal(12.0, 4.0) if x > 6 else np.random.normal(2.0, 1.0)
            ).round(2).clip(lower=0.1)
        
        elif col == 'FOBT_Resultado (Sangre en heces)':
            # Generamos el STRING directamente para el Global basado en severidad
            df_glob_ext[col] = df_glob_ext['Target_Severity_Score'].apply(
                lambda x: 'Positive' if (x > 5 and np.random.rand() > 0.2) else 'Negative'
            )
        
        elif col == 'FOBT_Resultado_n':
            # Sincronizamos la numérica con la de texto que acabamos de crear arriba
            df_glob_ext[col] = df_glob_ext['FOBT_Resultado (Sangre en heces)'].map({'Positive': 1, 'Negative': 0})

        else:
            # Otros factores binarios (Sedentarismo, etc.)
            df_glob_ext[col] = np.random.binomial(n=1, p=0.3, size=n_glob)

    # --- C. UNIÓN ---
    df_final = pd.concat([df_glob_ext, df_sim], axis=0, ignore_index=True).fillna(0)
    
    # Asegurar que el tipo de datos de la columna numérica sea int
    if 'FOBT_Resultado_n' in df_final.columns:
        df_final['FOBT_Resultado_n'] = df_final['FOBT_Resultado_n'].astype(int)

    output_path = os.path.join(output_dir, 'datos_combinados_completos.csv')
    df_final.to_csv(output_path, index=False)
    
    return df_final