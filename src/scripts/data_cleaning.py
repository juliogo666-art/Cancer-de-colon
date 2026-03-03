import pandas as pd
import os
import streamlit as st
import numpy as np

from src.scripts.sintetiza_historiales import sintetizar_historiales

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

def combinar_datos_s_g (df_global, output_dir):
    df_global = df_global.copy()
    df_global = sintetizar_historiales(df_global, output_dir)

    return df_global