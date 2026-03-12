import pandas as pd
import os
import streamlit as st
import numpy as np

from src.scripts.sintetiza_historiales import sintetizar_historiales, sintetizar_datos_kaggle

def limpiar_datos_globales(df, file_path_sin_d):
    df = df.copy()
    # if 'Cancer_Type' in df.columns:
    #     df = df[df['Cancer_Type'] == 'Colon']
        # Una vez filtrado, ya podemos eliminar la columna porque todos son 'Colon'
        # df = df.drop(columns=['Cancer_Type'])

    # 1. Eliminar columnas que realmente no sirven
    df = df.dropna()

    # 2. CREAR NUEVAS COLUMNAS (Sin tocar las originales)
    
    # Género numérico -> Gender_n
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    df['Gender_n'] = df['Gender'].map(gender_map)

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

    # df = df.drop(columns=['Country_Region'])

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

def limpiar_datos_kaggle(df, file_path_sin_d):
    df = df.copy()
    
    # 1. Limpieza inicial
    df = df.dropna()
    df = df.drop_duplicates()

    # 2. TRANSFORMACIÓN A NUMÉRICO (Mapeos exhaustivos)
    
    # Género: M -> 0, F -> 1
    gender_map = {'M': 0, 'F': 1}
    df['Gender'] = df['Gender'].map(gender_map)

    # Estadio del Cáncer (Corregido 'Localized' en lugar de 'Localizes')
    df['Cancer_Stage'] = df['Cancer_Stage'].map({'Localized': 0, 'Metastatic': 1, 'Regional': 2})

    # Columnas Binarias (Yes/No)
    columnas_yes_no = [
        'Family_History', 'Smoking_History', 'Alcohol_Consumption', 
        'Diabetes', 'Inflammatory_Bowel_Disease', 'Genetic_Mutation', 
        'Early_Detection', 'Survival_5_years', 'Mortality', 'Survival_Prediction'
    ]
    for col in columnas_yes_no:
        if col in df.columns:
            df[col] = df[col].map({'No': 0, 'Yes': 1})

    # Niveles de Riesgo y Actividad (Low/Moderate/High)
    riesgo_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    if 'Diet_Risk' in df.columns:
        df['Diet_Risk'] = df['Diet_Risk'].map(riesgo_map)
    if 'Physical_Activity' in df.columns:
        df['Physical_Activity'] = df['Physical_Activity'].map(riesgo_map)
    if 'Healthcare_Access' in df.columns:
        df['Healthcare_Access'] = df['Healthcare_Access'].map(riesgo_map)

    # Obesidad BMI (Normal/Overweight/Obese)
    bmi_map = {'Normal': 0, 'Overweight': 1, 'Obese': 2}
    if 'Obesity_BMI' in df.columns:
        df['Obesity_BMI'] = df['Obesity_BMI'].map(bmi_map)

    # Screening History (Never/Irregular/Regular)
    screening_map = {'Never': 0, 'Irregular': 1, 'Regular': 2}
    if 'Screening_History' in df.columns:
        df['Screening_History'] = df['Screening_History'].map(screening_map)

    # Entorno y Economía
    if 'Urban_or_Rural' in df.columns:
        df['Urban_or_Rural'] = df['Urban_or_Rural'].map({'Rural': 0, 'Urban': 1})
    
    if 'Economic_Classification' in df.columns:
        df['Economic_Classification'] = df['Economic_Classification'].map({'Developing': 0, 'Developed': 1})
    
    if 'Insurance_Status' in df.columns:
        df['Insurance_Status'] = df['Insurance_Status'].map({'Uninsured': 0, 'Insured': 1})

    # Tipo de Tratamiento
    tratamiento_map = {'Surgery': 0, 'Chemotherapy': 1, 'Radiation': 2, 'Combination': 3}
    if 'Treatment_Type' in df.columns:
        df['Treatment_Type'] = df['Treatment_Type'].map(tratamiento_map)

    # 3. Eliminar columnas de texto que no se pueden numerar o no sirven
    if 'Country' in df.columns:
        df = df.drop(columns=['Country'])

    # 4. Forzar que todo lo que quede sea numérico (por si acaso)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 5. Guardado
    directorio_destino = os.path.dirname(file_path_sin_d)
    output_path = os.path.join(directorio_destino, 'ej_kaggle_cancer_limpio.csv')
    
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)
        
    df.to_csv(output_path, index=False)
    
    return df

def limpiar_datos_kaggle_finales (df_global, output_dir):
    df_global = df_global.copy()
    df_global = sintetizar_datos_kaggle(df_global, output_dir)

    return df_global