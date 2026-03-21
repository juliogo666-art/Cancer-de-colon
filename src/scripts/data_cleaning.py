"""
data_cleaning.py — Funciones de limpieza y transformación de datos para el proyecto
de cáncer de colon.

Las funciones de limpieza SOLO transforman datos y devuelven DataFrames.
El guardado a CSV se realiza por separado con `guardar_csv()`.
"""

import pandas as pd
import os
import streamlit as st

from src.scripts.sintetiza_historiales import (
    sintetizar_historiales,
    sintetizar_datos_kaggle,
)


def guardar_csv(df, directorio_destino, nombre_archivo):
    """
    Guarda un DataFrame como CSV en el directorio especificado.

    Parameters
    ----------
    df : pd.DataFrame
    directorio_destino : str
    nombre_archivo : str
    """
    os.makedirs(directorio_destino, exist_ok=True)
    output_path = os.path.join(directorio_destino, nombre_archivo)

    try:
        df.to_csv(output_path, index=False)
    except PermissionError:
        st.error(f"Error de permisos: Cierra el archivo {output_path} si está abierto.")


def limpiar_datos_globales(df, file_path_sin_d):
    """
    Limpia y enriquece el dataset global de cáncer.
    Añade columnas numéricas para Gender y Cancer_Stage.
    """
    df = df.copy()

    # 1. Eliminar filas con nulos
    df = df.dropna()

    # 2. CREAR NUEVAS COLUMNAS NUMÉRICAS (Sin tocar las originales)

    # Género numérico -> Gender_n
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    df["Gender_n"] = df["Gender"].map(gender_map)

    # Etapa numérica -> Stage_n
    stage_map = {
        "Stage 0": 0,
        "Stage I": 1,
        "Stage II": 2,
        "Stage III": 3,
        "Stage IV": 4,
    }
    df["Stage_n"] = df["Cancer_Stage"].map(stage_map)

    # 3. Limpieza de tipos para compatibilidad con Streamlit (LargeUtf8 fix)
    # Aplicamos la limpieza a todas las columnas para que no rompa el visualizador
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            # Mantenemos el texto original pero limpiamos metadatos de Arrow
            df[col] = pd.Series(df[col].astype(str).tolist(), index=df.index)
        elif "int" in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        elif "float" in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    # df = df.drop(columns=['Country_Region'])

    # 4. Guardado del CSV con las nuevas columnas incluidas
    directorio_destino = os.path.dirname(file_path_sin_d)
    guardar_csv(df, directorio_destino, "ej_global_cancer_limpio.csv")

    return df


def limpiar_datos_sinteticos(df, file_path_con_d):
    """
    Limpia el dataset de pacientes simulados.
    Elimina duplicados, nulos e IDs; crea columnas numéricas.
    """
    df = df.copy()

    # 1. Eliminar duplicados y nulos
    df = df.drop_duplicates()
    df = df.dropna()

    # 2. Eliminar IDs (No sirven para cálculos numéricos)
    if "Paciente_ID" in df.columns:
        df = df.drop(columns=["Paciente_ID"])

    # 3. CREAR NUEVAS COLUMNAS NUMÉRICAS

    # Género (Asegúrate de que en el CSV sea 'Genero')
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    # Usamos .get para evitar errores si aparece un valor inesperado
    df["Gender_n"] = df["Genero"].map(gender_map)

    # FOBT Resultado
    result_fobt_map = {"Negative": 0, "Positive": 1}
    # Nota: El nombre exacto en tu CSV es 'FOBT_Resultado (Sangre en heces)'
    col_fobt = "FOBT_Resultado (Sangre en heces)"
    if col_fobt in df.columns:
        df["FOBT_Resultado_n"] = df[col_fobt].map(result_fobt_map)

    # 4. Asegurar que TODO sea numérico (para las columnas que ya eran 0 y 1)
    # Algunas columnas como 'Diagnostico' o 'Antecedentes' ya son 0/1,
    # pero a veces Pandas las lee como objetos.
    for col in df.columns:
        if "int" in str(df[col].dtype) or "float" in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 5. Limpieza final de metadatos para Streamlit
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            df[col] = pd.Series(df[col].astype(str).tolist(), index=df.index)

    # 6. Guardado
    directorio_destino = os.path.dirname(file_path_con_d)
    guardar_csv(df, directorio_destino, "ej_pacientes_simulador_limpio.csv")

    return df


def combinar_datos_s_g(df_global, output_dir):
    """Combina datos globales con datos sintéticos generados."""
    df_global = df_global.copy()
    df_global = sintetizar_historiales(df_global, output_dir)
    return df_global


def limpiar_datos_kaggle(df, file_path_sin_d):
    """
    Limpia y transforma el dataset de Kaggle (colorectal cancer).
    Convierte todas las columnas categóricas a numéricas.

    NOTA sobre fillna(0): En este dataset las columnas categóricas se mapean a enteros,
    por lo que un valor no reconocido se convierte en NaN y luego en 0.
    Para datos clínicos reales, sería preferible usar NaN o un valor centinela
    que no se confunda con una categoría válida (ej: -1).
    """
    df = df.copy()

    # 1. Limpieza inicial
    df = df.dropna()
    df = df.drop_duplicates()

    # 2. TRANSFORMACIÓN A NUMÉRICO (Mapeos exhaustivos)

    # Género: M -> 0, F -> 1
    gender_map = {"M": 0, "F": 1}
    df["Gender"] = df["Gender"].map(gender_map)

    # Estadio del Cáncer (Corregido 'Localized' en lugar de 'Localizes')
    df["Cancer_Stage"] = df["Cancer_Stage"].map(
        {"Localized": 0, "Metastatic": 1, "Regional": 2}
    )

    # Columnas Binarias (Yes/No)
    columnas_yes_no = [
        "Family_History",
        "Smoking_History",
        "Alcohol_Consumption",
        "Diabetes",
        "Inflammatory_Bowel_Disease",
        "Genetic_Mutation",
        "Early_Detection",
        "Survival_5_years",
        "Mortality",
        "Survival_Prediction",
    ]
    for col in columnas_yes_no:
        if col in df.columns:
            df[col] = df[col].map({"No": 0, "Yes": 1})

    # Niveles de Riesgo y Actividad (Low/Moderate/High)
    riesgo_map = {"Low": 0, "Moderate": 1, "High": 2}
    for col in ["Diet_Risk", "Physical_Activity", "Healthcare_Access"]:
        if col in df.columns:
            df[col] = df[col].map(riesgo_map)

    # Obesidad BMI (Normal/Overweight/Obese)
    bmi_map = {"Normal": 0, "Overweight": 1, "Obese": 2}
    if "Obesity_BMI" in df.columns:
        df["Obesity_BMI"] = df["Obesity_BMI"].map(bmi_map)

    # Screening History (Never/Irregular/Regular)
    screening_map = {"Never": 0, "Irregular": 1, "Regular": 2}
    if "Screening_History" in df.columns:
        df["Screening_History"] = df["Screening_History"].map(screening_map)

    # Entorno y Economía
    if "Urban_or_Rural" in df.columns:
        df["Urban_or_Rural"] = df["Urban_or_Rural"].map({"Rural": 0, "Urban": 1})

    if "Economic_Classification" in df.columns:
        df["Economic_Classification"] = df["Economic_Classification"].map(
            {"Developing": 0, "Developed": 1}
        )

    if "Insurance_Status" in df.columns:
        df["Insurance_Status"] = df["Insurance_Status"].map(
            {"Uninsured": 0, "Insured": 1}
        )

    # Tipo de Tratamiento
    tratamiento_map = {
        "Surgery": 0,
        "Chemotherapy": 1,
        "Radiation": 2,
        "Combination": 3,
    }
    if "Treatment_Type" in df.columns:
        df["Treatment_Type"] = df["Treatment_Type"].map(tratamiento_map)

    # 3. Eliminar columnas de texto que no se pueden numerar
    if "Country" in df.columns:
        df = df.drop(columns=["Country"])

    # 4. Forzar numérico (con documentación de la decisión de fillna)
    # NOTA: fillna(0) es aceptable aquí porque todas las columnas ya fueron
    # mapeadas a valores numéricos. Un NaN significaría un valor no mapeado.
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 5. Guardado
    directorio_destino = os.path.dirname(file_path_sin_d)
    guardar_csv(df, directorio_destino, "ej_kaggle_cancer_limpio.csv")

    return df


def limpiar_datos_kaggle_finales(df_global, output_dir):
    """Combina datos de Kaggle con datos sintéticos generados."""
    df_global = df_global.copy()
    df_global = sintetizar_datos_kaggle(df_global, output_dir)
    return df_global

def limpiar_datos_riesgo_def(df, file_path_sin_d):
    """
    Limpia el dataset de factores de riesgo y asigna Patient_ID único.
    """
    df = df.copy()

    # 1. Limpieza inicial
    df = df.dropna()
    df = df.drop_duplicates()

    df = df.drop(columns=['Age', 'Gender'], errors='ignore')

    # 2. Asignación de Patient_ID (Si no existe)
    # Generamos IDs únicos desde 30000 en adelante
    if "Patient_ID" not in df.columns:
        df.insert(0, 'Patient_ID', range(00000, 00000 + len(df)))

    # 3. Mapeo de niveles de riesgo
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    if "Risk_Level" in df.columns:
        # Creamos la versión numérica sin borrar la original para el EDA
        df["Risk_Level_n"] = df["Risk_Level"].map(risk_map)

    # 4. Convertir columnas a numérico de forma segura
    # Solo aplicamos to_numeric a las columnas que no son 'Risk_Level' (texto)
    cols_a_convertir = df.columns.drop(['Risk_Level']) if 'Risk_Level' in df.columns else df.columns
    for col in cols_a_convertir:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 5. Guardado
    directorio_destino = os.path.dirname(file_path_sin_d)
    guardar_csv(df, directorio_destino, "cancer_risk_clean.csv")
    
    return df