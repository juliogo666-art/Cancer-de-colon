import pandas as pd
import streamlit as st
import os

from src.scripts.data_cleaning import (
    limpiar_datos_globales,
    limpiar_datos_sinteticos,
    combinar_datos_s_g,
)
from src.utils.eda_visualization import (
    eda_datos_globales,
    eda_datos_sinteticos,
    eda_datos_combinados,
)

# Configuración principal de la página de Streamlit
st.set_page_config(page_title="Cancer Colon ML", layout="wide")


def mostrar_vista_previa(df):
    """Muestra una vista previa y descripción estadística del DataFrame."""
    st.write("Vista previa de los datos")
    st.dataframe(df.head())
    st.dataframe(df.describe())


def eda(base_path):
    st.title("Sistema de análisis de cáncer de colón")

    # Definición de rutas de archivos base
    ruta_historial = os.path.join(
        base_path, "src", "data", "raw", "historial_pacientes"
    )

    file_path_globales = os.path.join(
        ruta_historial,
        "pacientes_con_cancer_tratado",
        "global_cancer_patients_2015_2024.csv",
    )
    file_path_sinteticos = os.path.join(
        ruta_historial, "historiales_sinteticos", "pacientes_simulador_colon.csv"
    )
    file_path_combinados = ruta_historial

    # Menú lateral para navegación
    menu = st.sidebar.radio(
        "Navegación",
        [
            "1. Datos globales",
            "2. Datos simulados",
            "3. Datos combinados",
        ],
    )

    # 1. Análisis de datos globales
    if menu == "1. Datos globales":
        st.header("Análisis de datos globales")

        if os.path.exists(file_path_globales):
            df = pd.read_csv(file_path_globales)
            df = limpiar_datos_globales(df, file_path_globales)

            mostrar_vista_previa(df)

            with st.container():
                st.write("Generando gráficos...")
                eda_datos_globales(df)
        else:
            st.error("No se encuentra el archivo de datos globales. Verifica la ruta.")

    # 2. Análisis de datos simulados
    elif menu == "2. Datos simulados":
        st.header("Análisis de datos simulados")

        if os.path.exists(file_path_sinteticos):
            df = pd.read_csv(file_path_sinteticos)
            df = limpiar_datos_sinteticos(df, file_path_sinteticos)

            mostrar_vista_previa(df)

            with st.container():
                st.write("Generando gráficos...")
                eda_datos_sinteticos(df)
        else:
            st.error("No se encuentra el archivo CSV simulado. Verifica la ruta.")

    # 3. Análisis de Datos Combinados
    elif menu == "3. Datos combinados":
        st.header("Análisis de Datos Combinados")

        if os.path.exists(file_path_globales) and os.path.exists(file_path_sinteticos):
            # Cargar y limpiar ambos datasets
            df_g = pd.read_csv(file_path_globales)

            df_g = limpiar_datos_globales(df_g, file_path_globales)

            # Combinar los datos limpios
            df = combinar_datos_s_g(df_g, file_path_combinados)

            mostrar_vista_previa(df)

            with st.container():
                st.write("Generando gráficos...")
                eda_datos_combinados(df)
        else:
            st.error(
                "Faltan archivos necesarios para combinar los datos. Verifica las rutas."
            )
