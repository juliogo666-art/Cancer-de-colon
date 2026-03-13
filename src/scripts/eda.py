import pandas as pd
import streamlit as st
import os

from src.scripts.data_cleaning import (
    limpiar_datos_globales,
    limpiar_datos_kaggle_finales,
    limpiar_datos_sinteticos,
    combinar_datos_s_g,
    limpiar_datos_kaggle,
    
)
from src.utils.eda_visualization import (
    eda_datos_globales,
    eda_datos_sinteticos,
    eda_datos_combinados,
    eda_datos_kaggle,
    eda_datos_Kaggle_f
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
    ruta_historial = os.path.join(base_path, "src", "data", "raw", "historial_pacientes")

    file_path_globales = os.path.join(ruta_historial,"pacientes_con_cancer_tratado","global_cancer_patients_2015_2024.csv",)
    file_path_sinteticos = os.path.join(ruta_historial, "historiales_sinteticos", "pacientes_simulador_colon.csv")
    file_path_combinados = ruta_historial

    file_path_final = os.path.join(ruta_historial, "datos_combinados_global_extendido_3.csv")

    file_path_kaggle = os.path.join(ruta_historial,"pacientes_con_cancer_tratado","colorectal_cancer_dataset.csv",)

    file_path_kaggle_f = os.path.join(ruta_historial, "datos_finales_Kaggle.csv")

    # Menú lateral para navegación
    menu = st.sidebar.radio(
        "Navegación",
        [
            "1. Datos globales",
            "2. Datos simulados",
            "3. Datos combinados",
            "4. Datos Kaggle", 
            "5. Datos finales",
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

        if os.path.exists(file_path_final):
            df_combinados = pd.read_csv(file_path_final)
            mostrar_vista_previa(df_combinados)

            with st.container():
                st.write("Generando gráficos...")
                eda_datos_combinados(df_combinados)

        else:
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
    elif menu == "4. Datos Kaggle":
        st.header("Análisis de Datos Kaggle")

        if os.path.exists(file_path_kaggle):
            # Cargar y limpiar ambos datasets
            df_k = pd.read_csv(file_path_kaggle, sep=';')

            df_k =  limpiar_datos_kaggle(df_k, file_path_kaggle)

            mostrar_vista_previa(df_k)

            with st.container():
                st.write("Generando gráficos...")
                eda_datos_kaggle(df_k)
        else:
            st.error(
                "Faltan archivos necesarios para combinar los datos. Verifica las rutas."
            )
    elif menu == "5. Datos finales":
        st.header("Análisis de Datos Finales")

        if os.path.exists(file_path_kaggle_f):
            df_combinados = pd.read_csv(file_path_kaggle_f)
            mostrar_vista_previa(df_combinados)

            with st.container():
                st.write("Generando gráficos...")
                eda_datos_Kaggle_f(df_combinados)

        else:
            if os.path.exists(file_path_kaggle):
                # Cargar y limpiar ambos datasets
                df_g = pd.read_csv(file_path_kaggle, sep=';')

                df_g = limpiar_datos_kaggle(df_g, file_path_kaggle)

                # Combinar los datos limpios
                df = limpiar_datos_kaggle_finales(df_g, file_path_combinados)

                mostrar_vista_previa(df)

                with st.container():
                    st.write("Generando gráficos...")
                    eda_datos_Kaggle_f(df)
            else:
                st.error(
                    "Faltan archivos necesarios para combinar los datos. Verifica las rutas."
                )
    else:
        st.error(
            "Faltan archivos necesarios para combinar los datos. Verifica las rutas."
        )
