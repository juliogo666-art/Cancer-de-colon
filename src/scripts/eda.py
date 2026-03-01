import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
import os

from src.scripts.data_cleaning import limpiar_datos_globales, limpiar_datos_sinteticos, combinar_datos
from src.utils.eda_visualization import eda_datos_globales, eda_datos_sinteticos, eda_datos_combinados

st.set_page_config(page_title="Cancer Colon ML", layout="wide")

def eda(base_path):
    st.title("Sistema de Análisis de Cáncer de Colón")

    file_path_sin_d = os.path.join(base_path, 'src', 'data', 'raw', 'historial_pacientes', 
                             'pacientes_con_cancer_tratado', 'global_cancer_patients_2015_2024.csv')
    
    file_path_con_d = os.path.join(base_path, 'src', 'data', 'raw', 'historial_pacientes', 
                             'historiales_sinteticos', 'pacientes_simulador_colon.csv')
    
    file_path_datos_combinados = os.path.join(base_path, 'src', 'data', 'raw', 'historial_pacientes')


    # Sidebar
    menu = st.sidebar.radio("Navegación", ["1. Datos Globales", "2. Datos Simulados (Próximamente)", "3. Datos Combinados"])

    if menu == "1. Datos Globales":
        st.header("Análisis de Datos del Globales")
        
        if os.path.exists(file_path_sin_d):
            df = pd.read_csv(file_path_sin_d)
            df = limpiar_datos_globales(df, file_path_sin_d)
            
            st.write("### Vista previa de los datos")
            st.dataframe(df.head())
            st.dataframe(df.describe())

            with st.container():
                st.write("Generando gráficos...")
                eda_datos_globales(df)

    elif menu == "2. Datos Simulados (Próximamente)":

        st.header("Análisis de Datos Simulados")

        if os.path.exists(file_path_con_d):
            df = pd.read_csv(file_path_con_d)
            df = limpiar_datos_sinteticos(df, file_path_con_d)
            
            st.write("### Vista previa de los datos")
            st.dataframe(df.head())
            st.dataframe(df.describe())

            # Botones para acciones
            with st.container():
                st.write("Generando gráficos...")
                eda_datos_sinteticos(df)
        else:
            st.error("No se encuentra el archivo CSV. Verifica la ruta.")
    
    elif menu == "3. Datos Combinados":
        st.header("Análisis de Datos Combinados")
        
        if os.path.exists(file_path_sin_d):
            df_g = pd.read_csv(file_path_sin_d)
            df_s = pd.read_csv(file_path_con_d)

            df_g = limpiar_datos_globales(df_g, file_path_sin_d)
            df_s = limpiar_datos_sinteticos(df_s, file_path_con_d)
            
            df = combinar_datos(df_g, df_s, file_path_datos_combinados)
            
            st.write("### Vista previa de los datos")
            st.dataframe(df.head())
            st.dataframe(df.describe())

            with st.container():
                st.write("Generando gráficos...")
                eda_datos_combinados(df)