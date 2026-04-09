import streamlit as st
import joblib
import os
import sys
import numpy as np
import pandas as pd

# Configuración de página (debe ser lo primero)
st.set_page_config(page_title="ColonAI - Sistema Integral", layout="wide")

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# RUTAS
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.dirname(os.path.dirname(directorio_actual))
sys.path.append(directorio_raiz)

from src.utils.cargar_modelos_s import predecir, colonos, obtener_modelo_cnn, biopsias
from src.utils.data_load_s import datos_p, nombres_p, save_r

CSV_PATH = os.path.join(
    directorio_raiz,
    "src",
    "data",
    "raw",
    "historial_pacientes",
    "cancer_risk_final.csv",
)
MODEL_ML_PATH = os.path.join(
    directorio_raiz, "src", "models", "ml", "lgbm_clinico.pkl"
)

CSV_5000_PATH = os.path.join(directorio_raiz, "src", "data", "raw", "historial_pacientes", "nuevos_pacientes_5000.csv")
CSV_RISK_PATH = os.path.join(directorio_raiz, "src", "data", "raw", "historial_pacientes", "cancer_risk_final.csv")

@st.cache_resource
def cargar_recursos():
    if not os.path.exists(MODEL_ML_PATH):
        st.error(f"No se encuentra el archivo del modelo en: {MODEL_ML_PATH}")
        return None
    try:
        modelo = joblib.load(MODEL_ML_PATH)
        # ESTO TE DIRÁ LA VERDAD:
        print(f"DEBUG: El objeto cargado es de tipo: {type(modelo)}") 
        return modelo
    except Exception as e:
        st.error(f"Error al cargar modelo ML: {e}")
        return None

modelo_ia = cargar_recursos()

# --- INTERFAZ STREAMLIT ---
st.title("Sistema Integral ColonAI")

tab1, tab2 = st.tabs(["Análisis de Datos", "Visión por Computadora"])
with tab1:
    col_sidebar, col_main = st.columns([1, 2])

    # 1. INICIALIZACIÓN
    if "form_data" not in st.session_state:
        st.session_state.form_data = {
            "fuma": "No", "alc": 0.0, "fam": False,
            "diet_red": 5.0, "diet_salt": 5.0, "diet_veg": 5.0,
            "phys": 5.0, "bmi": 25.0, "sangre": "Negativo",
            "cea": 0.0
        }

    with col_sidebar:
        st.subheader("Búsqueda")
        # Obtenemos IDs de la tabla 5000
        ids_pacientes = nombres_p(CSV_5000_PATH)
        id_seleccionado = st.selectbox("ID Paciente (nuevos_pacientes_5000)", options=ids_pacientes)

        if st.button("CARGAR DATOS", use_container_width=True):
            # Buscamos datos en la tabla de Riesgo Final
            valores = datos_p(id_seleccionado, CSV_RISK_PATH)
            if valores:
                st.session_state.form_data.update(valores)
                st.success(f"Datos de ID {id_seleccionado} cargados.")
                st.rerun()
            else:
                st.error("ID no encontrado en cancer_risk_final.csv")

    with col_main:
        st.subheader("Panel Clínico")
        d = st.session_state.form_data

        r1_c1, r1_c2, r1_c3 = st.columns(3)
        in_fuma = r1_c1.selectbox("Fumador", ["No", "Sí"], index=1 if d["fuma"]=="Sí" else 0)
        in_alc = r1_c2.slider("Alcohol (0-10)", 0.0, 10.0, d["alc"])
        in_fam = r1_c3.checkbox("Herencia Familiar", value=d["fam"])

        r2_c1, r2_c2, r2_c3 = st.columns(3)
        in_bmi = r2_c1.number_input("BMI", value=d["bmi"])
        in_sangre = r2_c2.selectbox("FOBT (Sangre)", ["Negativo", "Positivo"], index=1 if d["sangre"]=="Positivo" else 0)
        in_cea = r2_c3.number_input("Nivel CEA", value=d["cea"])

        with st.expander("Factores de Estilo de Vida"):
            c1, c2, c3, c4 = st.columns(4)
            in_red = c1.number_input("Carne Roja", 0, 10, int(d["diet_red"]))
            in_salt = c2.number_input("Sal/Procesados", 0, 10, int(d["diet_salt"]))
            in_veg = c3.number_input("Fruta/Verdura", 0, 10, int(d["diet_veg"]))
            in_phys = c4.number_input("Act. Física", 0, 10, int(d["phys"]))

        st.divider()

        # BOTÓN CALCULAR (Predicción + SHAP)
        if st.button("CALCULAR RIESGO", type="primary", use_container_width=True):
            # Aquí llamamos a la función predecir que renderiza el HTML y el SHAP uno al lado del otro
            # 1. Convertimos los inputs de la interfaz a los valores que entiende la IA (0 y 1)
            # Importante: Revisa si tu modelo usa 1 para "Sí" o al revés
            fuma_n = 1 if in_fuma == "Sí" else 0
            sangre_n = 1 if in_sangre == "Positivo" else 0
            fam_n = 1 if in_fam else 0 # Checkbox a int
            predecir(
                modelo_ia, 
                id_seleccionado,
                in_fuma, in_alc, in_fam, 
                in_red, in_salt, in_veg, in_phys, 
                in_bmi, in_sangre, in_cea
            )

with tab2:
    st.subheader("Análisis de Diagnóstico por Imagen")

    # Selector para tipo de análisis
    tipo_analisis = st.radio(
        "Seleccione el tipo de estudio:",
        ["Captura de Colonoscopia (Localizar Pólipos)", "Muestra de Biopsia"],
        horizontal=True,
    )
    st.divider()

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        if "Colonoscopia" in tipo_analisis:
            label_upload = "Subir captura de Colonoscopia"
        else:
            label_upload = "Subir microfotografía de Biopsia"

        img_input = st.file_uploader(label_upload, type=["jpg", "png", "jpeg"])

    if img_input:
        import numpy as np
        from PIL import Image
        import cv2

        image = Image.open(img_input)
        img_array = np.array(image)

        if st.button("ANALIZAR IMAGEN", type="primary"):
            if "Colonoscopia" in tipo_analisis:
                with st.spinner("Analizando pólipos con TensorFlow..."):
                    # Obtenemos el HTML y la imagen procesada (Grad-CAM)
                    txt_resultado, img_gradcam = colonos(img_array)
                    
                    # Redimensionamos la original para que coincida en tamaño con el Grad-CAM (150x150)
                    img_original_res = cv2.resize(img_array, (150, 150))
                
                with img_col2:
                    # Creamos sub-columnas para mostrar Original y Grad-CAM lado a lado
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        st.image(img_original_res, caption="Captura Original", use_container_width=True)
                    with sub2:
                        st.image(img_gradcam, caption="Localización (Grad-CAM)", use_container_width=True)
                    
                    # El informe (porcentaje y recomendación) aparece justo debajo de las fotos
                    st.markdown(txt_resultado, unsafe_allow_html=True)
            else:
                with st.spinner("Analizando malignidad con PyTorch + Grad-CAM..."):
                    txt_resultado, img_heatmap, img_original = biopsias(img_array)
                
                with img_col2:
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        st.image(img_original, caption="Biopsia Original", use_container_width=True)
                    with sub_col2:
                        st.image(img_heatmap, caption="Mapa de Calor (Grad-CAM)", use_container_width=True)
                    st.markdown(txt_resultado, unsafe_allow_html=True)