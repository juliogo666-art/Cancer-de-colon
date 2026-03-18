import streamlit as st
import joblib
import os
import sys

# Configuración de página (debe ser lo primero)
st.set_page_config(page_title="ColonAI - Sistema Integral", layout="wide")

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# RUTAS
directorio_actual = os.path.dirname(os.path.abspath(__file__))
sys.path.append(directorio_actual)

from src.utils.cargar_modelos_s import predecir, colonos, obtener_modelo_cnn
from src.utils.data_load_s import datos_p, nombres_p, save_r

CSV_PATH = os.path.join(directorio_actual, 'src','data', 'raw', 'historial_pacientes', 'datos_finales_Kaggle.csv')
MODEL_ML_PATH = os.path.join(directorio_actual, 'src', 'models', 'ml', 'best_rf_model.pkl')

# Carga de modelos con caché para evitar recargas constantes
@st.cache_resource
def cargar_recursos():
    try:
        modelo = joblib.load(MODEL_ML_PATH)
        obtener_modelo_cnn()
        return modelo
    except Exception as e:
        st.error(f"Error inicial: {e}")
        return None

modelo_ia = cargar_recursos()

# --- INTERFAZ STREAMLIT ---
st.title("Sistema Integral ColonAI")

tab1, tab2 = st.tabs(["Análisis de Datos", "Visión por Computadora"])

with tab1:
    col_sidebar, col_main = st.columns([1, 2])

    with col_sidebar:
        st.subheader("Búsqueda")
        pacientes = nombres_p(CSV_PATH)
        selector = st.selectbox("Buscar Paciente", options=pacientes)
        
        btn_cargar = st.button("Cargar Datos", use_container_width=True)
        btn_nuevo = st.button("Nuevo Paciente", use_container_width=True)

    # Inicializar valores en session_state para que sean mutables por los botones
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            "edad": 0, "genero": "Masculino", "estadio": 1, "tumor": 0.0,
            "sangre": "No", "cea": 0.0, "fuma": "No", "alc": 0.0,
            "diab": "No", "fam": False, "ibd": False, "peso": 0.0, "altura": 0.0
        }

    # Lógica de botones de carga/limpieza
    if btn_cargar:
        # Aquí asumo que datos_p devuelve una lista/tupla con los valores
        vals = datos_p(selector, CSV_PATH)
        keys = ["edad", "genero", "estadio", "tumor", "fam", "fuma", "alc", "diab", "ibd", "sangre", "cea", "altura", "peso"]
        st.session_state.form_data.update(dict(zip(keys, vals)))

    if btn_nuevo:
        for k in st.session_state.form_data:
            st.session_state.form_data[k] = False if isinstance(st.session_state.form_data[k], bool) else 0
            if k == "genero": st.session_state.form_state[k] = "Masculino"
            if k in ["sangre", "fuma", "diab"]: st.session_state.form_data[k] = "No"

    with col_main:
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        in_edad = r1_c1.number_input("Edad", value=int(st.session_state.form_data["edad"]))
        in_genero = r1_c2.selectbox("Género", ["Masculino", "Femenino"], index=0 if st.session_state.form_data["genero"]=="Masculino" else 1)
        in_estadio = r1_c3.selectbox("Estadio", [1, 2, 3, 4], index=int(st.session_state.form_data["estadio"])-1)

        r2_c1, r2_c2, r2_c3 = st.columns(3)
        in_tumor = r2_c1.number_input("Tamaño Tumor (mm)", value=float(st.session_state.form_data["tumor"]))
        in_sangre = r2_c2.radio("Sangre en Heces", ["No", "Sí"], index=0 if st.session_state.form_data["sangre"]=="No" else 1, horizontal=True)
        in_cea = r2_c3.number_input("Nivel CEA", value=float(st.session_state.form_data["cea"]))

        r3_c1, r3_c2, r3_c3 = st.columns(3)
        in_fuma = r3_c1.radio("Fumador", ["No", "Sí"], index=0 if st.session_state.form_data["fuma"]=="No" else 1, horizontal=True)
        in_alc = r3_c2.number_input("Alcohol semanal", value=float(st.session_state.form_data["alc"]))
        in_diab = r3_c3.radio("Diabetes", ["No", "Sí"], index=0 if st.session_state.form_data["diab"]=="No" else 1, horizontal=True)

        r4_c1, r4_c2, r4_c3, r4_c4 = st.columns(4)
        in_fam = r4_c1.checkbox("Antecedentes Familiares", value=st.session_state.form_data["fam"])
        in_ibd = r4_c2.checkbox("Enfermedad Inflamatoria", value=st.session_state.form_data["ibd"])
        in_peso = r4_c3.number_input("Peso (kg)", value=float(st.session_state.form_data["peso"]))
        in_altura = r4_c4.number_input("Altura (cm)", value=float(st.session_state.form_data["altura"]))

    st.divider()
    
    c_btn1, c_btn2 = st.columns(2)
    if c_btn1.button("CALCULAR RIESGO", type="primary", use_container_width=True):
        res_html = predecir(modelo_ia, selector, in_edad, in_genero, in_estadio, in_tumor, in_sangre, in_cea, in_fuma, in_alc, in_diab, in_fam, in_ibd, in_peso, in_altura)
        st.markdown(res_html, unsafe_allow_html=True)

    if c_btn2.button("UARDAR REGISTRO", use_container_width=True):
        res_save = save_r(directorio_actual, selector, in_edad, in_genero, in_estadio, in_tumor, in_fam, in_fuma, in_alc, in_diab, in_ibd, in_sangre, in_cea, in_altura, in_peso)
        st.markdown(res_save, unsafe_allow_html=True)

with tab2:
    st.subheader("Análisis de Imagen de Colonoscopia")
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        img_input = st.file_uploader("Subir captura de Colonoscopia", type=["jpg", "png", "jpeg"])
    
    if img_input:
        # Para que funcione igual que Gradio, hay que pasarle la imagen procesada
        # Streamlit entrega un objeto BytesIO, algunas funciones necesitan la ruta o el array
        import numpy as np
        from PIL import Image
        image = Image.open(img_input)
        img_array = np.array(image)
        
        if st.button("ANALIZAR IMAGEN", type="primary"):
            txt_resultado, img_output = colonos(img_array)
            
            with img_col2:
                st.image(img_output, caption="Procesado OpenCV")
                st.markdown(txt_resultado, unsafe_allow_html=True)