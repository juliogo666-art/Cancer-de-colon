"""
Aplicado en el app finalmente:

    - El formulario completo con todos los campos del paciente
    - La búsqueda por DNI/NUSS con resolución de Patient_ID
    - Los botones de Cargar, Actualizar y Guardar Registro
    - La predicción ML con SHAP (el informe visual con % de riesgo)
    - El análisis de biopsias (PyTorch + Grad-CAM)
    - El radio button para elegir entre Colonoscopia y Biopsia
    - La visualización lado a lado (original + Grad-CAM)

Arrancar con:
    streamlit run src/frontend/app.py
"""

import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image

###############################################################################
# Configuración de la pagina
###############################################################################
st.set_page_config(
    page_title="Galeno - Sistema de Diagnóstico de Cáncer de Colon",
    page_icon="⚕️",
    layout="wide",
)

# ==========================================
# ESTILO CUSTOM: CLINICAL CLEAN & HOPE (WHITE & GREEN)
# ==========================================
st.markdown(
    """
<style>
    /* 1. Fondo principal blanco y limpio */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* 2. Botones en Verde Vibrante (Sanación y Esperanza) */
    div.stButton > button:first-child {
        background-color: #22c55e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    div.stButton > button:first-child:hover {
        background-color: #16a34a;
        color: white;
        border: none;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(34, 197, 94, 0.3);
    }
    
    /* Botón primario (el de Calcular) con un toque especial */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #22c55e 0%, #10b981 100%);
    }

    /* 3. Estilo de Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8fafc;
        padding: 0 20px;
        border-radius: 12px 12px 0 0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        color: #64748b;
    }

    .stTabs [aria-selected="true"] {
        color: #22c55e !important;
        border-bottom-color: #22c55e !important;
        font-weight: bold;
    }

    /* 4. Inputs y Selects */
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #22c55e;
        box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2);
    }

    /* 5. Títulos y Textos (Forzados para visibilidad) */
    h1, h2, h3, h4, [data-testid="stHeader"] {
        color: #0f172a !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    
    .main-title {
        color: #0f172a;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -1px;
    }
    
    .main-subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .stMarkdown p, .stMarkdown li {
        color: #334155 !important;
    }

    /* 5b. Etiquetas y Radio Buttons (Corrección visibilidad Tab 2) */
    label, [data-testid="stWidgetLabel"] p, [data-testid="stRadio"] label p {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stRadio"] div[role="radiogroup"] {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }

    /* 6. Tarjetas Informativas */
    [data-testid="stMetric"] {
        background-color: #f1f5f9;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #22c55e;
    }

    /* 7. Centrado vertical de Columnas (Header) */
    [data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Parche necesario para que TensorFlow use la versión correcta de Keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

###############################################################################
# Configurar rutas para que Python encuentre nuestros módulos
###############################################################################
# Calculamos dónde está la raíz del proyecto para poder importar nuestro código
directorio_de_este_archivo = os.path.dirname(os.path.abspath(__file__))
directorio_raiz_del_proyecto = os.path.dirname(
    os.path.dirname(directorio_de_este_archivo)
)
sys.path.append(directorio_raiz_del_proyecto)

###############################################################################
# Importar nuestros módulos (funciones de predicción, carga de modelos, etc.)
###############################################################################
from src.utils.cargar_modelos_s import predecir, colonos, biopsias
from src.config.settings import settings
import base64

ruta_fondo = os.path.join(settings.CARPETA_IMAGENES_UI, "fondo1.png")
if os.path.exists(ruta_fondo):
    with open(ruta_fondo, "rb") as fondo_file:
        encoded_string = base64.b64encode(fondo_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

###############################################################################
# Cargar datos de pacientes
###############################################################################
# Estos dos CSVs se cargan al inicio porque los necesitamos para buscar pacientes

# CSV con los datos clínicos y nivel de riesgo de cada paciente
datos_clinicos_dataframe = pd.read_csv(settings.CSV_RISK_CLEAN_PATH)

# CSV con los 5000 pacientes sintéticos (tienen DNI, NUSS, nombre, apellidos...)
# Usamos este para buscar pacientes por DNI o NUSS y resolver su Patient_ID
if os.path.exists(settings.CSV_PACIENTES_5000_PATH):
    pacientes_5000_dataframe = pd.read_csv(settings.CSV_PACIENTES_5000_PATH)
else:
    pacientes_5000_dataframe = None


###############################################################################
# Cargar modelo ML (LightGBM) — Se carga una sola vez gracias a @st.cache_resource
###############################################################################
@st.cache_resource
def cargar_modelo_ml():
    """
    Carga el modelo LightGBM desde el archivo .pkl.
    @st.cache_resource hace que solo se cargue la primera vez.
    Las siguientes veces que Streamlit recargue la página, reutiliza el mismo modelo.
    """
    import joblib

    if not os.path.exists(settings.MODEL_ML_PATH):
        st.error(f"No se encuentra el modelo ML en: {settings.MODEL_ML_PATH}")
        return None
    try:
        modelo = joblib.load(settings.MODEL_ML_PATH)
        print(f"[OK] Modelo ML cargado: {type(modelo).__name__}")
        return modelo
    except Exception as error:
        st.error(f"Error al cargar el modelo ML: {error}")
        return None


# Cargar el modelo ML al arrancar la app
modelo_inteligencia_artificial = cargar_modelo_ml()


###############################################################################
# Funciones auxiliares de búsqueda de pacientes
###############################################################################


def detectar_tipo_de_busqueda(texto_introducido):
    """
    Detecta si el texto que el usuario ha escrito es un DNI o un NUSS.

    Reglas:
        - DNI: 8 números + 1 letra (ejemplo: 12345678A)
        - NUSS: 10 o más números seguidos (ejemplo: 1234567890)

    Devuelve:
        "DNI", "NUSS" o None si no reconoce el formato.
    """
    if not texto_introducido:
        return None
    texto_limpio = str(texto_introducido).strip().upper().replace(" ", "")

    # Comprobar si es un DNI: 8 dígitos + 1 letra al final
    if (
        len(texto_limpio) == 9
        and texto_limpio[:-1].isdigit()
        and texto_limpio[-1].isalpha()
    ):
        return "DNI"

    # Comprobar si es un NUSS: 10 o más dígitos seguidos
    if texto_limpio.isdigit() and len(texto_limpio) >= 10:
        return "NUSS"

    return None


def buscar_patient_id_por_dni_o_nuss(texto_introducido):
    """
    Busca el Patient_ID correspondiente a un DNI o NUSS en el CSV de 5000 pacientes.

    Devuelve:
        (patient_id, fila_completa_del_paciente) si lo encuentra
        (None, None) si no lo encuentra
    """
    if not texto_introducido:
        return None, None

    tipo_busqueda = detectar_tipo_de_busqueda(texto_introducido)
    texto_limpio = str(texto_introducido).strip().upper()

    if tipo_busqueda is None:
        return None, None

    # Buscamos en el CSV de 5000 pacientes
    if pacientes_5000_dataframe is not None:
        if tipo_busqueda == "DNI":
            # Buscamos por la columna "dni"
            filas_encontradas = pacientes_5000_dataframe[
                pacientes_5000_dataframe["dni"].astype(str).str.upper() == texto_limpio
            ]
        else:
            # Buscamos por la columna "nuss"
            filas_encontradas = pacientes_5000_dataframe[
                pacientes_5000_dataframe["nuss"].astype(str).str.zfill(12)
                == texto_limpio.zfill(12)
            ]

        if filas_encontradas.empty:
            return None, None

        # Tomamos la primera fila encontrada
        fila_del_paciente = filas_encontradas.iloc[0]
        return int(fila_del_paciente["Patient_ID"]), fila_del_paciente

    return None, None


###############################################################################
# Interfaz principal de streamlit
###############################################################################
# Logo y Título Centrado
col1, col2, col3 = st.columns([1, 1, 0.7])
with col1:
    st.image(
        os.path.join(directorio_raiz_del_proyecto, "static", "galeno.png"),
        use_container_width=True,
    )

with col2:
    st.markdown(
        """
        <div style="text-align: left; padding-bottom: 2rem;">
        <h1 class="main-title">Galeno</h1>
        <p class="main-subtitle">Sistema Inteligente de Diagnóstico y Prevención de Cáncer de Colon</p>
    </div>
""",
        unsafe_allow_html=True,
    )

# Las dos pestañas principales de la aplicación
pestana_datos, pestana_vision = st.tabs(["Análisis de Datos", "Visión por Computadora"])


###############################################################################
# Pestaña 1: Análisis de datos clínicos
###############################################################################
with pestana_datos:
    # Fila superior: Búsqueda + Info + Imagen del paciente
    columna_busqueda, columna_info_paciente, columna_imagen = st.columns([2, 2, 1])

    with columna_busqueda:
        st.subheader("Búsqueda")
        texto_buscar = st.text_input(
            "Búsqueda por DNI o NUSS",
            placeholder="Escribe un DNI (12345678A) o NUSS (1234567890...)",
        )
        boton_cargar = st.button("Cargar Datos", use_container_width=True)
        boton_actualizar = st.button("Actualizar Paciente", use_container_width=True)

    # Inicializar el estado del formulario en session_state
    # session_state es la "memoria" de Streamlit entre recargas de página.
    # Sin esto, al pulsar un botón todos los campos se resetearían a cero.
    if "form_data" not in st.session_state:
        st.session_state.form_data = {
            "age": 0,
            "gender": "Male",
            "smoking": 0,
            "alcohol_use": 0,
            "obesity": 0,
            "family_history": False,
            "diet_red_meat": 0,
            "diet_salted_processed": 0,
            "fruit_veg_intake": 0,
            "physical_activity": 0,
            "bmi": 0.0,
            "overall_risk_score": 0.0,
            "risk_level": "Low",
            "risk_level_n": 0,
        }
    # Sincronizar las claves del widget con los valores guardados
    for clave, valor in st.session_state.form_data.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor

    if "patient_info" not in st.session_state:
        st.session_state.patient_info = None

    # Lógica del botón "CARGAR DATOS"
    if boton_cargar:
        patient_id, fila_paciente_5000 = buscar_patient_id_por_dni_o_nuss(texto_buscar)
        # Buscamos ese Patient_ID en el CSV de datos clínicos
        paciente_encontrado = (
            datos_clinicos_dataframe[
                datos_clinicos_dataframe["Patient_ID"] == patient_id
            ]
            if patient_id is not None
            else pd.DataFrame()
        )

        if paciente_encontrado.empty:
            st.error(" No se encontró ningún paciente con ese DNI o NUSS.")
            st.session_state.patient_info = None
        else:
            # Extraemos los datos del paciente encontrado
            paciente = paciente_encontrado.iloc[0]

            # Mapeamos las columnas del CSV a los campos del formulario
            edad = int(paciente.get("Age", 0))
            genero_numerico = int(paciente.get("Gender", 0))
            genero_texto = "Male" if genero_numerico == 0 else "Female"
            nivel_tabaquismo = int(paciente.get("Smoking", 0))
            consumo_alcohol = int(paciente.get("Alcohol_Use", 0))
            nivel_obesidad = int(paciente.get("Obesity", 0))
            tiene_historial_familiar = bool(paciente.get("Family_History", 0))
            consumo_carne_roja = int(paciente.get("Diet_Red_Meat", 0))
            consumo_sal_procesados = int(paciente.get("Diet_Salted_Processed", 0))
            consumo_fruta_verdura = int(paciente.get("Fruit_Veg_Intake", 0))
            nivel_actividad_fisica = int(paciente.get("Physical_Activity", 0))
            indice_masa_corporal = float(paciente.get("BMI", 0) or 0)
            puntuacion_riesgo_global = float(paciente.get("Overall_Risk_Score", 0) or 0)
            nivel_riesgo_texto = str(paciente.get("Risk_Level", "Low"))
            nivel_riesgo_numerico = int(paciente.get("Risk_Level_n", 0))

            # Guardamos todos los datos en el session_state
            st.session_state.form_data.update(
                {
                    "age": edad,
                    "gender": genero_texto,
                    "smoking": nivel_tabaquismo,
                    "alcohol_use": consumo_alcohol,
                    "obesity": nivel_obesidad,
                    "family_history": tiene_historial_familiar,
                    "diet_red_meat": consumo_carne_roja,
                    "diet_salted_processed": consumo_sal_procesados,
                    "fruit_veg_intake": consumo_fruta_verdura,
                    "physical_activity": nivel_actividad_fisica,
                    "bmi": indice_masa_corporal,
                    "overall_risk_score": puntuacion_riesgo_global,
                    "risk_level": nivel_riesgo_texto,
                    "risk_level_n": nivel_riesgo_numerico,
                }
            )
            # Sincronizar con los widgets de Streamlit
            for clave, valor in st.session_state.form_data.items():
                st.session_state[clave] = valor

            # Guardar la info del paciente para mostrarla en la columna derecha
            info_del_paciente = {
                "risk_level": nivel_riesgo_texto,
                "overall_risk_score": puntuacion_riesgo_global,
            }
            # Si encontramos datos personales en el CSV de 5000, los añadimos
            if fila_paciente_5000 is not None:
                info_del_paciente.update(
                    {
                        "dni": str(fila_paciente_5000.get("dni", "")).strip(),
                        "nuss": str(fila_paciente_5000.get("nuss", "")).strip(),
                        "nombre": " ".join(
                            [
                                str(fila_paciente_5000.get("nombre", "")).strip(),
                                str(fila_paciente_5000.get("apellido1", "")).strip(),
                                str(fila_paciente_5000.get("apellido2", "")).strip(),
                            ]
                        ).strip(),
                    }
                )
            st.session_state.patient_info = info_del_paciente

    # Mostrar información del paciente cargado
    with columna_info_paciente:
        st.markdown("**Datos de la persona**")
        info_paciente = st.session_state.patient_info
        if info_paciente:
            if info_paciente.get("nombre"):
                st.markdown(f"Nombre: `{info_paciente['nombre']}`")
            if info_paciente.get("dni") or info_paciente.get("nuss"):
                st.markdown(f"DNI: `{info_paciente.get('dni', '')}`")
                st.markdown(f"NUSS: `{info_paciente.get('nuss', '')}`")
            st.markdown(f"Risk Level: `{info_paciente['risk_level']}`")
        else:
            st.markdown("_Sin paciente cargado._")

    with columna_imagen:
        st.markdown("**Imagen**")
        st.markdown(
            """
            <div style="border: 2px dashed #e2e8f0; border-radius: 12px; height: 220px;
                 display: flex; align-items: center; justify-content: center;
                 color: #94a3b8; background: #f8fafc;">
                Sin imagen
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Formulario de datos clínicos
    st.markdown("**Demografía y antropometría**")
    columna_edad, columna_genero, columna_bmi = st.columns(3)
    input_edad = columna_edad.number_input(
        "Age", min_value=0, max_value=120, step=1, key="age"
    )
    input_genero = columna_genero.selectbox(
        "Gender",
        ["Male", "Female"],
        index=0 if st.session_state.gender == "Male" else 1,
        key="gender",
    )
    input_bmi = columna_bmi.number_input(
        "BMI", min_value=0.0, max_value=60.0, step=0.1, key="bmi"
    )

    st.markdown("**Hábitos y dieta (0-10)**")
    fila1_col1, fila1_col2, fila1_col3 = st.columns(3)
    input_tabaquismo = fila1_col1.number_input(
        "Smoking", min_value=0, max_value=10, step=1, key="smoking"
    )
    input_alcohol = fila1_col2.number_input(
        "Alcohol_Use", min_value=0, max_value=10, step=1, key="alcohol_use"
    )
    input_obesidad = fila1_col3.number_input(
        "Obesity", min_value=0, max_value=10, step=1, key="obesity"
    )

    fila2_col1, fila2_col2, fila2_col3 = st.columns(3)
    input_carne_roja = fila2_col1.number_input(
        "Diet_Red_Meat", min_value=0, max_value=10, step=1, key="diet_red_meat"
    )
    input_sal_procesados = fila2_col2.number_input(
        "Diet_Salted_Processed",
        min_value=0,
        max_value=10,
        step=1,
        key="diet_salted_processed",
    )
    input_fruta_verdura = fila2_col3.number_input(
        "Fruit_Veg_Intake", min_value=0, max_value=10, step=1, key="fruit_veg_intake"
    )

    fila3_col1, fila3_col2, fila3_col3 = st.columns(3)
    input_actividad_fisica = fila3_col1.number_input(
        "Physical_Activity", min_value=0, max_value=10, step=1, key="physical_activity"
    )
    fila3_col2.markdown("")
    fila3_col3.markdown("")

    st.markdown("**Factores y resultado**")
    fila4_col1, fila4_col2, fila4_col3 = st.columns(3)
    input_historial_familiar = fila4_col1.checkbox(
        "Family_History", key="family_history"
    )
    fila4_col2.markdown("")
    input_nivel_riesgo = fila4_col3.selectbox(
        "Risk_Level",
        ["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(st.session_state.risk_level),
        key="risk_level",
    )

    fila5_col1, fila5_col2, fila5_col3 = st.columns(3)
    input_riesgo_global = fila5_col1.number_input(
        "Overall_Risk_Score",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="overall_risk_score",
    )
    # Calcular el valor numérico del nivel de riesgo (Low=0, Medium=1, High=2)
    riesgo_numerico_calculado = {"Low": 0, "Medium": 1, "High": 2}.get(
        input_nivel_riesgo, 0
    )
    fila5_col2.metric("Risk_Level_n", riesgo_numerico_calculado)
    fila5_col3.markdown("")

    # Lógica del botón "Actualizar Paciente"
    if boton_actualizar:
        if not texto_buscar:
            st.error("Introduce un DNI o NUSS para actualizar.")
        else:
            dataframe_actualizar = pd.read_csv(settings.CSV_RISK_CLEAN_PATH)
            patient_id, _ = buscar_patient_id_por_dni_o_nuss(texto_buscar)
            mascara = (
                dataframe_actualizar["Patient_ID"] == patient_id
                if patient_id is not None
                else pd.Series([False] * len(dataframe_actualizar))
            )

            if not mascara.any():
                st.error(" No se encontró ningún paciente para actualizar.")
            else:
                indice_fila = dataframe_actualizar[mascara].index[0]

                # Función auxiliar para actualizar una columna si existe en el CSV
                def actualizar_columna(nombre_columna, nuevo_valor):
                    if nombre_columna in dataframe_actualizar.columns:
                        dataframe_actualizar.at[indice_fila, nombre_columna] = (
                            nuevo_valor
                        )

                # Actualizar cada campo con los valores del formulario
                actualizar_columna("Age", int(input_edad))
                actualizar_columna("Gender", 0 if input_genero == "Male" else 1)
                actualizar_columna("Smoking", int(input_tabaquismo))
                actualizar_columna("Alcohol_Use", int(input_alcohol))
                actualizar_columna("Obesity", int(input_obesidad))
                actualizar_columna(
                    "Family_History", 1 if input_historial_familiar else 0
                )
                actualizar_columna("Diet_Red_Meat", int(input_carne_roja))
                actualizar_columna("Diet_Salted_Processed", int(input_sal_procesados))
                actualizar_columna("Fruit_Veg_Intake", int(input_fruta_verdura))
                actualizar_columna("Physical_Activity", int(input_actividad_fisica))
                actualizar_columna("BMI", float(input_bmi))
                actualizar_columna("Overall_Risk_Score", float(input_riesgo_global))
                actualizar_columna("Risk_Level", input_nivel_riesgo)
                actualizar_columna("Risk_Level_n", int(riesgo_numerico_calculado))

                # Guardar el CSV actualizado
                dataframe_actualizar.to_csv(settings.CSV_RISK_CLEAN_PATH, index=False)
                datos_clinicos_dataframe = dataframe_actualizar

                if st.session_state.patient_info:
                    st.session_state.patient_info.update(
                        {
                            "risk_level": input_nivel_riesgo,
                            "overall_risk_score": float(input_riesgo_global),
                        }
                    )
                st.success(" Paciente actualizado correctamente.")

    st.divider()

    # Botones de acción (Calcular Riesgo + Guardar)
    columna_boton_calcular, columna_boton_guardar = st.columns(2)

    # Botón "Calcular Riesgo" (Predicción con ML + SHAP)
    # Esta es la funcionalidad que venía de app_s.py
    if columna_boton_calcular.button(
        "CALCULAR RIESGO (IA)", type="primary", use_container_width=True
    ):
        if modelo_inteligencia_artificial is None:
            st.error(
                "El modelo ML no se ha podido cargar. Revisa que el archivo .pkl exista."
            )
        else:
            # Convertimos los valores del formulario a lo que el modelo entiende
            valor_fumador = input_tabaquismo  # Ya es 0-10
            valor_sangre_oculta = (
                "Negativo"  # Por defecto (este CSV no tiene FOBT directo)
            )
            valor_historial = input_historial_familiar

            # Llamamos a la función de predicción que muestra el informe HTML + SHAP
            predecir(
                modelo_inteligencia_artificial,
                texto_buscar or "Desconocido",  # Identificador del paciente
                "Sí" if input_tabaquismo > 5 else "No",  # Fumador Sí/No
                float(input_alcohol),  # Alcohol (0-10)
                input_historial_familiar,  # Herencia familiar (True/False)
                float(input_carne_roja),  # Carne roja (0-10)
                float(input_sal_procesados),  # Sal/procesados (0-10)
                float(input_fruta_verdura),  # Fruta/verdura (0-10)
                float(input_actividad_fisica),  # Actividad física (0-10)
                float(input_bmi),  # BMI
                valor_sangre_oculta,  # FOBT (Negativo/Positivo)
                0.0,  # CEA (por defecto 0 si no disponible)
            )

    # ─── BOTÓN "GUARDAR REGISTRO" ───
    if columna_boton_guardar.button("GUARDAR REGISTRO", use_container_width=True):
        if not texto_buscar:
            st.error("Introduce un DNI o NUSS para guardar.")
        else:
            patient_id, _ = buscar_patient_id_por_dni_o_nuss(texto_buscar)
            if patient_id is None:
                st.error("No se pudo resolver el Patient_ID desde ese DNI/NUSS.")
            else:
                dataframe_guardar = pd.read_csv(settings.CSV_RISK_CLEAN_PATH)
                if (dataframe_guardar["Patient_ID"] == patient_id).any():
                    st.error("Ya existe ese Patient_ID. Usa 'Actualizar Paciente'.")
                else:
                    nuevo_registro = {
                        "Patient_ID": int(patient_id),
                        "Age": int(input_edad),
                        "Gender": 0 if input_genero == "Male" else 1,
                        "Smoking": int(input_tabaquismo),
                        "Alcohol_Use": int(input_alcohol),
                        "Obesity": int(input_obesidad),
                        "Family_History": 1 if input_historial_familiar else 0,
                        "Diet_Red_Meat": int(input_carne_roja),
                        "Diet_Salted_Processed": int(input_sal_procesados),
                        "Fruit_Veg_Intake": int(input_fruta_verdura),
                        "Physical_Activity": int(input_actividad_fisica),
                        "BMI": float(input_bmi),
                        "Overall_Risk_Score": float(input_riesgo_global),
                        "Risk_Level": input_nivel_riesgo,
                        "Risk_Level_n": int(riesgo_numerico_calculado),
                    }
                    dataframe_guardar = pd.concat(
                        [dataframe_guardar, pd.DataFrame([nuevo_registro])],
                        ignore_index=True,
                    )
                    dataframe_guardar.to_csv(settings.CSV_RISK_CLEAN_PATH, index=False)
                    datos_clinicos_dataframe = dataframe_guardar
                    st.session_state.patient_info = {
                        "id": int(patient_id),
                        "risk_level": input_nivel_riesgo,
                        "overall_risk_score": float(input_riesgo_global),
                    }
                    st.success("Registro guardado correctamente.")

    # Botón "Ver Resumen"
    st.divider()
    if st.button("Ver Resumen", use_container_width=True):
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #d7e6f7;
                 padding: 16px; border-radius: 12px;">
                <h4 style="margin: 0; color: #0b1f35;">Resumen de Riesgo</h4>
                <p style="margin: 6px 0;">Risk Level: <strong>{input_nivel_riesgo}</strong></p>
                <p style="margin: 6px 0;">BMI: <strong>{input_bmi}</strong></p>
                <p style="margin: 6px 0;">Risk Score: <strong>{input_riesgo_global}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


################################################################################
# PESTAÑA 2: Análisis de Imágenes
################################################################################
with pestana_vision:
    st.subheader("Análisis de Diagnóstico por Imagen")

    # Radio button para elegir el tipo de estudio (viene de app_s.py)
    tipo_de_analisis = st.radio(
        "Seleccione el tipo de estudio:",
        [
            "Captura de Colonoscopia (Localizar Pólipos)",
            "Muestra de Biopsia (Malignidad)",
        ],
        horizontal=True,
    )
    st.divider()

    # Dos columnas: izquierda = subir imagen, derecha = resultado
    columna_subir_imagen, columna_resultado_imagen = st.columns(2)

    with columna_subir_imagen:
        # Cambiar el texto del uploader según el tipo de análisis
        if "Colonoscopia" in tipo_de_analisis:
            texto_uploader = "Subir captura de Colonoscopia"
        else:
            texto_uploader = "Subir microfotografía de Biopsia"

        archivo_imagen = st.file_uploader(texto_uploader, type=["jpg", "png", "jpeg"])

    # Solo si el usuario ha subido una imagen
    if archivo_imagen:
        # Convertimos el archivo subido a un array de numpy (formato que entiende OpenCV y los modelos)
        imagen_pil = Image.open(archivo_imagen)
        imagen_como_array = np.array(imagen_pil)

        if st.button("ANALIZAR IMAGEN", type="primary"):
            # ─── ANÁLISIS DE COLONOSCOPIA (TensorFlow + Grad-CAM) ───
            if "Colonoscopia" in tipo_de_analisis:
                with st.spinner("Analizando pólipos con TensorFlow..."):
                    texto_resultado_html, imagen_gradcam = colonos(imagen_como_array)

                    # Redimensionamos la original para que coincida con el Grad-CAM (150x150)
                    imagen_original_redimensionada = cv2.resize(
                        imagen_como_array, (150, 150)
                    )

                with columna_resultado_imagen:
                    # Mostramos original y Grad-CAM lado a lado
                    subcolumna_original, subcolumna_gradcam = st.columns(2)
                    with subcolumna_original:
                        st.image(
                            imagen_original_redimensionada,
                            caption="Captura Original",
                            use_container_width=True,
                        )
                    with subcolumna_gradcam:
                        st.image(
                            imagen_gradcam,
                            caption="Localización (Grad-CAM)",
                            use_container_width=True,
                        )
                    # El informe (porcentaje y recomendación) aparece debajo de las fotos
                    st.markdown(texto_resultado_html, unsafe_allow_html=True)

            # ANÁLISIS DE BIOPSIA (PyTorch + Grad-CAM)
            else:
                with st.spinner("Analizando malignidad con PyTorch + Grad-CAM..."):
                    texto_resultado_html, imagen_heatmap, imagen_biopsia_original = (
                        biopsias(imagen_como_array)
                    )

                with columna_resultado_imagen:
                    subcolumna_original, subcolumna_heatmap = st.columns(2)
                    with subcolumna_original:
                        st.image(
                            imagen_biopsia_original,
                            caption="Biopsia Original",
                            use_container_width=True,
                        )
                    with subcolumna_heatmap:
                        st.image(
                            imagen_heatmap,
                            caption="Mapa de Calor (Grad-CAM)",
                            use_container_width=True,
                        )
                    st.markdown(texto_resultado_html, unsafe_allow_html=True)
