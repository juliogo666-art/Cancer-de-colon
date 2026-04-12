"""
Aplicación Frontend (Streamlit) de Galeno.

Versión Final:
    - Desacoplada: No carga modelos localmente, usa peticiones HTTP REST a FastAPI.
    - Internacionalizada: Lee `translations.yaml` para soportar Español/Inglés en caliente.
    - Trazabilidad: Envía el `patient_id` a la API para loguear las predicciones.
    - Código humanamente legible: Variables en castellano y comentarios extensos.

Arrancar con:
    streamlit run src/frontend/app.py
"""

import streamlit as st
import os
import sys
import yaml
import requests
import pandas as pd
import base64

# =============================================================================
# Configuración Inicial y Carga de Rutas
# =============================================================================

# Definimos las rutas relativas para poder importar la configuración general
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.dirname(os.path.dirname(directorio_actual))
sys.path.append(directorio_raiz)

from src.config.settings import settings

# URL base de nuestra API (FastAPI) a la que le haremos las peticiones HTTP
API_BASE_URL = "http://localhost:8000/api/v1"

# =============================================================================
# Carga de Idiomas (Internacionalización)
# =============================================================================


def cargar_traducciones():
    """Lee el archivo translations.yaml y devuelve los diccionarios de texto."""
    ruta_yaml = os.path.join(directorio_raiz, "src", "config", "translations.yaml")
    with open(ruta_yaml, "r", encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


# Guardamos las traducciones en memoria
traducciones = cargar_traducciones()


# =============================================================================
# Selector de Idiomas Dinámico
# =============================================================================

col_espacio, col_idioma = st.columns([6, 1])
with col_idioma:
    idioma_seleccionado = st.selectbox(
        "Idioma", ["es", "en"], label_visibility="collapsed"
    )

# Extraemos el diccionario de textos correspondiente al idioma elegido
textos = traducciones[idioma_seleccionado]

# =============================================================================
# Configuración de la Página de Streamlit
# =============================================================================

st.set_page_config(
    page_title="Galeno - Cáncer de Colon",
    page_icon="⚕️",
    layout="wide",
)

# Estilo visual (Blanco y Verde Clínico)
st.markdown(
    """
<style>
    .stApp { background-color: #FFFFFF; }
    div.stButton > button:first-child {
        background-color: #22c55e; color: white; border: none;
        border-radius: 8px; padding: 0.6rem 1.2rem; font-weight: 600;
        transition: all 0.3s ease; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div.stButton > button:first-child:hover {
        background-color: #16a34a; color: white;
        transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(34, 197, 94, 0.3);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #22c55e 0%, #10b981 100%);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: #f8fafc; padding: 0 20px; border-radius: 12px 12px 0 0; }
    .stTabs [data-baseweb="tab"] { height: 50px; color: #64748b; }
    .stTabs [aria-selected="true"] { color: #22c55e !important; border-bottom-color: #22c55e !important; font-weight: bold; }
    .stTextInput>div>div>input, .stSelectbox>div>div>div { border-radius: 8px; border: 1px solid #e2e8f0; }
    .stTextInput>div>div>input:focus { border-color: #22c55e; box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2); }
    h1, h2, h3, h4, [data-testid="stHeader"] { color: #0f172a !important; font-family: 'Inter', sans-serif !important; }
    .main-title { color: #0f172a; font-size: 3rem !important; font-weight: 800 !important; margin-bottom: 0.5rem !important; }
    .main-subtitle { color: #64748b; font-size: 1.1rem; margin-bottom: 2rem; }
    label, [data-testid="stWidgetLabel"] p, [data-testid="stRadio"] label p { color: #1e293b !important; font-weight: 600 !important; font-size: 1rem !important; }
    [data-testid="stRadio"] div[role="radiogroup"] { background-color: #f8fafc; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    [data-testid="stMetric"] { background-color: #f1f5f9; padding: 15px; border-radius: 12px; border-left: 5px solid #22c55e; }
</style>
""",
    unsafe_allow_html=True,
)

# Imagen de fondo si existe
ruta_fondo = os.path.join(settings.CARPETA_IMAGENES_UI, "fondo1.png")
if os.path.exists(ruta_fondo):
    with open(ruta_fondo, "rb") as fondo_file:
        cadena_codificada = base64.b64encode(fondo_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{cadena_codificada});
            background-size: cover; background-attachment: fixed; background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Lógica de Datos y Búsqueda
# =============================================================================

# Cargar los datasets en memoria (CSVs en crudo) para permitir las búsquedas
df_master = pd.read_csv(settings.CSV_MASTER_PATH)

def detectar_formato_busqueda(entrada_texto):
    """Detecta si el texto es DNI (9 caracteres) o NUSS (10+ números)."""
    if not entrada_texto:
        return None
    limpieza = str(entrada_texto).strip().upper().replace(" ", "")

    if len(limpieza) == 9 and limpieza[:-1].isdigit() and limpieza[-1].isalpha():
        return "DNI"
    if limpieza.isdigit() and len(limpieza) >= 10:
        return "NUSS"
    return None


def buscar_paciente_por_documento(entrada_texto):
    """Retorna el (Patient_ID, fila_datos) buscando en el CSV consolidado."""
    if not entrada_texto:
        return None, None

    tipo = detectar_formato_busqueda(entrada_texto)
    limpieza = str(entrada_texto).strip().upper()

    if tipo == "DNI":
        coincidencias = df_master[
            df_master["dni"].astype(str).str.upper() == limpieza
        ]
    elif tipo == "NUSS":
        coincidencias = df_master[
            df_master["nuss"].astype(str).str.zfill(12) == limpieza.zfill(12)
        ]
    else:
        return None, None

    if coincidencias.empty:
        return None, None

    fila_encontrada = coincidencias.iloc[0]
    return int(fila_encontrada["Patient_ID"]), fila_encontrada


# =============================================================================
# Interfaz de Usuario Principal
# =============================================================================

col_logo, col_titulos, _ = st.columns([1, 1.5, 0.5])
with col_logo:
    st.image(
        os.path.join(directorio_raiz, "static", "galeno.png"), use_container_width=True
    )

with col_titulos:
    st.markdown(
        f"""
        <div style="text-align: left; padding-bottom: 2rem;">
        <h1 class="main-title">{textos["title"]}</h1>
        <p class="main-subtitle">{textos["subtitle"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Pestañas principales
pestana_datos, pestana_vision = st.tabs([textos["tab_data"], textos["tab_vision"]])


# #############################################################################
# Pestaña 1: Análisis de Datos Clínicos
# #############################################################################
with pestana_datos:
    # --- Cabecera: Búsqueda e Información del Paciente ---
    col_buscar, col_info, col_img = st.columns([2, 2, 1])

    with col_buscar:
        st.subheader(textos["search_header"])
        texto_buscar = st.text_input(textos["search_placeholder"], key="txt_search")
        btn_cargar = st.button(textos["btn_load"], use_container_width=True)
        btn_actualizar = st.button(textos["btn_update"], use_container_width=True)

    # Memoria de la aplicación (Session State) para no perder los datos del formulario al recargar
    if "form_datos_paciente" not in st.session_state:
        st.session_state.form_datos_paciente = {
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
            "patient_id": None,
        }
    if "info_sesion_paciente" not in st.session_state:
        st.session_state.info_sesion_paciente = None

    # Bloque 1: El botón de CARGAR DATOS
    if btn_cargar:
        id_paciente, info_paciente = buscar_paciente_por_documento(texto_buscar)

        if info_paciente is None:
            st.error(textos["error_not_found"])
            st.session_state.info_sesion_paciente = None
        else:
            fila_paciente = info_paciente
            # Mapeo de datos para actualizar el formulario visual
            st.session_state.form_datos_paciente.update(
                {
                    "patient_id": id_paciente,
                    "age": int(fila_paciente.get("Age", 0)),
                    "gender": str(fila_paciente.get("Gender", "Male")),
                    "smoking": int(fila_paciente.get("Smoking", 0)),

                    "alcohol_use": int(fila_paciente.get("Alcohol_Use", 0)),
                    "obesity": int(fila_paciente.get("Obesity", 0)),
                    "family_history": bool(fila_paciente.get("Family_History", 0)),
                    "diet_red_meat": int(fila_paciente.get("Diet_Red_Meat", 0)),
                    "diet_salted_processed": int(
                        fila_paciente.get("Diet_Salted_Processed", 0)
                    ),
                    "fruit_veg_intake": int(fila_paciente.get("Fruit_Veg_Intake", 0)),
                    "physical_activity": int(fila_paciente.get("Physical_Activity", 0)),
                    "bmi": float(fila_paciente.get("BMI", 0)),
                    "overall_risk_score": float(
                        fila_paciente.get("Overall_Risk_Score", 0)
                    ),
                    "risk_level": str(fila_paciente.get("Risk_Level", "Low")),
                    "risk_level_n": int(fila_paciente.get("Risk_Level_n", 0)),
                }
            )

            # Guardamos la info básica en sesión para mostrar el recuadro
            info_resumen = {
                "risk_level": fila_paciente.get("Risk_Level", "Low"),
                "patient_id": id_paciente,
            }

            if info_paciente is not None:
                nombres = str(info_paciente.get("nombre", "")).strip()
                ape1 = str(info_paciente.get("apellido1", "")).strip()
                ape2 = str(info_paciente.get("apellido2", "")).strip()
                info_resumen["nombre"] = f"{nombres} {ape1} {ape2}".strip()
                info_resumen["dni"] = str(info_paciente.get("dni", "")).strip()
                info_resumen["nuss"] = str(info_paciente.get("nuss", "")).strip()

            st.session_state.info_sesion_paciente = info_resumen

    # --- Bloque Visual: Ficha del Paciente (Mitad Superior) ---
    with col_info:
        st.markdown(f"**{textos['info_header']}**")
        datos_paciente = st.session_state.info_sesion_paciente
        if datos_paciente:
            # Reporte visual bonito y legible en lugar de texto pequeño
            html_info = '<div style="background-color: #f1f5f9; border-left: 5px solid #22c55e; padding: 15px; border-radius: 10px; color: #0f172a; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">'
            if "nombre" in datos_paciente:
                html_info += f"<p style='font-size: 1.15em; margin: 4px 0;'><b>{textos['info_name']}:</b> {datos_paciente['nombre']}</p>"
            if "dni" in datos_paciente:
                html_info += f"<p style='font-size: 1.15em; margin: 4px 0;'><b>{textos['info_dni']}:</b> {datos_paciente['dni']}</p>"
            if "nuss" in datos_paciente:
                html_info += f"<p style='font-size: 1.15em; margin: 4px 0;'><b>{textos['info_nuss']}:</b> {datos_paciente['nuss']}</p>"

            # Color del nivel de riesgo según su valor
            color_riesgo = (
                "#22c55e"
                if datos_paciente["risk_level"] == "Low"
                else "#eab308"
                if datos_paciente["risk_level"] == "Medium"
                else "#ef4444"
            )
            html_info += f"<p style='font-size: 1.15em; margin: 4px 0; margin-top: 10px; border-top: 1px solid #cbd5e1; padding-top: 10px;'><b>{textos['info_risk']}:</b> <span style='color: {color_riesgo}; font-weight: 800;'>{datos_paciente['risk_level']}</span></p>"
            html_info += "</div>"

            st.markdown(html_info, unsafe_allow_html=True)
        else:
            st.markdown(textos["info_empty"])

    with col_img:
        st.markdown(f"**{textos['img_header']}**")
        st.markdown(
            f"""<div style="border: 2px dashed #e2e8f0; border-radius: 12px; height: 180px; 
                             display: flex; align-items: center; justify-content: center; color: #94a3b8; background: #f8fafc;">
                        {textos["img_empty"]}
                        </div>""",
            unsafe_allow_html=True,
        )
    st.divider()

    # --- ZONA DE FORMULARIO (Datos Médicos Modificables) ---
    st.markdown(f"**{textos['demo_header']}**")
    c_edad, c_genero, c_bmi = st.columns(3)
    val_edad = c_edad.number_input(
        textos["age"],
        min_value=0,
        max_value=120,
        step=1,
        value=st.session_state.form_datos_paciente["age"],
    )
    val_genero = c_genero.selectbox(
        textos["gender"],
        [textos["gender_m"], textos["gender_f"]],
        index=0 if st.session_state.form_datos_paciente["gender"] == "Male" else 1,
    )
    val_bmi = c_bmi.number_input(
        textos["bmi"],
        min_value=0.0,
        max_value=60.0,
        step=0.1,
        value=st.session_state.form_datos_paciente["bmi"],
    )

    st.markdown(f"**{textos['habits_header']}**")
    c_hab1, c_hab2, c_hab3 = st.columns(3)
    val_Fumar = c_hab1.number_input(
        textos["smoking"],
        min_value=0,
        max_value=10,
        value=st.session_state.form_datos_paciente["smoking"],
    )
    val_Alcohol = c_hab2.number_input(
        textos["alcohol"],
        min_value=0,
        max_value=10,
        value=st.session_state.form_datos_paciente["alcohol_use"],
    )
    val_Obesidad = c_hab3.number_input(
        textos["obesity"],
        min_value=0,
        max_value=10,
        value=st.session_state.form_datos_paciente["obesity"],
    )

    c_die1, c_die2, c_die3 = st.columns(3)
    val_carne_roja = c_die1.number_input(
        textos["meat"],
        min_value=0,
        max_value=10,
        value=st.session_state.form_datos_paciente["diet_red_meat"],
    )
    val_salados = c_die2.number_input(
        textos["salt"],
        min_value=0,
        max_value=10,
        value=st.session_state.form_datos_paciente["diet_salted_processed"],
    )
    val_fruta = c_die3.number_input(
        textos["fruits"],
        min_value=0,
        max_value=10,
        value=st.session_state.form_datos_paciente["fruit_veg_intake"],
    )

    c_act1, c_act2, c_act3 = st.columns(3)
    val_actividad = c_act1.number_input(
        textos["activity"],
        min_value=0,
        max_value=10,
        value=st.session_state.form_datos_paciente["physical_activity"],
    )

    st.markdown(f"**{textos['factors_header']}**")
    c_fac1, c_fac2, c_fac3 = st.columns(3)
    val_familia = c_fac1.checkbox(
        textos["family"], value=st.session_state.form_datos_paciente["family_history"]
    )

    opciones_riesgo = [textos["risk_low"], textos["risk_medium"], textos["risk_high"]]
    idx_riesgo_defecto = 0
    if st.session_state.form_datos_paciente["risk_level"] == "Medium":
        idx_riesgo_defecto = 1
    elif st.session_state.form_datos_paciente["risk_level"] == "High":
        idx_riesgo_defecto = 2

    val_nivel_riesgo = c_fac3.selectbox(
        textos["risk_lbl"], opciones_riesgo, index=idx_riesgo_defecto
    )

    c_res1, c_res2, c_res3 = st.columns(3)
    val_riesgo_global = c_res1.number_input(
        textos["risk_score"],
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=st.session_state.form_datos_paciente["overall_risk_score"],
    )

    # Calcular numérico interno escondido
    diccionario_niveles = {
        textos["risk_low"]: 0,
        textos["risk_medium"]: 1,
        textos["risk_high"]: 2,
    }
    val_riesgo_numerico = diccionario_niveles.get(val_nivel_riesgo, 0)
    c_res2.metric("Risk Level n.", val_riesgo_numerico)

    st.divider()

    # ####################################################
    # Boton actualizar paciente
    # ####################################################
    if btn_actualizar:
        if not texto_buscar:
            st.error(textos["error_no_dni"])
        else:
            id_paciente, _ = buscar_paciente_por_documento(texto_buscar)
            mascara = (
                df_master["Patient_ID"] == id_paciente
                if id_paciente
                else pd.Series([False] * len(df_master))
            )

            if not mascara.any():
                st.error(textos["error_not_found"])
            else:
                idx = df_master[mascara].index[0]
                df_master.at[idx, "Age"] = val_edad
                df_master.at[idx, "Gender"] = (
                    "Male" if val_genero == textos["gender_m"] else "Female"
                )
                df_master.at[idx, "Smoking"] = val_Fumar
                df_master.at[idx, "Alcohol_Use"] = val_Alcohol
                df_master.at[idx, "Obesity"] = val_Obesidad
                df_master.at[idx, "Family_History"] = 1 if val_familia else 0
                df_master.at[idx, "Diet_Red_Meat"] = val_carne_roja
                df_master.at[idx, "Diet_Salted_Processed"] = val_salados
                df_master.at[idx, "Fruit_Veg_Intake"] = val_fruta
                df_master.at[idx, "Physical_Activity"] = val_actividad
                df_master.at[idx, "BMI"] = val_bmi
                df_master.at[idx, "Overall_Risk_Score"] = val_riesgo_global
                df_master.at[idx, "Risk_Level"] = ["Low", "Medium", "High"][
                    val_riesgo_numerico
                ]
                df_master.at[idx, "Risk_Level_n"] = val_riesgo_numerico

                df_master.to_csv(settings.CSV_MASTER_PATH, index=False)
                st.success(textos["success_update"])

    # =========================================================================
    # BOTONES PRINCIPALES: CALCULAR (VÍA API HTTP) Y GUARDAR
    # =========================================================================
    col_btn_calc, col_btn_guardar = st.columns(2)

    # Llamada a la API de predicción de riesgo
    if col_btn_calc.button(
        textos["btn_calc"], type="primary", use_container_width=True
    ):
        # 1. Empaquetamos los datos en formato diccionario (JSON / Query params)
        parametros_api = {
            "patient_id": st.session_state.form_datos_paciente.get("patient_id") or 0,
            "smoking": val_Fumar,
            "alcohol_use": val_Alcohol,
            "obesity": val_Obesidad,
            "family_history": 1 if val_familia else 0,
            "diet_red_meat": val_carne_roja,
            "diet_salted_processed": val_salados,
            "fruit_veg_intake": val_fruta,
            "physical_activity": val_actividad,
            "bmi": val_bmi,
            "fobt_resultado": 0,  # Asumimos Default
            "cea_level": 0.0,  # Asumimos Default
        }

        # 2. Hacemos la llamada HTTP a nuestra API de FastAPI que está sirviendo los modelos
        try:
            respuesta_api = requests.post(
                f"{API_BASE_URL}/predict/risk",
                params=parametros_api,
                timeout=10,  # Tiempo máximo de espera 10s
            )

            if respuesta_api.status_code == 200:
                # 3. El servidor nos contesta con el resultado en JSON ¡Procesado inmediatamente!
                datos_respuesta = respuesta_api.json()
                nivel_riesgo_devuelto = datos_respuesta.get("risk_level", "Unknown")
                score_riesgo_devuelto = datos_respuesta.get("risk_score", 0.0)
                probabilidades = datos_respuesta.get("probabilities", {})

                st.success(
                    f"Análisis completado. Nivel estimado: {nivel_riesgo_devuelto}"
                )

                # Armamos un reporte visual moderno
                st.markdown(
                    f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px; margin-top: 15px;">
                    <h3 style="margin-top: 0; color: #1e293b;">{textos["summary_header"]}</h3>
                    <p><b>{textos["risk_lbl"]}:</b> {nivel_riesgo_devuelto}</p>
                    <p><b>{textos["risk_score"]}:</b> {score_riesgo_devuelto}</p>
                    <hr style="margin: 10px 0; border: none; border-top: 1px solid #e2e8f0;">
                    <p style="margin: 0; color: #64748b; font-size: 0.9em;">
                    Probabilidades — Low: {probabilidades.get("Low", 0) * 100:.1f}% | 
                    Medium: {probabilidades.get("Medium", 0) * 100:.1f}% | 
                    High: {probabilidades.get("High", 0) * 100:.1f}%
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(
                    f"Error HTTP {respuesta_api.status_code}: {respuesta_api.text}"
                )

        except requests.exceptions.RequestException:
            # Si el servidor no contesta, significa que el alumno olvidó levantar 'main.py api'
            st.error(textos["api_error"])

    # BOTÓN GUARDAR (AÑADIR NUEVO A CSV)
    if col_btn_guardar.button(textos["btn_save"], use_container_width=True):
        if not texto_buscar:
            st.error(textos["error_no_dni"])
        else:
            id_paciente, _ = buscar_paciente_por_documento(texto_buscar)
            if id_paciente is not None:
                if (df_master["Patient_ID"] == id_paciente).any():
                    st.error(textos["error_exists"])
                else:
                    nuevo_registro = {
                        "Patient_ID": id_paciente,
                        "Age": val_edad,
                        "Gender": "Male" if val_genero == textos["gender_m"] else "Female",
                        "Smoking": val_Fumar,
                        "Alcohol_Use": val_Alcohol,
                        "Obesity": val_Obesidad,
                        "Family_History": 1 if val_familia else 0,
                        "Diet_Red_Meat": val_carne_roja,
                        "Diet_Salted_Processed": val_salados,
                        "Fruit_Veg_Intake": val_fruta,
                        "Physical_Activity": val_actividad,
                        "BMI": val_bmi,
                        "Overall_Risk_Score": val_riesgo_global,
                        "Risk_Level": ["Low", "Medium", "High"][val_riesgo_numerico],
                        "Risk_Level_n": val_riesgo_numerico,
                    }
                    df_master = pd.concat(
                        [df_master, pd.DataFrame([nuevo_registro])],
                        ignore_index=True,
                    )
                    df_master.to_csv(settings.CSV_MASTER_PATH, index=False)
                    st.success(textos["success_save"])


# #############################################################################
# Pestaña 2: Visión por Computadora (Colonoscopias y Biopsias)
# #############################################################################
with pestana_vision:
    st.subheader(textos["vision_header"])

    # Selector de modelo
    tipo_estudio = st.radio(
        textos["vision_select"],
        [textos["vision_opt1"], textos["vision_opt2"]],
        horizontal=True,
    )
    st.divider()

    col_subir, col_resultados = st.columns(2)

    with col_subir:
        texto_subir = (
            textos["upload_opt1"]
            if tipo_estudio == textos["vision_opt1"]
            else textos["upload_opt2"]
        )
        archivo_subido = st.file_uploader(texto_subir, type=["jpg", "png", "jpeg"])

    if archivo_subido:
        # LLAMADA A LA API DE IMÁGENES (Petición Multimedia)
        if st.button(textos["btn_analyze"], type="primary"):
            spin_texto = (
                textos["analyzing_opt1"]
                if tipo_estudio == textos["vision_opt1"]
                else textos["analyzing_opt2"]
            )
            url_endpoint = (
                f"{API_BASE_URL}/analyze/colonoscopy"
                if tipo_estudio == textos["vision_opt1"]
                else f"{API_BASE_URL}/analyze/biopsy"
            )

            with st.spinner(spin_texto):
                try:
                    # Empaquetamos la imagen en el Payload HTTP
                    archivos_http = {
                        "file": (
                            archivo_subido.name,
                            archivo_subido.getvalue(),
                            archivo_subido.type,
                        )
                    }

                    parametros_extra = {}
                    if st.session_state.form_datos_paciente.get("patient_id"):
                        parametros_extra["patient_id"] = (
                            st.session_state.form_datos_paciente["patient_id"]
                        )

                    # Lanzamos el cohete hacia FastAPI
                    respuesta_imagen = requests.post(
                        url_endpoint,
                        files=archivos_http,
                        params=parametros_extra,
                        timeout=60,
                    )

                    if respuesta_imagen.status_code == 200:
                        datos_json = respuesta_imagen.json()
                        diagnostico_final = datos_json.get("diagnosis", "")
                        confianza_final = datos_json.get("confidence", 0.0)
                        recomendacion = datos_json.get("recommendation", "")
                        base64_heatmap = datos_json.get("gradcam_base64")

                        # Mostramos resultados visuales en la columna derecha
                        with col_resultados:
                            sub_col1, sub_col2 = st.columns(2)

                            with sub_col1:
                                st.image(
                                    archivo_subido,
                                    caption=textos["img_orig"],
                                    use_container_width=True,
                                )

                            with sub_col2:
                                if base64_heatmap:
                                    # Extraemos y decodificamos el mapa de calor que nos generó la API remotamente
                                    imagen_bytes = base64.b64decode(base64_heatmap)
                                    st.image(
                                        imagen_bytes,
                                        caption=textos["img_grad"],
                                        use_container_width=True,
                                    )
                                else:
                                    st.info("Heatmap no disponible desde la API.")

                            # Reporte HTML Bonito
                            color_alerta = (
                                "#22c55e"
                                if "SANO" in diagnostico_final
                                or "BENIGNO" in diagnostico_final
                                else "#ef4444"
                            )
                            st.markdown(
                                f"""
                            <div style="background-color: #f8fafc; border-left: 5px solid {color_alerta}; padding: 15px; border-radius: 8px; margin-top: 20px;">
                                <h3 style="margin-top: 0; color: #1e293b;">{diagnostico_final}</h3>
                                <p style="font-size: 1.1em;">Confianza: <b>{confianza_final * 100:.1f}%</b></p>
                                <p style="margin-bottom: 0;"><i>{recomendacion}</i></p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                    else:
                        st.error(
                            f"Error HTTP {respuesta_imagen.status_code}: {respuesta_imagen.text}"
                        )

                except requests.exceptions.RequestException:
                    st.error(textos["api_error"])
