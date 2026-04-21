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

# IMPORTANTE: set_page_config debe ser la primera llamada a Streamlit.
# Si se coloca despues de otros widgets (columns, selectbox...),
# Streamlit lanza un error o warning.
st.set_page_config(
    page_title="Galeno - Cancer de Colon",
    page_icon="",
    layout="wide",
)

# ======= SISTEMA DE APAGADO AUTOMÁTICO =======
import streamlit.components.v1 as components

components.html(
    """
    <script>
    // Heartbeat: Envía un pulso a la API local cada 2 segundos
    // Si el navegador se cierra, el pulso dejará de llegar y main.py matará los procesos.
    setInterval(function(){
        fetch('http://localhost:8000/api/v1/heartbeat', { method: 'GET', keepalive: true }).catch(e => {});
    }, 2000);
    </script>
    """,
    height=0,
    width=0,
)

import os
import sys
import yaml
import requests
import pandas as pd
import base64

import matplotlib.pyplot as plt
import pickle
import joblib

# =============================================================================
# Configuración Inicial y Carga de Rutas
# =============================================================================

# Definimos las rutas relativas para poder importar la configuración general
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.dirname(os.path.dirname(directorio_actual))
sys.path.append(directorio_raiz)

from src.config.settings import settings
from src.utils.gradcam_utils import generar_explicacion_shap

# =============================================================================
# Cargar modelo ensemble para SHAP (solo una vez)
# Usamos joblib.load y la ruta centralizada en `settings` para evitar errores
# al deserializar objetos sklearn/XGBoost/LightGBM.
# =============================================================================
try:
    modelo_path = settings.MODEL_ML_FINAL_PATH
    if os.path.exists(modelo_path):
        modelo_ensemble = joblib.load(modelo_path)
    else:
        modelo_ensemble = None
        print(f"No se encontró el fichero del ensemble SHAP en: {modelo_path}")
except Exception as e:
    modelo_ensemble = None
    print(f"No se pudo cargar el modelo ensemble para SHAP: {e}")

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
# Estilos Visuales (CSS)
# =============================================================================

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
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: #0f172a; padding: 0 20px; border-radius: 12px 12px 0 0; }
    .stTabs [data-baseweb="tab"] { height: 50px; color: #cbd5e1; }
    .stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom-color: #22c55e !important; background-color: #1e293b !important; font-weight: bold; border-radius: 8px 8px 0 0; }
    .stTextInput>div>div>input, .stSelectbox>div>div>div { border-radius: 8px; border: 1px solid #e2e8f0; }
    .stTextInput>div>div>input:focus { border-color: #22c55e; box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2); }
    [data-testid="stFileUploaderFileName"] { color: #0f172a !important; }
    [data-testid="stFileUploaderFile"] { color: #0f172a !important; }
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
        coincidencias = df_master[df_master["dni"].astype(str).str.upper() == limpieza]
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
        btn_nuevo = st.button("Nuevo Paciente", use_container_width=True)

    # Memoria de la aplicación (Session State) para no perder los datos del formulario al recargar
    if "modo_nuevo_paciente" not in st.session_state:
        st.session_state.modo_nuevo_paciente = False

    if "form_datos_paciente" not in st.session_state:
        st.session_state.form_datos_paciente = {
            "age": 0,
            "gender": "Male",
            "altura_cm": 170.0,
            "peso_kg": 70.0,
            "smoking": 0,
            "alcohol_use": 0,
            "obesity": 0,
            "family_history": False,
            "diet_red_meat": 0,
            "diet_salted_processed": 0,
            "fruit_veg_intake": 0,
            "physical_activity": 0,
            "bmi": 0.0,
            "fobt_resultado_n": -1,
            "cea_level_ng_ml": -1.0,
            "overall_risk_score": 0.0,
            "risk_level": "Low",
            "risk_level_n": 0,
            "patient_id": None,
        }
        st.session_state.info_sesion_paciente = None

    if btn_nuevo:
        st.session_state.modo_nuevo_paciente = True
        st.session_state.info_sesion_paciente = None
        # Resetear campos
        st.session_state.form_datos_paciente.update(
            {
                "age": 0,
                "gender": "Male",
                "altura_cm": 170.0,
                "peso_kg": 70.0,
                "smoking": 0,
                "alcohol_use": 0,
                "obesity": 0,
                "family_history": False,
                "diet_red_meat": 0,
                "diet_salted_processed": 0,
                "fruit_veg_intake": 0,
                "physical_activity": 0,
                "bmi": 0.0,
                "fobt_resultado_n": -1,
                "cea_level_ng_ml": -1.0,
                "overall_risk_score": 0.0,
                "risk_level": "Low",
                "risk_level_n": 0,
                "patient_id": None,
            }
        )

    # Bloque 1: El botón de CARGAR DATOS
    if btn_cargar:
        st.session_state.modo_nuevo_paciente = False
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
                    "altura_cm": float(fila_paciente.get("altura_cm", 170.0)),
                    "peso_kg": float(fila_paciente.get("peso_kg", 70.0)),
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
                    "fobt_resultado_n": int(fila_paciente.get("FOBT_Resultado_n", 0)),
                    "cea_level_ng_ml": float(fila_paciente.get("CEA_Level_ng_mL", 0.0)),
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

        if st.session_state.modo_nuevo_paciente:
            st.info("Modo de Inscripción Continua: Rellene los datos de identidad.")
            st.text_input("Documento (DNI/NIE)", key="form_new_dni")
            st.text_input("NUSS / Seguridad Social", key="form_new_nuss")
            st.text_input("Nombre", key="form_new_nombre")
            ext1, ext2 = st.columns(2)
            ext1.text_input("Primer Apellido", key="form_new_ape1")
            ext2.text_input("Segundo Apellido", key="form_new_ape2")
            loc1, loc2 = st.columns(2)
            loc1.text_input("Ciudad", key="form_new_city")
            loc2.text_input("País", key="form_new_country")

        elif datos_paciente:
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
    c_edad, c_genero, c_altura, c_peso = st.columns(4)
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
    val_altura = c_altura.number_input(
        "Altura (cm)",
        min_value=50.0,
        max_value=250.0,
        step=1.0,
        value=float(st.session_state.form_datos_paciente.get("altura_cm", 170.0)),
    )
    val_peso = c_peso.number_input(
        "Peso (kg)",
        min_value=20.0,
        max_value=300.0,
        step=1.0,
        value=float(st.session_state.form_datos_paciente.get("peso_kg", 70.0)),
    )

    # Calculamos el BMI automáticamente y no permitimos que el usuario lo edite
    # ya que es redundante y puede dar lugar a errores de cálculo si no coinciden.
    val_bmi = 0.0
    if val_altura > 0:
        val_bmi = round(val_peso / ((val_altura / 100.0) ** 2), 2)

    st.markdown(f"**{textos['habits_header']}**")
    c_hab1, c_hab2, c_hab3 = st.columns(3)
    valor_tabaquismo = c_hab1.number_input(
        textos["smoking"],
        min_value=0,
        max_value=10,
        value=st.session_state.form_datos_paciente["smoking"],
    )
    valor_consumo_alcohol = c_hab2.number_input(
        textos["alcohol"],
        min_value=0,
        max_value=10,
        value=st.session_state.form_datos_paciente["alcohol_use"],
    )
    valor_nivel_obesidad = c_hab3.number_input(
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

    st.markdown("**Analíticas Confirmadas (Dejar en Blanco si Triaje)**")
    c_ana1, c_ana2 = st.columns(2)

    fobt_opts = ["Desconocido (-)", "Negativo", "Positivo"]
    fobt_opt_idx = 0
    if st.session_state.form_datos_paciente["fobt_resultado_n"] == 0:
        fobt_opt_idx = 1
    elif st.session_state.form_datos_paciente["fobt_resultado_n"] == 1:
        fobt_opt_idx = 2

    val_fobt_ui = c_ana1.selectbox(
        "Prueba sangre oculta heces (FOBT)", fobt_opts, index=fobt_opt_idx
    )
    # Re-mapeo visual a valor numérico
    if val_fobt_ui == "Negativo":
        st.session_state.form_datos_paciente["fobt_resultado_n"] = 0
    elif val_fobt_ui == "Positivo":
        st.session_state.form_datos_paciente["fobt_resultado_n"] = 1
    else:
        st.session_state.form_datos_paciente["fobt_resultado_n"] = -1

    cea_input_str = ""
    if st.session_state.form_datos_paciente["cea_level_ng_ml"] >= 0:
        cea_input_str = str(st.session_state.form_datos_paciente["cea_level_ng_ml"])

    val_cea_ui = c_ana2.text_input(
        "Marcador tumoral CEA (ng/mL)",
        value=cea_input_str,
        placeholder="Ej: 3.42 (Dejar vacío si no hay)",
    )

    try:
        val_cea_num = float(val_cea_ui)
        st.session_state.form_datos_paciente["cea_level_ng_ml"] = val_cea_num
    except ValueError:
        st.session_state.form_datos_paciente["cea_level_ng_ml"] = -1.0

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
                df_master.at[idx, "Smoking"] = valor_tabaquismo
                df_master.at[idx, "Alcohol_Use"] = valor_consumo_alcohol
                df_master.at[idx, "Obesity"] = valor_nivel_obesidad
                df_master.at[idx, "Family_History"] = 1 if val_familia else 0
                df_master.at[idx, "Diet_Red_Meat"] = val_carne_roja
                df_master.at[idx, "Diet_Salted_Processed"] = val_salados
                df_master.at[idx, "Fruit_Veg_Intake"] = val_fruta
                df_master.at[idx, "Physical_Activity"] = val_actividad
                df_master.at[idx, "BMI"] = val_bmi
                # No actualizamos manualmente Overall_Risk_Score ni Risk_Level_n
                # porque eso ahora solo se hace a través del botón "Calcular" (IA).

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
            "smoking": valor_tabaquismo,
            "alcohol_use": valor_consumo_alcohol,
            "obesity": valor_nivel_obesidad,
            "family_history": 1 if val_familia else 0,
            "diet_red_meat": val_carne_roja,
            "diet_salted_processed": val_salados,
            "fruit_veg_intake": val_fruta,
            "physical_activity": val_actividad,
            "bmi": val_bmi,
            "fobt_resultado": st.session_state.form_datos_paciente.get(
                "fobt_resultado_n", 0
            ),
            "cea_level": st.session_state.form_datos_paciente.get(
                "cea_level_ng_ml", 0.0
            ),
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
                recomendacion = datos_respuesta.get("recommendation", "")

                # Actualizar el riesgo en el state de manera oculta
                st.session_state.form_datos_paciente["risk_level"] = (
                    nivel_riesgo_devuelto
                )
                st.session_state.form_datos_paciente["overall_risk_score"] = (
                    score_riesgo_devuelto
                )

                st.success(
                    f"Análisis completado. Nivel estimado: {nivel_riesgo_devuelto}"
                )

                # Armamos un reporte visual moderno
                st.markdown(
                    f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px; margin-top: 15px;">
                    <h3 style="margin-top: 0; color: #1e293b;">{textos["summary_header"]}</h3>
                    <p style="color: #0f172a; font-size: 1.1em;"><b>{textos["risk_lbl"]}:</b> {nivel_riesgo_devuelto}</p>
                    <p style="color: #0f172a; font-size: 1.1em;"><b>{textos["risk_score"]}:</b> {score_riesgo_devuelto}</p>
                    <p style="color: #b91c1c; font-size: 1.1em; background-color: #fef2f2; padding: 10px; border-radius: 5px;">{recomendacion}</p>
                    <hr style="margin: 10px 0; border: none; border-top: 1px solid #e2e8f0;">
                    <p style="margin: 0; color: #475569; font-size: 1em;">
                    <b>Probabilidades</b> — Low: {probabilidades.get("Low", 0) * 100:.1f}% | 
                    Medium: {probabilidades.get("Medium", 0) * 100:.1f}% | 
                    High: {probabilidades.get("High", 0) * 100:.1f}%
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # --- SHAP: Explicación del riesgo ---
                if modelo_ensemble is not None:
                    # Construir el array de features en el orden correcto
                    features_array = [
                        valor_tabaquismo,
                        valor_consumo_alcohol,
                        valor_nivel_obesidad,
                        1 if val_familia else 0,
                        val_carne_roja,
                        val_salados,
                        val_fruta,
                        val_actividad,
                        val_bmi,
                        st.session_state.form_datos_paciente.get("fobt_resultado_n", 0),
                        st.session_state.form_datos_paciente.get(
                            "cea_level_ng_ml", 0.0
                        ),
                    ]
                    # Determinar la clase predicha como índice
                    clase_map = {"Low": 0, "Medium": 1, "High": 2}
                    clase_predicha = clase_map.get(nivel_riesgo_devuelto, 0)
                    fig, df_importancia = generar_explicacion_shap(
                        modelo_ensemble, [features_array], clase_predicha
                    )
                    if fig is not None and df_importancia is not None:
                        st.markdown("### Explicacion del modelo (SHAP)")
                        st.pyplot(fig)
                        st.write("Importancia de cada variable en tu predicción:")
                        st.dataframe(df_importancia)
                    else:
                        st.warning(
                            "No se pudo generar la explicación SHAP para este caso."
                        )
                else:
                    st.info("El modelo ensemble para SHAP no está disponible.")

            else:
                st.error(
                    f"Error HTTP {respuesta_api.status_code}: {respuesta_api.text}"
                )

        except requests.exceptions.RequestException:
            # Si el servidor no contesta, significa que el alumno olvidó levantar 'main.py api'
            st.error(textos["api_error"])

    # BOTÓN GUARDAR (AÑADIR NUEVO A CSV)
    if col_btn_guardar.button(textos["btn_save"], use_container_width=True):
        if st.session_state.modo_nuevo_paciente:
            nuevo_id = (
                int(df_master["Patient_ID"].max() + 1) if not df_master.empty else 1
            )
            dni_val = st.session_state.get("form_new_dni", "00000000A")
            nuss_val = st.session_state.get("form_new_nuss", "0")
            nombre_val = st.session_state.get("form_new_nombre", "Desconocido")
            ape1_val = st.session_state.get("form_new_ape1", "")
            ape2_val = st.session_state.get("form_new_ape2", "")
            city_val = st.session_state.get("form_new_city", "Unknown")
            country_val = st.session_state.get("form_new_country", "Unknown")

            nuevo_registro = {
                "Patient_ID": nuevo_id,
                "Gender": "Male" if val_genero == textos["gender_m"] else "Female",
                "City": city_val,
                "Country": country_val,
                "nombre": nombre_val,
                "apellido1": ape1_val,
                "apellido2": ape2_val,
                "Age": val_edad,
                "altura_cm": val_altura,
                "peso_kg": val_peso,
                "dni": dni_val,
                "nuss": nuss_val,
                "Smoking": valor_tabaquismo,
                "Alcohol_Use": valor_consumo_alcohol,
                "Obesity": valor_nivel_obesidad,
                "Family_History": 1 if val_familia else 0,
                "Diet_Red_Meat": val_carne_roja,
                "Diet_Salted_Processed": val_salados,
                "Fruit_Veg_Intake": val_fruta,
                "Physical_Activity": val_actividad,
                "BMI": val_bmi,
                "Overall_Risk_Score": 0.0,
                "Risk_Level": "Low",
                "Risk_Level_n": 0,
                "FOBT_Resultado": {-1: "Desconocido", 0: "Negativo", 1: "Positivo"}.get(
                    st.session_state.form_datos_paciente["fobt_resultado_n"],
                    "Desconocido",
                ),
                "FOBT_Resultado_n": st.session_state.form_datos_paciente[
                    "fobt_resultado_n"
                ],
                "CEA_Level_ng_mL": st.session_state.form_datos_paciente[
                    "cea_level_ng_ml"
                ],
            }
            df_master = pd.concat(
                [df_master, pd.DataFrame([nuevo_registro])], ignore_index=True
            )
            df_master.to_csv(settings.CSV_MASTER_PATH, index=False)
            st.success(
                "¡Paciente creado y guardado permanentemente en la base de datos!"
            )

            # Quitar modo nuevo y auto-cargar la vista solo lectura
            st.session_state.modo_nuevo_paciente = False
            st.session_state.info_sesion_paciente = {
                "risk_level": nuevo_registro["Risk_Level"],
                "patient_id": nuevo_id,
                "nombre": f"{nombre_val} {ape1_val} {ape2_val}",
                "dni": dni_val,
                "nuss": "0",
            }
            st.session_state.form_datos_paciente["patient_id"] = nuevo_id
            st.rerun()

        else:
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
                            "Gender": "Male"
                            if val_genero == textos["gender_m"]
                            else "Female",
                            "Smoking": valor_tabaquismo,
                            "Alcohol_Use": valor_consumo_alcohol,
                            "Obesity": valor_nivel_obesidad,
                            "Family_History": 1 if val_familia else 0,
                            "Diet_Red_Meat": val_carne_roja,
                            "Diet_Salted_Processed": val_salados,
                            "Fruit_Veg_Intake": val_fruta,
                            "Physical_Activity": val_actividad,
                            "BMI": val_bmi,
                            "Overall_Risk_Score": 0.0,
                            "Risk_Level": "Low",
                            "Risk_Level_n": 0,
                            "FOBT_Resultado": {
                                -1: "Desconocido",
                                0: "Negativo",
                                1: "Positivo",
                            }.get(
                                st.session_state.form_datos_paciente[
                                    "fobt_resultado_n"
                                ],
                                "Desconocido",
                            ),
                            "FOBT_Resultado_n": st.session_state.form_datos_paciente[
                                "fobt_resultado_n"
                            ],
                            "CEA_Level_ng_mL": st.session_state.form_datos_paciente[
                                "cea_level_ng_ml"
                            ],
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
                            # Etiqueta resultado (Maligno o benigno + confianza + comentario)
                            st.markdown(
                                f"""
                            <div style="background-color: #f8fafc; border-left: 5px solid {color_alerta}; padding: 15px; border-radius: 8px; margin-top: 20px;">
                                <h3 style="margin-top: 0; color: #1e293b;">{diagnostico_final}</h3>
                                <p style="font-size: 1.1em;color: #1e293b;">Confianza: <b>{confianza_final * 100:.1f}%</b></p>
                                <p style="margin-bottom: 0; color: #1e293b;"><i>{recomendacion}</i></p>
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
