import streamlit as st
import joblib
import os
import sys
import pandas as pd



# Cargar csv (búsqueda por DNI o ID)
CSV_PATH = "src/data/raw/historial_pacientes/datos_combinados_global_extendido_3.csv"
df = pd.read_csv(CSV_PATH)


# Configuración de página (debe ser lo primero)
st.set_page_config(page_title="ColonAI - Sistema Integral", layout="wide")

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# RUTAS
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.dirname(os.path.dirname(directorio_actual))
sys.path.append(directorio_raiz)

from src.utils.cargar_modelos_s import predecir, colonos, obtener_modelo_cnn
from src.utils.data_load_s import save_r

MODEL_ML_PATH = os.path.join(directorio_raiz, 'src', 'models', 'ml', 'best_rf_model.pkl')

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
    col_busqueda, col_info, col_img = st.columns([2, 2, 1])
    with col_busqueda:
        st.subheader("Búsqueda")
        texto_buscar = st.text_input("Búsqueda por ID, DNI o NUSS")
        btn_cargar = st.button("Cargar Datos", use_container_width=True)
        btn_actualizar = st.button("Actualizar Paciente", use_container_width=True)

    # Inicializar valores en session_state para que sean mutables por los botones
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            "edad": 0, "genero": "Masculino", "estadio": 1, "tumor": 0.0,
            "sangre": "No", "cea": 0.0, "fuma": "No", "alc": 0.0,
            "diab": "No", "fam": False, "ibd": False, "peso": 0.0, "altura": 0.0
        }
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = None

    # Lógica de botones de carga/limpieza
    def _detectar_tipo_busqueda(texto):
        if not texto:
            return "ID"
        t = str(texto).strip().upper().replace(" ", "")
        # DNI: 8 dígitos + letra. NIE: X/Y/Z + 7 dígitos + letra
        if (
            (len(t) == 9 and t[:-1].isdigit() and t[-1].isalpha())
            or (len(t) == 9 and t[0] in ["X", "Y", "Z"] and t[1:8].isdigit() and t[-1].isalpha())
        ):
            return "DNI"
        # NUSS: solo dígitos (normalmente 10-11)
        if t.isdigit():
            return "NUSS"
        return "ID"

    def _buscar_paciente_por_id_o_dni(texto, tipo):
        if not texto:
            return None

        if tipo == "ID":
            pid_txt = str(texto).strip()
            fila = df[df["Patient_ID"].astype(str) == pid_txt]
        elif tipo == "DNI":
            dni_txt = str(texto).strip().upper()
            fila = df[df["dni"].astype(str).str.upper() == dni_txt]
        else:
            nuss_txt = str(texto).strip()
            fila = df[df["nuss"].astype(str) == nuss_txt]

        if fila.empty:
            return None
        return fila.iloc[0]

    def _parse_estadio(valor):
        if valor is None:
            return 1
        try:
            estadio_int = int(valor)
            return estadio_int if estadio_int in [1, 2, 3, 4] else 1
        except Exception:
            pass

        texto = str(valor).strip().upper()
        if texto.startswith("STAGE"):
            texto = texto.replace("STAGE", "").strip()

        mapa = {"I": 1, "II": 2, "III": 3, "IV": 4}
        return mapa.get(texto, 1)

    if btn_cargar:
        tipo_busqueda = _detectar_tipo_busqueda(texto_buscar)
        paciente = _buscar_paciente_por_id_o_dni(texto_buscar, tipo_busqueda)
        if paciente is None:
            st.error("No se encontró ningún paciente con ese ID, DNI o NUSS.")
            st.session_state.patient_info = None
        else:
            # Mapear columnas del CSV extendido a los campos del formulario
            edad = float(paciente.get("Age", 0))
            gender_raw = str(paciente.get("Gender", "")).strip().upper()
            gender_n = paciente.get("Gender_n", None)
            if gender_raw in ["MALE", "MASCULINO", "M", "1"]:
                genero = "Masculino"
            elif gender_raw in ["FEMALE", "FEMENINO", "F", "0"]:
                genero = "Femenino"
            elif gender_n in [0, "0"]:
                genero = "Masculino"
            elif gender_n in [1, "1"]:
                genero = "Femenino"
            else:
                genero = "Masculino"
            estadio = _parse_estadio(paciente.get("Cancer_Stage", 1))
            tumor = float(paciente.get("Tumor_Size_mm", 0)) if "Tumor_Size_mm" in paciente else 0.0
            fam = bool(paciente.get("Antecedentes_Familiares", 0))
            fuma = "Sí" if int(paciente.get("Smoking", 0)) == 1 else "No"
            alc = float(paciente.get("Alcohol_Use", paciente.get("Alcohol_Consumption", 0)) or 0)
            diab = "Sí" if int(paciente.get("Diabetes_tipo_2", paciente.get("Diabetes", 0)) or 0) == 1 else "No"
            ibd = bool(paciente.get("Enfermedad_Inflamatoria_Intestinal", 0))
            sangre = "Sí" if int(paciente.get("FOBT_Resultado_n", 0)) == 1 else "No"
            cea = float(paciente.get("CEA_Level_ng_mL..Marcador.Tumoral.", 0) or 0)
            altura = float(paciente.get("altura_cm", 0) or 0)
            peso = float(paciente.get("peso_kg", 0) or 0)

            st.session_state.form_data.update({
                "edad": edad,
                "genero": genero,
                "estadio": estadio,
                "tumor": tumor,
                "fam": fam,
                "fuma": fuma,
                "alc": alc,
                "diab": diab,
                "ibd": ibd,
                "sangre": sangre,
                "cea": cea,
                "altura": altura,
                "peso": peso,
            })

            nombre = str(paciente.get("nombre", "")).strip()
            apellido1 = str(paciente.get("apellido1", "")).strip()
            apellido2 = str(paciente.get("apellido2", "")).strip()
            nombre_completo = " ".join([p for p in [nombre, apellido1, apellido2] if p])
            st.session_state.patient_info = {
                "nombre": nombre_completo if nombre_completo else "Sin nombre",
                "edad": int(edad),
                "dni": str(paciente.get("dni", "")).strip(),
                "nuss": str(paciente.get("nuss", "")).strip(),
                "peso": float(peso),
                "altura": float(altura),
            }


    with col_info:
        st.markdown("**Datos de la persona**")
        info = st.session_state.patient_info
        if info:
            st.markdown(f"Nombre: `{info['nombre']}`")
            st.markdown(f"Edad: `{info['edad']}`")
            st.markdown(f"DNI: `{info['dni']}`")
            st.markdown(f"NUSS: `{info['nuss']}`")
            st.markdown(f"Peso: `{info['peso']}` kg")
            st.markdown(f"Altura: `{info['altura']}` cm")
        else:
            st.markdown("_Sin paciente cargado._")
    with col_img:
        st.markdown("**Imagen**")
        st.markdown(
            """
            <div style="border: 2px dashed #444; border-radius: 12px; height: 220px; display: flex; align-items: center; justify-content: center; color: #888; background: #111;">
                Sin imagen
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

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

    if btn_actualizar:
        if not texto_buscar:
            st.error("Introduce un ID, DNI o NUSS para actualizar.")
        else:
            tipo_busqueda = _detectar_tipo_busqueda(texto_buscar)
            df_update = pd.read_csv(CSV_PATH)
            if tipo_busqueda == "ID":
                mask = df_update["Patient_ID"].astype(str) == str(texto_buscar).strip()
            elif tipo_busqueda == "DNI":
                mask = df_update["dni"].astype(str).str.upper() == str(texto_buscar).strip().upper()
            else:
                mask = df_update["nuss"].astype(str) == str(texto_buscar).strip()

            if not mask.any():
                st.error("No se encontró ningún paciente para actualizar.")
            else:
                idx = df_update[mask].index[0]

                def _set_col(col, val):
                    if col in df_update.columns:
                        df_update.at[idx, col] = val

                _set_col("Age", int(in_edad))
                if in_genero == "Masculino":
                    _set_col("Gender", "Male")
                    _set_col("Gender_n", 0)
                else:
                    _set_col("Gender", "Female")
                    _set_col("Gender_n", 1)
                _set_col("Cancer_Stage", int(in_estadio))
                _set_col("Tumor_Size_mm", float(in_tumor))
                _set_col("FOBT_Resultado_n", 1 if in_sangre == "SÃ­" else 0)
                _set_col("CEA_Level_ng_mL..Marcador.Tumoral.", float(in_cea))
                _set_col("Smoking", 1 if in_fuma == "SÃ­" else 0)
                _set_col("Alcohol_Use", float(in_alc))
                _set_col("Diabetes_tipo_2", 1 if in_diab == "SÃ­" else 0)
                _set_col("Antecedentes_Familiares", 1 if in_fam else 0)
                _set_col("Enfermedad_Inflamatoria_Intestinal", 1 if in_ibd else 0)
                _set_col("altura_cm", float(in_altura))
                _set_col("peso_kg", float(in_peso))

                df_update.to_csv(CSV_PATH, index=False)
                df = df_update

                if st.session_state.patient_info:
                    st.session_state.patient_info.update({
                        "edad": int(in_edad),
                        "peso": float(in_peso),
                        "altura": float(in_altura),
                    })
                st.success("Paciente actualizado correctamente.")

    st.divider()
    
    c_btn1, c_btn2 = st.columns(2)
    if c_btn1.button("CALCULAR RIESGO", type="primary", use_container_width=True):
        selector = texto_buscar if texto_buscar else "Sin ID/DNI/NUSS"
        res_html = predecir(modelo_ia, selector, in_edad, in_genero, in_estadio, in_tumor, in_sangre, in_cea, in_fuma, in_alc, in_diab, in_fam, in_ibd, in_peso, in_altura)
        st.markdown(res_html, unsafe_allow_html=True)

    if c_btn2.button("GUARDAR REGISTRO", use_container_width=True):
        selector = texto_buscar if texto_buscar else "Sin ID/DNI/NUSS"
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
