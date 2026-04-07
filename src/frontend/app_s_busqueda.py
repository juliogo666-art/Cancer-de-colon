import streamlit as st
import os
import sys
import pandas as pd



# Cargar csv (nuevo dataset de riesgo)
CSV_PATH = "src/data/raw/historial_pacientes/cancer_risk_final.csv"
PATIENTS_PATH = "src/data/raw/historial_pacientes/nuevos_pacientes_5000.csv"
df = pd.read_csv(CSV_PATH)
patients_df = pd.read_csv(PATIENTS_PATH) if os.path.exists(PATIENTS_PATH) else None


# Configuración de página (debe ser lo primero)
st.set_page_config(page_title="ColonAI - Sistema Integral", layout="wide")

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# RUTAS
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.dirname(os.path.dirname(directorio_actual))
sys.path.append(directorio_raiz)

from src.utils.cargar_modelos_s import colonos, obtener_modelo_cnn

# (El modelo ML anterior ya no se usa con el nuevo CSV)

# --- INTERFAZ STREAMLIT ---
st.title("Sistema Integral ColonAI")

tab1, tab2 = st.tabs(["Análisis de Datos", "Visión por Computadora"])

with tab1:
    col_busqueda, col_info, col_img = st.columns([2, 2, 1])
    with col_busqueda:
        st.subheader("Búsqueda")
        texto_buscar = st.text_input("Búsqueda por Patient_ID, DNI o NUSS")
        btn_cargar = st.button("Cargar Datos", use_container_width=True)
        btn_actualizar = st.button("Actualizar Paciente", use_container_width=True)

    # Inicializar valores en session_state para que sean mutables por los botones
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
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
            "fobt_resultado": "Negativo",
            "fobt_resultado_n": 0,
            "cea_level": 0.0
        }
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = None

    # Lógica de botones de carga/actualización (nuevo CSV)
    def _detectar_tipo_busqueda(texto):
        if not texto:
            return "ID"
        t = str(texto).strip().upper().replace(" ", "")
        if len(t) == 9 and t[:-1].isdigit() and t[-1].isalpha():
            return "DNI"
        if t.isdigit():
            return "NUSS" if len(t) >= 10 else "ID"
        return "ID"

    def _resolver_patient_id(texto):
        if not texto:
            return None, None
        tipo = _detectar_tipo_busqueda(texto)
        t = str(texto).strip().upper()

        if tipo == "ID":
            try:
                return int(t), None
            except ValueError:
                return None, None

        if patients_df is not None:
            if tipo == "DNI":
                fila = patients_df[patients_df["dni"].astype(str).str.upper() == t]
            else:
                fila = patients_df[patients_df["nuss"].astype(str) == t]
            if fila.empty:
                return None, None
            row = fila.iloc[0]
            return int(row["Patient_ID"]), row

        # Fallback si el CSV principal tuviera DNI/NUSS
        if tipo == "DNI" and "dni" in df.columns:
            fila = df[df["dni"].astype(str).str.upper() == t]
            return (int(fila.iloc[0]["Patient_ID"]), fila.iloc[0]) if not fila.empty else (None, None)
        if tipo == "NUSS" and "nuss" in df.columns:
            fila = df[df["nuss"].astype(str) == t]
            return (int(fila.iloc[0]["Patient_ID"]), fila.iloc[0]) if not fila.empty else (None, None)

        return None, None

    if btn_cargar:
        pid, info_row = _resolver_patient_id(texto_buscar)
        paciente = df[df["Patient_ID"] == pid] if pid is not None else pd.DataFrame()
        if paciente.empty:
            st.error("No se encontró ningún paciente con ese Patient_ID, DNI o NUSS.")
            st.session_state.patient_info = None
        else:
            paciente = paciente.iloc[0]
            # Mapear columnas del nuevo CSV a los campos del formulario
            smoking = int(paciente.get("Smoking", 0))
            alcohol_use = int(paciente.get("Alcohol_Use", 0))
            obesity = int(paciente.get("Obesity", 0))
            family_history = bool(paciente.get("Family_History", 0))
            diet_red_meat = int(paciente.get("Diet_Red_Meat", 0))
            diet_salted_processed = int(paciente.get("Diet_Salted_Processed", 0))
            fruit_veg_intake = int(paciente.get("Fruit_Veg_Intake", 0))
            physical_activity = int(paciente.get("Physical_Activity", 0))
            bmi = float(paciente.get("BMI", 0) or 0)
            overall_risk_score = float(paciente.get("Overall_Risk_Score", 0) or 0)
            risk_level = str(paciente.get("Risk_Level", "Low"))
            risk_level_n = int(paciente.get("Risk_Level_n", 0))
            fobt_resultado = str(paciente.get("FOBT_Resultado", "Negativo"))
            fobt_resultado_n = int(paciente.get("FOBT_Resultado_n", 0))
            cea_level = float(paciente.get("CEA_Level_ng_mL", 0) or 0)

            st.session_state.form_data.update({
                "smoking": smoking,
                "alcohol_use": alcohol_use,
                "obesity": obesity,
                "family_history": family_history,
                "diet_red_meat": diet_red_meat,
                "diet_salted_processed": diet_salted_processed,
                "fruit_veg_intake": fruit_veg_intake,
                "physical_activity": physical_activity,
                "bmi": bmi,
                "overall_risk_score": overall_risk_score,
                "risk_level": risk_level,
                "risk_level_n": risk_level_n,
                "fobt_resultado": fobt_resultado,
                "fobt_resultado_n": fobt_resultado_n,
                "cea_level": cea_level,
            })

            patient_info = {
                "id": int(paciente.get("Patient_ID", 0)),
                "risk_level": risk_level,
                "overall_risk_score": overall_risk_score,
                "fobt": fobt_resultado,
                "cea": cea_level,
            }
            if info_row is not None:
                patient_info.update({
                    "dni": str(info_row.get("dni", "")).strip(),
                    "nuss": str(info_row.get("nuss", "")).strip(),
                    "nombre": " ".join([str(info_row.get("nombre", "")).strip(),
                                        str(info_row.get("apellido1", "")).strip(),
                                        str(info_row.get("apellido2", "")).strip()]).strip()
                })
            st.session_state.patient_info = patient_info


    with col_info:
        st.markdown("**Datos de la persona**")
        info = st.session_state.patient_info
        if info:
            st.markdown(f"Patient_ID: `{info['id']}`")
            if info.get("nombre"):
                st.markdown(f"Nombre: `{info['nombre']}`")
            if info.get("dni") or info.get("nuss"):
                st.markdown(f"DNI: `{info.get('dni','')}`")
                st.markdown(f"NUSS: `{info.get('nuss','')}`")
            st.markdown(f"Risk Level: `{info['risk_level']}`")
            st.markdown(f"Overall Risk Score: `{info['overall_risk_score']}`")
            st.markdown(f"FOBT Resultado: `{info['fobt']}`")
            st.markdown(f"CEA Level: `{info['cea']}`")
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

    st.markdown("**Hábitos y dieta (0-10)**")
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    in_smoking = r1_c1.number_input("Smoking", min_value=0, max_value=10, step=1, value=int(st.session_state.form_data["smoking"]))
    in_alcohol_use = r1_c2.number_input("Alcohol_Use", min_value=0, max_value=10, step=1, value=int(st.session_state.form_data["alcohol_use"]))
    in_obesity = r1_c3.number_input("Obesity", min_value=0, max_value=10, step=1, value=int(st.session_state.form_data["obesity"]))

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    in_diet_red_meat = r2_c1.number_input("Diet_Red_Meat", min_value=0, max_value=10, step=1, value=int(st.session_state.form_data["diet_red_meat"]))
    in_diet_salted_processed = r2_c2.number_input("Diet_Salted_Processed", min_value=0, max_value=10, step=1, value=int(st.session_state.form_data["diet_salted_processed"]))
    in_fruit_veg_intake = r2_c3.number_input("Fruit_Veg_Intake", min_value=0, max_value=10, step=1, value=int(st.session_state.form_data["fruit_veg_intake"]))

    r3_c1, r3_c2, r3_c3 = st.columns(3)
    in_physical_activity = r3_c1.number_input("Physical_Activity", min_value=0, max_value=10, step=1, value=int(st.session_state.form_data["physical_activity"]))
    in_bmi = r3_c2.number_input("BMI", min_value=0.0, max_value=60.0, step=0.1, value=float(st.session_state.form_data["bmi"]))
    in_cea_level = r3_c3.number_input("CEA_Level_ng_mL", min_value=0.0, max_value=100.0, step=0.1, value=float(st.session_state.form_data["cea_level"]))

    st.markdown("**Factores y resultado**")
    r4_c1, r4_c2, r4_c3 = st.columns(3)
    in_family_history = r4_c1.checkbox("Family_History", value=st.session_state.form_data["family_history"])
    in_fobt_resultado = r4_c2.radio("FOBT_Resultado", ["Negativo", "Positivo"], index=0 if st.session_state.form_data["fobt_resultado"]=="Negativo" else 1, horizontal=True)
    in_risk_level = r4_c3.selectbox("Risk_Level", ["Low", "Medium", "High"], index=["Low","Medium","High"].index(st.session_state.form_data["risk_level"]))

    r5_c1, r5_c2, r5_c3 = st.columns(3)
    in_overall_risk_score = r5_c1.number_input("Overall_Risk_Score", min_value=0.0, max_value=1.0, step=0.01, value=float(st.session_state.form_data["overall_risk_score"]))
    risk_level_n_val = {"Low": 0, "Medium": 1, "High": 2}.get(in_risk_level, 0)
    fobt_n_val = 1 if in_fobt_resultado == "Positivo" else 0
    r5_c2.metric("Risk_Level_n", risk_level_n_val)
    r5_c3.metric("FOBT_Resultado_n", fobt_n_val)

    if btn_actualizar:
        if not texto_buscar:
            st.error("Introduce un Patient_ID, DNI o NUSS para actualizar.")
        else:
            df_update = pd.read_csv(CSV_PATH)
            pid, _ = _resolver_patient_id(texto_buscar)
            mask = df_update["Patient_ID"] == pid if pid is not None else pd.Series([False] * len(df_update))

            if not mask.any():
                st.error("No se encontró ningún paciente para actualizar.")
            else:
                idx = df_update[mask].index[0]

                def _set_col(col, val):
                    if col in df_update.columns:
                        df_update.at[idx, col] = val

                _set_col("Smoking", int(in_smoking))
                _set_col("Alcohol_Use", int(in_alcohol_use))
                _set_col("Obesity", int(in_obesity))
                _set_col("Family_History", 1 if in_family_history else 0)
                _set_col("Diet_Red_Meat", int(in_diet_red_meat))
                _set_col("Diet_Salted_Processed", int(in_diet_salted_processed))
                _set_col("Fruit_Veg_Intake", int(in_fruit_veg_intake))
                _set_col("Physical_Activity", int(in_physical_activity))
                _set_col("BMI", float(in_bmi))
                _set_col("Overall_Risk_Score", float(in_overall_risk_score))
                _set_col("Risk_Level", in_risk_level)
                _set_col("Risk_Level_n", int(risk_level_n_val))
                _set_col("FOBT_Resultado", in_fobt_resultado)
                _set_col("FOBT_Resultado_n", int(fobt_n_val))
                _set_col("CEA_Level_ng_mL", float(in_cea_level))

                df_update.to_csv(CSV_PATH, index=False)
                df = df_update

                if st.session_state.patient_info:
                    st.session_state.patient_info.update({
                        "risk_level": in_risk_level,
                        "overall_risk_score": float(in_overall_risk_score),
                        "fobt": in_fobt_resultado,
                        "cea": float(in_cea_level),
                    })
                st.success("Paciente actualizado correctamente.")

    st.divider()
    
    c_btn1, c_btn2 = st.columns(2)
    if c_btn1.button("VER RESUMEN", type="primary", use_container_width=True):
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #d7e6f7; padding: 16px; border-radius: 12px;">
                <h4 style="margin: 0; color: #0b1f35;">Resumen de Riesgo</h4>
                <p style="margin: 6px 0;">Risk Level: <strong>{in_risk_level}</strong></p>
                <p style="margin: 6px 0;">Overall Risk Score: <strong>{in_overall_risk_score:.3f}</strong></p>
                <p style="margin: 6px 0;">FOBT: <strong>{in_fobt_resultado}</strong></p>
                <p style="margin: 6px 0;">CEA: <strong>{in_cea_level:.2f}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if c_btn2.button("GUARDAR REGISTRO", use_container_width=True):
        if not texto_buscar:
            st.error("Introduce un Patient_ID, DNI o NUSS para guardar.")
        else:
            pid, _ = _resolver_patient_id(texto_buscar)
            if pid is None:
                st.error("No se pudo resolver el Patient_ID.")
            else:
                df_save = pd.read_csv(CSV_PATH)
                if (df_save["Patient_ID"] == pid).any():
                    st.error("Ya existe ese Patient_ID. Usa 'Actualizar Paciente'.")
                else:
                    new_row = {
                        "Patient_ID": int(pid),
                        "Smoking": int(in_smoking),
                        "Alcohol_Use": int(in_alcohol_use),
                        "Obesity": int(in_obesity),
                        "Family_History": 1 if in_family_history else 0,
                        "Diet_Red_Meat": int(in_diet_red_meat),
                        "Diet_Salted_Processed": int(in_diet_salted_processed),
                        "Fruit_Veg_Intake": int(in_fruit_veg_intake),
                        "Physical_Activity": int(in_physical_activity),
                        "BMI": float(in_bmi),
                        "Overall_Risk_Score": float(in_overall_risk_score),
                        "Risk_Level": in_risk_level,
                        "Risk_Level_n": int(risk_level_n_val),
                        "FOBT_Resultado": in_fobt_resultado,
                        "FOBT_Resultado_n": int(fobt_n_val),
                        "CEA_Level_ng_mL": float(in_cea_level),
                    }
                    df_save = pd.concat([df_save, pd.DataFrame([new_row])], ignore_index=True)
                    df_save.to_csv(CSV_PATH, index=False)
                    df = df_save
                    st.session_state.patient_info = {
                        "id": int(pid),
                        "risk_level": in_risk_level,
                        "overall_risk_score": float(in_overall_risk_score),
                        "fobt": in_fobt_resultado,
                        "cea": float(in_cea_level),
                    }
                    st.success("Registro guardado correctamente.")

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
