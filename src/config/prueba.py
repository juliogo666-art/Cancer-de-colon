import streamlit as st
import pandas as pd
from datetime import datetime

# Configuración profesional
st.set_page_config(page_title="HCP Portal - Gestión de Colonoscopia", layout="wide")

# --- SIMULACIÓN DE DATOS SEGÚN EL ESQUEMA ---
if 'db' not in st.session_state:
    # Tabla Patient
    st.session_state.patients = pd.DataFrame([{
        'PK_id_patient': 101, 'first_name': 'Juan', 'last_name': 'Pérez', 
        'gender': 'M', 'age': 45, 'dni_pasaporte': '12345678X', 
        'doctor_name': 'Dr. Smith', 'contact_phone': '600123456'
    }])
    # Tabla Medical_Historical
    st.session_state.history = pd.DataFrame([{
        'id_patient': 101, 'family_cancer': True, 'smoking_status': False,
        'alcohol_consumption': 'Moderado', 'diet_type': 'Mediterránea',
        'prev_polyps': False, 'weight_kg': 80.5, 'height_cm': 175
    }])
    # Tabla Prediction (Simulada)
    st.session_state.predictions = pd.DataFrame([{
        'id_image': 501, 'id_patient': 101, 'prediction_label': 'Pólipo Detectado',
        'prediction_result': 0.94, 'ai_observations': 'Sospecha de adenoma en segmento sigmoide.'
    }])

# --- INTERFAZ ---
st.title("🩺 Sistema de Diagnóstico Gastrointestinal")

# Buscador en la barra lateral
search_term = st.sidebar.text_input("🔍 Buscar por DNI o Apellido")
df_p = st.session_state.patients
results = df_p[df_p['dni_pasaporte'].str.contains(search_term, na=False) | 
               df_p['last_name'].str.contains(search_term, case=False, na=False)]

if not results.empty:
    selected_name = st.selectbox("Seleccione Paciente:", results['first_name'] + " " + results['last_name'])
    p_id = results.iloc[0]['PK_id_patient'] # Simplificado para la demo

    # --- ORGANIZACIÓN POR TABS (Pestañas) ---
    tab1, tab2, tab3 = st.tabs(["👤 Datos Personales", "📋 Antecedentes Médicos", "🤖 Predicción IA"])

    with tab1:
        st.subheader("Información del Paciente")
        # Formulario para editar datos del usuario
        with st.form("edit_patient"):
            col1, col2 = st.columns(2)
            p_data = st.session_state.patients.loc[st.session_state.patients['PK_id_patient'] == p_id].iloc[0]
            
            new_first = col1.text_input("Nombre", p_data['first_name'])
            new_last = col2.text_input("Apellidos", p_data['last_name'])
            new_dni = col1.text_input("DNI/Pasaporte", p_data['dni_pasaporte'])
            new_doc = col2.text_input("Médico Asignado", p_data['doctor_name'])
            
            if st.form_submit_button("Actualizar Datos Personales"):
                st.session_state.patients.loc[st.session_state.patients['PK_id_patient'] == p_id, 
                    ['first_name', 'last_name', 'dni_pasaporte', 'doctor_name']] = [new_first, new_last, new_dni, new_doc]
                st.success("Datos de usuario actualizados.")

    with tab2:
        st.subheader("Historial Médico (Medical_Historical)")
        h_data = st.session_state.history.loc[st.session_state.history['id_patient'] == p_id].iloc[0]
        
        with st.form("edit_history"):
            c1, c2, c3 = st.columns(3)
            f_cancer = c1.checkbox("Antecedentes Familiares Cáncer", h_data['family_cancer'])
            smoking = c2.checkbox("Fumador", h_data['smoking_status'])
            p_polyps = c3.checkbox("Pólipos Previos", h_data['prev_polyps'])
            
            diet = st.selectbox("Tipo de Dieta", ["Sana", "Mediterránea", "Alta en grasas"], index=1)
            alcohol = st.text_input("Consumo Alcohol", h_data['alcohol_consumption'])
            
            weight = st.number_input("Peso (kg)", value=float(h_data['weight_kg']))
            
            if st.form_submit_button("Guardar Cambios en Historial"):
                # Aquí actualizarías el dataframe session_state.history
                st.success("Historial médico actualizado en la base de datos.")

    with tab3:
        st.subheader("Último Análisis de Colonoscopia")
        pred = st.session_state.predictions[st.session_state.predictions['id_patient'] == p_id].iloc[0]
        
        col_img, col_info = st.columns([1, 1])
        with col_img:
            # Placeholder para la imagen de la tabla 'Images'
            st.image("https://via.placeholder.com/400x300.png?text=Captura+Colonoscopia", caption="Frame analizado")
        
        with col_info:
            st.metric(label="Resultado IA", value=pred['prediction_label'], delta=f"{pred['prediction_result']*100}% Confianza")
            st.warning(f"**Observaciones IA:** {pred['ai_observations']}")
            
        st.divider()
        st.subheader("Nuevo Informe Médico")
        nuevo_informe = st.text_area("Anotaciones del doctor para esta predicción:", placeholder="Escriba aquí...")
        if st.button("Finalizar y Guardar Consulta"):
            st.balloons()
            st.success("Informe vinculado a la predicción correctamente.")

else:
    st.info("Utilice el buscador lateral para encontrar un paciente.")

# Exportación CSV en el Sidebar
st.sidebar.divider()
if st.sidebar.button("📦 Generar Reporte CSV"):
    # Unimos tablas para el reporte
    full_report = pd.merge(st.session_state.patients, st.session_state.history, left_on='PK_id_patient', right_on='id_patient')
    st.sidebar.download_button("Descargar Informe Completo", full_report.to_csv(), "reporte.csv")

# import streamlit as st
# import pandas as pd
# from datetime import datetime

# # Configuración profesional
# st.set_page_config(page_title="HCP Portal - Diagnóstico", layout="wide")

# # --- 1. INICIALIZACIÓN DEL ESTADO (Simulación de BDD) ---
# if 'patients' not in st.session_state:
#     st.session_state.patients = pd.DataFrame([{
#         'PK_id_patient': 101, 'first_name': 'Juan', 'last_name': 'Pérez', 
#         'gender': 'M', 'age': 45, 'dni_pasaporte': '12345678X', 
#         'doctor_name': 'Dr. Smith', 'contact_phone': '600123456'
#     }])

# if 'history' not in st.session_state:
#     st.session_state.history = pd.DataFrame([{
#         'id_patient': 101, 'family_cancer': True, 'smoking_status': False,
#         'alcohol_consumption': 'Moderado', 'diet_type': 'Mediterránea',
#         'prev_polyps': False, 'weight_kg': 80.5, 'height_cm': 175,
#         'last_update': datetime.now().strftime("%Y-%m-%d %H:%M")
#     }])

# if 'predictions' not in st.session_state:
#     st.session_state.predictions = pd.DataFrame([{
#         'id_image': 501, 'id_patient': 101, 'prediction_label': 'Pólipo Detectado',
#         'prediction_result': 0.94, 'ai_observations': 'Sospecha de adenoma en segmento sigmoide.'
#     }])

# # --- 2. BARRA LATERAL (Buscador y Descarga) ---
# st.sidebar.header("📋 Gestión de Datos")
# search_term = st.sidebar.text_input("🔍 Buscar por DNI o Apellido")

# # Lógica de descarga: Siempre genera el CSV con los datos actuales del session_state
# full_report = pd.merge(st.session_state.patients, st.session_state.history, 
#                        left_on='PK_id_patient', right_on='id_patient')
# csv_data = full_report.to_csv(index=False).encode('utf-8')

# st.sidebar.download_button(
#     label="📥 Descargar Base de Datos Actualizada",
#     data=csv_data,
#     file_name=f"reporte_clinico_{datetime.now().strftime('%Y%m%d')}.csv",
#     mime='text/csv',
#     help="Descarga todos los datos incluyendo las últimas modificaciones realizadas."
# )

# # --- 3. CUERPO PRINCIPAL ---
# st.title("🩺 Sistema de Gestión de Colonoscopia")

# # Filtrado
# results = st.session_state.patients[
#     st.session_state.patients['dni_pasaporte'].str.contains(search_term, na=False) | 
#     st.session_state.patients['last_name'].str.contains(search_term, case=False, na=False)
# ]

# if not results.empty:
#     selected_row = st.selectbox("Seleccione Paciente:", results.index, 
#                                 format_func=lambda x: f"{results.loc[x, 'first_name']} {results.loc[x, 'last_name']}")
    
#     p_id = results.loc[selected_row, 'PK_id_patient']

#     tab1, tab2, tab3 = st.tabs(["👤 Perfil del Paciente", "📋 Antecedentes Médicos", "🤖 Análisis IA"])

#     with tab1:
#         st.subheader("Información General (Editable)")
#         with st.form("form_paciente"):
#             # Obtenemos datos actuales
#             idx = st.session_state.patients.index[st.session_state.patients['PK_id_patient'] == p_id][0]
#             curr = st.session_state.patients.loc[idx]
            
#             c1, c2 = st.columns(2)
#             new_first = c1.text_input("Nombre", curr['first_name'])
#             new_last = c2.text_input("Apellidos", curr['last_name'])
#             new_dni = c1.text_input("DNI/Pasaporte", curr['dni_pasaporte'])
#             new_doc = c2.text_input("Médico Responsable", curr['doctor_name'])
            
#             if st.form_submit_button("💾 Guardar Cambios de Perfil"):
#                 st.session_state.patients.at[idx, 'first_name'] = new_first
#                 st.session_state.patients.at[idx, 'last_name'] = new_last
#                 st.session_state.patients.at[idx, 'dni_pasaporte'] = new_dni
#                 st.session_state.patients.at[idx, 'doctor_name'] = new_doc
#                 st.session_state.patients.at[idx, 'contact_phone'] = new_doc
#                 st.success("Cambios guardados en la base de datos.")
#                 st.rerun()

#     with tab2:
#         st.subheader("Historial Clínico")
#         h_idx = st.session_state.history.index[st.session_state.history['id_patient'] == p_id][0]
#         h_curr = st.session_state.history.loc[h_idx]

#         with st.form("form_historial"):
#             c1, c2, c3 = st.columns(3)
#             f_can = c1.checkbox("Cáncer Familiar", h_curr['family_cancer'])
#             smok = c2.checkbox("Fumador", h_curr['smoking_status'])
#             poly = c3.checkbox("Pólipos Previos", h_curr['prev_polyps'])
            
#             diet = st.selectbox("Dieta Habitual", ["Sana", "Mediterránea", "Alta en grasas"], 
#                                 index=["Sana", "Mediterránea", "Alta en grasas"].index(h_curr['diet_type']))
            
#             alc = st.text_input("Consumo de Alcohol", h_curr['alcohol_consumption'])
#             peso = st.number_input("Peso (kg)", value=float(h_curr['weight_kg']))
            
#             if st.form_submit_button("💾 Actualizar Historial Médico"):
#                 # Actualización directa del estado
#                 st.session_state.history.at[h_idx, 'family_cancer'] = f_can
#                 st.session_state.history.at[h_idx, 'smoking_status'] = smok
#                 st.session_state.history.at[h_idx, 'prev_polyps'] = poly
#                 st.session_state.history.at[h_idx, 'diet_type'] = diet
#                 st.session_state.history.at[h_idx, 'alcohol_consumption'] = alc
#                 st.session_state.history.at[h_idx, 'weight_kg'] = peso
#                 st.session_state.history.at[h_idx, 'last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                
#                 st.success("Historial médico actualizado.")
#                 st.rerun()

#     with tab3:
#         # Pestaña de IA (Igual que antes, pero con datos vinculados)
#         pred = st.session_state.predictions[st.session_state.predictions['id_patient'] == p_id].iloc[0]
#         st.metric(label="Diagnóstico IA", value=pred['prediction_label'], delta=f"{pred['prediction_result']*100}% de certeza")
#         st.info(f"**Observación del sistema:** {pred['ai_observations']}")
#         st.image("https://via.placeholder.com/600x200.png?text=Imagen+Analizada+Segmento+Colon", use_container_width=True)

# else:
#     st.warning("No se ha encontrado ningún paciente con esos criterios.")