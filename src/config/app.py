# prueba sencilla de una app de gestión clínica con Streamlit
# todo el codigo a sido por ia (de momento)

import streamlit as st
import pandas as pd
from datetime import datetime

# 1. Configuración de la página y Estilos
st.set_page_config(page_title="Portal Médico v1.0", layout="wide")

# Simulación de base de datos (En la vida real sería SQL o MongoDB)
if 'db_pacientes' not in st.session_state:
    data = {
        'ID': [101, 102, 103],
        'Nombre': ['Juan Pérez', 'María García', 'Roberto Gómez'],
        'Edad': [45, 32, 58],
        'Historial': [
            "Paciente con hipertensión controlada. Última visita: estable.",
            "Alergia a la penicilina. Seguimiento por asma.",
            "Recuperación post-operatoria de rodilla."
        ]
    }
    st.session_state.db_pacientes = pd.DataFrame(data)

# --- SIDEBAR: Descarga y Filtros ---
st.sidebar.header("Panel de Control")

# Botón para descargar CSV
csv = st.session_state.db_pacientes.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Descargar Base de Datos (CSV)",
    data=csv,
    file_name='historial_medico.csv',
    mime='text/csv',
)

# --- CUERPO PRINCIPAL ---
st.title("🩺 Sistema de Gestión Clínica")

# 2. Buscador de Usuarios
search_term = st.text_input("🔍 Buscar paciente por nombre o ID", "")

# Filtrado de datos
df = st.session_state.db_pacientes
# = df[df['Nombre'].str.contains(search_term, case=False) | df['ID'].astype(str).contains(search_term)]

# Versión robusta que ignora errores de valores vacíos
results = df[
    df['Nombre'].str.contains(search_term, case=False, na=False) | 
    df['ID'].astype(str).str.contains(search_term, na=False)
]

if not results.empty:
    # Seleccionar paciente de los resultados
    paciente_sel = st.selectbox("Seleccione el perfil para ver detalle:", results['Nombre'])
    
    # Extraer datos del paciente seleccionado
    datos_paciente = df[df['Nombre'] == paciente_sel].iloc[0]
    idx_paciente = df[df['Nombre'] == paciente_sel].index[0]

    # 3. Visualización del Historial
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Información General")
        st.info(f"**Nombre:** {datos_paciente['Nombre']}\n\n**ID:** {datos_paciente['ID']}\n\n**Edad:** {datos_paciente['Edad']} años")

    with col2:
        st.subheader("Historial Médico Actual")
        st.write(datos_paciente['Historial'])

    st.divider()

    # 4. Nuevo Informe / Actualización
    st.subheader("📝 Crear Nuevo Informe de Consulta")
    nuevo_relato = st.text_area("Escriba las observaciones de la visita de hoy:", placeholder="El paciente presenta...")

    if st.button("Guardar y Actualizar Historial"):
        if nuevo_relato:
            fecha_hoy = datetime.now().strftime("%Y-%m-%d")
            # Actualizamos el string del historial (añadiendo lo nuevo arriba)
            actualizacion = f"\n--- Consulta {fecha_hoy} ---\n{nuevo_relato}\n" + datos_paciente['Historial']
            
            # Guardar en el "estado" de la app
            st.session_state.db_pacientes.at[idx_paciente, 'Historial'] = actualizacion
            st.success("✅ Informe guardado correctamente.")
            st.rerun() # Recarga para mostrar los cambios
        else:
            st.warning("Por favor, escriba algo antes de guardar.")

else:
    st.warning("No se encontraron pacientes con ese nombre.")