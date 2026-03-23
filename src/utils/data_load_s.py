import pandas as pd
import os
import streamlit as st

# --- CARGA DE DATOS OPTIMIZADA ---
@st.cache_data
def nombres_p(path):
    """Obtiene la lista de nombres para el selector de Streamlit."""
    try:
        if not os.path.exists(path):
            return []
        # Leemos solo las columnas necesarias para ahorrar memoria
        df = pd.read_csv(path, usecols=['nombre', 'apellido1', 'Patient_ID'])
        df['display_name'] = df['nombre'] + " " + df['apellido1'] + " (ID: " + df['Patient_ID'].astype(str) + ")"
        return df['display_name'].tolist()
    except Exception as e:
        print(f"Error cargando nombres: {e}")
        return []

@st.cache_data
def datos_p(nombre_completo, path):
    """Busca un paciente y devuelve sus datos formateados para los inputs de Streamlit."""
    if not nombre_completo or "ID: " not in nombre_completo:
        return [None] * 13
        
    try:
        df = pd.read_csv(path)
        # Extraemos el ID del string del selector
        pid = int(nombre_completo.split("ID: ")[1].replace(")", ""))
        p = df[df['Patient_ID'] == pid].iloc[0]
        
        # Validación de estadio
        estadio_csv = p['Cancer_Stage']
        estadio_seguro = int(estadio_csv) if estadio_csv in [1, 2, 3, 4] else 1

        # Devolvemos una tupla con los tipos de datos correctos para Streamlit
        return (
            float(p['Age']), 
            "Masculino" if p['Gender'] == 1 else "Femenino",
            estadio_seguro,
            float(p['Tumor_Size_mm']), 
            bool(p['Family_History']),
            "Sí" if p['Smoking_History'] == 1 else "No", 
            float(p['Alcohol_Consumption']),
            "Sí" if p['Diabetes'] == 1 else "No", 
            bool(p['Inflammatory_Bowel_Disease']),
            "Sí" if p['FOBT_Resultado_n'] == 1 else "No", 
            float(p['CEA_Level_ng_mL..Marcador.Tumoral.']),
            float(p['altura_cm']), 
            float(p['peso_kg'])
        )
    except Exception as e:
        st.error(f"Error al buscar datos del paciente: {e}")
        return [None] * 13
    
def save_r(folder, selector, edad, genero, estadio, tumor, fam, fuma, alc, diab, ibd, sangre, cea, altura, peso):
    """Guarda el registro en un archivo CSV local."""
    try:
        path_nuevo = os.path.join(folder, 'registros_nuevos.csv')
        
        # Creamos el diccionario con los datos actuales de la interfaz
        nuevo_dato = {
            'Paciente': selector, 
            'Edad': edad, 
            'Genero': genero, 
            'Estadio': estadio,
            'Tamaño_Tumor': tumor,
            'Historial_Familiar': fam,
            'Fumador': fuma,
            'Alcohol': alc,
            'Diabetes': diab,
            'IBD': ibd,
            'Sangre_Heces': sangre,
            'CEA': cea,
            'Altura': altura,
            'Peso': peso,
            'Fecha_Registro': pd.Timestamp.now()
        }
        
        df = pd.DataFrame([nuevo_dato])
        
        # Guardado en modo 'append' (añadir al final)
        header_necesario = not os.path.exists(path_nuevo)
        df.to_csv(path_nuevo, mode='a', index=False, header=header_necesario)
        
        return "Registro guardado correctamente en local."
    except Exception as e:
        return f"Error al guardar: {str(e)}"