import pandas as pd
import os
import streamlit as st

# --- CARGA DE DATOS OPTIMIZADA ---
@st.cache_data
def nombres_p(path_5000):
    """Obtiene la lista de IDs desde nuevos_pacientes_5000.csv."""
    try:
        if not os.path.exists(path_5000):
            return []
        df = pd.read_csv(path_5000, usecols=['Patient_ID'])
        # Devolvemos solo la lista de IDs
        return df['Patient_ID'].tolist()
    except Exception as e:
        print(f"Error cargando IDs: {e}")
        return []

@st.cache_data
def datos_p(pid, path_risk):
    """Busca los datos clínicos del ID en cancer_risk_final.csv."""
    try:
        if not os.path.exists(path_risk):
            return None
        
        df = pd.read_csv(path_risk)
        p = df[df['Patient_ID'] == pid]
        
        if p.empty:
            return None
            
        p = p.iloc[0]
        
        # Retornamos un diccionario mapeado a lo que necesita el session_state
        return {
            "fuma": "Sí" if p['Smoking'] > 5 else "No",
            "alc": float(p['Alcohol_Use']),
            "fam": bool(p['Family_History']),
            "diet_red": float(p['Diet_Red_Meat']),
            "diet_salt": float(p['Diet_Salted_Processed']),
            "diet_veg": float(p['Fruit_Veg_Intake']),
            "phys": float(p['Physical_Activity']),
            "bmi": float(p['BMI']),
            "sangre": str(p['FOBT_Resultado']),
            "cea": float(p['CEA_Level_ng_mL'])
        }
    except Exception as e:
        st.error(f"Error al buscar datos clínicos: {e}")
        return None
    
def save_r(folder, pid, fuma, alc, fam, diet_red, diet_salt, diet_veg, phys, bmi, sangre, cea):
    """Guarda el resultado del análisis."""
    try:
        path_nuevo = os.path.join(folder, 'registros_analizados.csv')
        nuevo_dato = {
            'Patient_ID': pid,
            'Fecha': pd.Timestamp.now(),
            'Prob_Riesgo': st.session_state.get('ultima_prob', 0),
            'FOBT': sangre,
            'CEA': cea
        }
        df = pd.DataFrame([nuevo_dato])
        header = not os.path.exists(path_nuevo)
        df.to_csv(path_nuevo, mode='a', index=False, header=header)
        return "✅ Registro guardado en historial."
    except Exception as e:
        return f"❌ Error al guardar: {str(e)}"