import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# PARCHE DE EMERGENCIA Y CARGA (CACHÉ)
# ==========================================
@st.cache_resource
def inicializar_entorno_y_modelo():
    """Ejecuta el parche de Keras y carga el modelo una sola vez"""
    # 1. Aplicar Parche Maestro para Keras 3
    original_dense_init = tf.keras.layers.Dense.__init__
    def patched_dense_init(self, *args, **kwargs):
        if 'quantization_config' in kwargs:
            kwargs.pop('quantization_config')
        return original_dense_init(self, *args, **kwargs)
    
    tf.keras.layers.Dense.__init__ = patched_dense_init

    # 2. Localizar y cargar modelo
    # Ajustamos la ruta para que funcione desde la carpeta raíz o subcarpetas
    directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_CNN_PATH = os.path.join(directorio_raiz, 'networks', 'dl', 'modelo_pro_agresivo.keras')
    
    if not os.path.exists(MODEL_CNN_PATH):
        st.error(f"ARCHIVO NO ENCONTRADO EN: {MODEL_CNN_PATH}")
        return None

    try:
        modelo = tf.keras.models.load_model(MODEL_CNN_PATH, compile=False)
        print("Modelo CNN cargado exitosamente en Streamlit.")
        return modelo
    except Exception as e:
        # Intento de compatibilidad extrema
        try:
            return tf.keras.models.load_model(MODEL_CNN_PATH, compile=False, safe_mode=False)
        except:
            st.error(f"Error fatal en carga de modelo: {e}")
            return None

# --- AÑADE ESTA LÍNEA AQUÍ ---
# Creamos un alias para que app.py encuentre la función con el nombre antiguo
obtener_modelo_cnn = inicializar_entorno_y_modelo

# Instanciamos el modelo globalmente dentro del contexto de Streamlit
# modelo_vision_global = inicializar_entorno_y_modelo()

def predecir(modelo, selector, edad, genero, estadio, tumor, sangre, cea, fuma, alc, diab, fam, ibd, peso, altura):
    """Lógica de predicción clínica (Idéntica a la original)"""
    try:
        edad_v = float(edad) if (edad and edad != "") else 0
        alc_v = float(alc) if (alc and alc != "") else 0
        cea_v = float(cea) if (cea and cea != "") else 0
        
        tumor_v = float(tumor) if (tumor and tumor != "") else 0
        altura_v = float(altura) if (altura and altura != "") else 1.0
        peso_v = float(peso) if (peso and peso != "") else 0
        
        # Calcular BMI
        altura_m = altura_v / 100.0 if altura_v > 0 else 1.0
        bmi_val = peso_v / (altura_m ** 2)
        if bmi_val >= 30:
            obesity_bmi = 2
        elif bmi_val >= 25:
            obesity_bmi = 1
        else:
            obesity_bmi = 0

        def es_positivo(valor):
            return valor in [True, 1, "1", "Sí", "si", "SÍ", "Si", "Yes", "yes"]

        factores = []
        if es_positivo(fuma): factores.append("Tabaquismo activo")
        if alc_v > 7: factores.append(f"Consumo de alcohol ({alc_v} u/semana)")
        if es_positivo(sangre): factores.append("Sangre detectable en heces (FOBT+)")
        if es_positivo(fam): factores.append("Antecedentes familiares")
        if es_positivo(ibd): factores.append("Enfermedad Inflamatoria Intestinal")
        if cea_v > 5: factores.append(f"Marcador CEA elevado ({cea_v} ng/mL)")

        gen_n = 0 if genero in ["Masculino", 0, "0", "M", "m"] else 1
        
        # 13 variables que espera el modelo ML RandomForest entrenado en ml_v2.py
        features = np.array([[
            edad_v, 
            gen_n, 
            int(float(estadio)) if estadio else 1, 
            tumor_v, 
            1 if es_positivo(fam) else 0, 
            1 if es_positivo(fuma) else 0, 
            alc_v, 
            obesity_bmi, 
            1 if es_positivo(ibd) else 0, 
            1 if es_positivo(sangre) else 0, 
            cea_v, 
            altura_v, 
            peso_v
        ]])

        if modelo is not None:
            prob = modelo.predict_proba(features)[0][1]
        else:
            prob = min((edad_v / 110) * 0.4 + (0.2 if es_positivo(sangre) else 0), 0.95)

        # Determinar color y recomendación
        necesita_colonoscopia = prob > 0.4 or es_positivo(sangre)
        rec_html = """<div style='background-color: #330000; border: 2px solid #ff0000; padding: 15px; border-radius: 10px; margin-top: 15px;'><h3 style='color: #ff4b4b; margin: 0; font-size: 18px;'>🚨 RECOMENDACIÓN URGENTE</h3><p style='color: white; margin: 5px 0; font-weight: bold;'>SE REQUIERE COLONOSCOPIA DE DIAGNÓSTICO</p></div>""" if necesita_colonoscopia else """<div style='background-color: #002200; border: 1px solid #28a745; padding: 10px; border-radius: 10px; margin-top: 15px;'><p style='color: #28a745; margin: 0; font-size: 13px;'>✅ Seguimiento rutinario recomendado.</p></div>"""
        
        color_borde = "#ff4b4b" if prob > 0.6 else "#ffa500" if prob > 0.3 else "#28a745"
        factores_html = "".join([f"<li style='margin-bottom: 5px;'>{f}</li>" for f in factores])
        explicacion_html = f"<ul style='text-align: left; display: inline-block; color: white; font-size: 14px; margin: 0;'>{factores_html}</ul>" if factores else "<p style='color: #666;'>Sin factores externos detectados.</p>"

        return f"""
        <div style='background-color: #000000; padding: 30px; border-radius: 15px; border: 3px solid {color_borde}; text-align: center; color: white; font-family: sans-serif;'>
            <h2 style='color: white; margin: 0; letter-spacing: 2px;'>INFORME MÉDICO ColonAI</h2>
            <hr style='border: 0; border-top: 1px solid #222; width: 100%; margin: 15px 0;'>
            <p style='font-size: 12px; color: #aaa; text-transform: uppercase;'>Probabilidad de Hallazgos Patológicos</p>
            <div style='font-size: 64px; font-weight: 900; color: {color_borde}; margin: 10px 0;'>{prob*100:.1f}%</div>
            <div style='background-color: #0a0a0a; padding: 15px; border-radius: 10px; border: 1px solid #1a1a1a;'>{explicacion_html}</div>
            {rec_html}
            <p style='font-size: 10px; color: #444; margin-top: 20px;'>ID: {selector} | Reporte generado automáticamente.</p>
        </div>
        """
    except Exception as e:
        return f"<div style='color:red;'>Error en predicción: {str(e)}</div>"

def colonos(imagen):
    """Procesamiento de imagen adaptado a Streamlit"""
    if imagen is None:
        return "<div style='color:red;'>No se ha subido ninguna imagen</div>", None
    
    try:
        # --- CORRECCIÓN CLAVE ---
        # Llamamos a la función para obtener el modelo cargado en caché
        modelo_instancia = obtener_modelo_cnn() 
        
        if modelo_instancia is None:
            return "<div style='color:orange;'> El modelo no se pudo cargar.</div>", imagen

        # 1. Redimensionar
        img_res = cv2.resize(imagen, (150, 150))
        
        # 2. Pre-procesamiento
        img_batch = np.expand_dims(img_res, axis=0).astype(np.float32)
        img_preprocessed = preprocess_input(img_batch)
        
        # 3. Predicción usando la instancia real del modelo
        pred = modelo_instancia.predict(img_preprocessed, verbose=0)[0][0]
        
        # 4. Lógica de resultados
        es_polipo = pred < 0.5
        resultado = "PÓLIPO DETECTADO" if es_polipo else "TEJIDO SANO"
        confianza = (1.0 - pred) if es_polipo else pred
        color = "#ff4b4b" if es_polipo else "#28a745"
        subtitulo = "Se recomienda revisión inmediata." if es_polipo else "No se observan anomalías evidentes."

        html_vision = f"""
        <div style='background-color: #000; border: 3px solid {color}; padding: 25px; border-radius: 15px; text-align: center; color: white; font-family: sans-serif;'>
            <h2 style='margin: 10px 0; color: {color}; font-size: 28px; font-weight: 900;'>{resultado}</h2>
            <div style='margin: 15px 0;'>
                <span style='font-size: 40px; font-weight: bold;'>{confianza*100:.1f}%</span>
                <span style='font-size: 18px; color: #888;'> de confianza</span>
            </div>
            <div style='background-color: {color}22; border: 1px solid {color}; padding: 10px; border-radius: 8px;'>
                <p style='margin: 0; color: white; font-size: 14px;'>{subtitulo}</p>
            </div>
        </div>
        """
        # IMPORTANTE: Devolvemos img_res (que es un array de numpy válido para st.image)
        return html_vision, img_res

    except Exception as e:
        # Si hay un error, devolvemos un mensaje y la imagen original para evitar el 'NoneType'
        return f"<div style='color:red;'>Error en el motor de visión: {str(e)}</div>", imagen