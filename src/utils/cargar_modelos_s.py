import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os
import pandas as pd     # Necesario para SHAP
import shap              # Necesario para SHAP
import matplotlib.pyplot as plt # Necesario para SHAP
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.utils.gradcam_utils import generate_gradcam, generate_gradcam_colon, generar_explicacion_shap


# ==========================================
# PARCHE DE EMERGENCIA Y CARGA (CACHÉ)
# ==========================================
@st.cache_resource
def inicializar_entorno_y_modelo():
    """Ejecuta el parche de Keras y carga el modelo una sola vez"""
    # 1. Aplicar Parche Maestro para Keras 3
    original_dense_init = tf.keras.layers.Dense.__init__

    def patched_dense_init(self, *args, **kwargs):
        if "quantization_config" in kwargs:
            kwargs.pop("quantization_config")
        return original_dense_init(self, *args, **kwargs)

    tf.keras.layers.Dense.__init__ = patched_dense_init

    # 2. Localizar y cargar modelo
    # Ajustamos la ruta para que funcione desde la carpeta raíz o subcarpetas
    directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_CNN_PATH = os.path.join(
        directorio_raiz, "networks", "dl", "modelo_pro_agresivo.keras"
    )

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
            return tf.keras.models.load_model(
                MODEL_CNN_PATH, compile=False, safe_mode=False
            )
        except Exception as e2:
            st.error(f"Error fatal en carga de modelo Keras: {e} | {e2}")
            return None


# ==========================================
# MODELO BIOPSIAS (PYTORCH)
# ==========================================


class BiopsyClassifier(nn.Module):
    """Misma arquitectura usada en el entrenamiento"""

    def __init__(self):
        super(BiopsyClassifier, self).__init__()
        self.model = models.resnet18(
            weights=None
        )  # No necesitamos ImageNet aquí, cargaremos los nuestros
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def cargar_modelo_biopsia():
    """Carga los pesos de PyTorch para Biopsias"""
    directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(
        directorio_raiz, "networks", "dl_biopsia", "biopsia_resnet18_best.pth"
    )

    if not os.path.exists(MODEL_PATH):
        st.error(f"Pesos de Biopsia no encontrados en: {MODEL_PATH}")
        return None

    try:
        model = BiopsyClassifier()
        # Forzamos cargar en CPU ya que el Streamlit correrá en CPU/Incompatibilidad
        state_dict = torch.load(
            MODEL_PATH, map_location=torch.device("cpu"), weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error cargando modelo de Biopsia: {e}")
        return None


# --- AÑADE ESTA LÍNEA AQUÍ ---
# Creamos un alias para que app.py encuentre la función con el nombre antiguo
obtener_modelo_cnn = inicializar_entorno_y_modelo
def predecir(modelo, selector, fuma, alc, fam, diet_red, diet_salt, diet_veg, phys, bmi, sangre, cea):
    try:
        # --- 1. PREPARACIÓN ---
        fuma_n = 10 if fuma == "Sí" else 0
        fam_n = 1 if fam else 0
        sangre_n = 1 if (sangre == "Positivo" or sangre == "Sí") else 0
        obesidad_n = 1 if float(bmi) >= 30 else 0
        
        features_input = np.array([[
            float(fuma_n), float(alc), float(obesidad_n), float(fam_n),
            float(diet_red), float(diet_salt), float(diet_veg),
            float(phys), float(bmi), float(sangre_n), float(cea)
        ]])

        # --- 2. PREDICCIÓN ---
        probabilidades = modelo.predict_proba(features_input)[0]
        if len(probabilidades) == 3:
            p_bajo, p_medio, p_alto = probabilidades
            riesgo_calculado = 0.005 + (p_medio * 0.45) + (p_alto * 1.0)
            clase_predicha = np.argmax(probabilidades)
        else:
            riesgo_calculado = 0.005 + probabilidades[1]
            clase_predicha = 1

        riesgo_calculado = min(riesgo_calculado, 1.0)

        # --- 3. SHAP ---
        fig_shap, tabla_importancia = generar_explicacion_shap(modelo, features_input, clase_predicha)

        # --- 4. RENDERIZADO VISUAL ---
        col_res, col_shap = st.columns([1, 1.2])
        
        color = "#28a745" if riesgo_calculado < 0.25 else "#ffa500" if riesgo_calculado < 0.60 else "#ff4b4b"

        with col_res:
            # Encabezado del Informe
            st.markdown(f"""
                <div style='background-color: #111; padding: 20px; border-radius: 15px; border: 1px solid {color};'>
                    <h2 style='color: white; text-align: center; margin: 0;'>INFORME MÉDICO</h2>
                    <p style='text-align: center; color: #666; font-size: 0.8em;'>ID Paciente: {selector}</p>
                    <div style='text-align: center; padding: 20px 0;'>
                        <span style='font-size: 55px; font-weight: bold; color: {color};'>{riesgo_calculado * 100:.1f}%</span>
                        <br><span style='color: #888; letter-spacing: 2px;'>RIESGO ESTIMADO</span>
                    </div>
                </div>
                <div style='margin-top: 15px;'>
                    <p style='color: white; font-weight: bold; margin-bottom: 10px;'>ANÁLISIS DE FACTORES:</p>
                </div>
            """, unsafe_allow_html=True)

            # Rejilla de Factores de Riesgo Estilizada
            # Creamos las columnas para que los "bloques" se vean ordenados
            c1, c2 = st.columns(2)
            
            # Definimos un estilo común para las mini-tarjetas
            def card_style(label, value, is_alert):
                bg = "#4d1111" if is_alert else "#1a1a1a"
                border = "#ff4b4b" if is_alert else "#333"
                txt = "#ff4b4b" if is_alert else "#aaa"
                return f"""<div style='background:{bg}; border: 1px solid {border}; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                            <p style='margin:0; font-size: 0.7em; color: {txt}; font-weight: bold;'>{label}</p>
                            <p style='margin:0; font-size: 0.9em; color: white;'>{value}</p>
                          </div>"""

            with c1:
                st.markdown(card_style("SANGRE (FOBT)", "DETECTADA 🚨" if sangre_n else "NEGATIVO ✅", sangre_n), unsafe_allow_html=True)
                st.markdown(card_style("NIVEL CEA", f"{cea} ng/mL {'⚠️' if cea > 3 else 'OK'}", cea > 3), unsafe_allow_html=True)
                st.markdown(card_style("TABAQUISMO", "FUMADOR 🚬" if fuma == "Sí" else "NO FUMA 🚭", fuma == "Sí"), unsafe_allow_html=True)
            
            with c2:
                st.markdown(card_style("ESTADO IMC", f"{bmi} ({'OBESIDAD' if bmi >= 30 else 'NORMAL'})", bmi >= 30), unsafe_allow_html=True)
                st.markdown(card_style("DIETA ROJA", f"Nivel {diet_red}/10 {'⚠️' if diet_red > 7 else 'OK'}", diet_red > 7), unsafe_allow_html=True)
                st.markdown(card_style("HISTORIAL", "FAMILIAR 👪" if fam else "SIN ANTECED.", fam), unsafe_allow_html=True)

        with col_shap:
            if fig_shap:
                st.pyplot(fig_shap)
                st.markdown("### 📊 Desglose de Influencia")
                # Usamos dataframe en lugar de table para que se vea más moderno y permita scroll
                st.dataframe(
                    tabla_importancia, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Impacto": st.column_config.TextColumn("Influencia (%)"),
                        "Sentido": st.column_config.TextColumn("Efecto en Riesgo")
                    }
                )

    except Exception as e:
        st.error(f"Error en visualización: {e}")
    
def colonos(imagen):
    """Procesamiento de imagen adaptado a Streamlit"""
    if imagen is None:
        return "<div style='color:red;'>No se ha subido ninguna imagen</div>", None

    try:
        # --- CORRECCIÓN CLAVE ---
        # Llamamos a la función para obtener el modelo cargado en caché
        modelo_instancia = obtener_modelo_cnn()

        if modelo_instancia is None:
            return (
                "<div style='color:orange;'> El modelo no se pudo cargar.</div>",
                imagen,
            )

        # 1. Redimensionar
        img_res = cv2.resize(imagen, (150, 150))

        # 2. Pre-procesamiento
        img_batch = np.expand_dims(img_res, axis=0).astype(np.float32)
        img_preprocessed = preprocess_input(img_batch)

        # 3. Predicción usando la instancia real del modelo
        pred = modelo_instancia.predict(img_preprocessed, verbose=0)[0][0]

        # heatmap_img, pred_val = generate_gradcam_colon(modelo_instancia, imagen, "post_relu")
        # heatmap_img, pred_val = generate_gradcam_colon(modelo_instancia, imagen, "out_relu")
        heatmap_img, pred_val = generate_gradcam_colon(modelo_instancia, imagen)
    
        # Ajustamos la lógica de pólipo/sano según tu entrenamiento
        es_polipo = pred_val < 0.5
        resultado = "PÓLIPO DETECTADO" if es_polipo else "TEJIDO SANO"
        confianza = (1.0 - pred) if es_polipo else pred
        color = "#ff4b4b" if es_polipo else "#28a745"
        subtitulo = (
            "Se recomienda revisión inmediata."
            if es_polipo
            else "No se observan anomalías evidentes."
        )

        html_vision = f"""
        <div style='background-color: #000; border: 3px solid {color}; padding: 25px; border-radius: 15px; text-align: center; color: white; font-family: sans-serif;'>
            <h2 style='margin: 10px 0; color: {color}; font-size: 28px; font-weight: 900;'>{resultado}</h2>
            <div style='margin: 15px 0;'>
                <span style='font-size: 40px; font-weight: bold;'>{confianza * 100:.1f}%</span>
                <span style='font-size: 18px; color: #888;'> de confianza</span>
            </div>
            <div style='background-color: {color}22; border: 1px solid {color}; padding: 10px; border-radius: 8px;'>
                <p style='margin: 0; color: white; font-size: 14px;'>{subtitulo}</p>
            </div>
        </div>
        """
        # IMPORTANTE: Devolvemos img_res (que es un array de numpy válido para st.image)
        return html_vision, heatmap_img

    except Exception as e:
        # Si hay un error, devolvemos un mensaje y la imagen original para evitar el 'NoneType'
        return (
            f"<div style='color:red;'>Error en el motor de visión: {str(e)}</div>",
            imagen,
        )


##################################################################################################


def biopsias(imagen):
    """Procesamiento de imagen de Biopsia usando PyTorch"""
    if imagen is None:
        return "<div style='color:red;'>No se ha subido ninguna imagen</div>", None

    try:
        modelo_instancia = cargar_modelo_biopsia()
        if modelo_instancia is None:
            return (
                "<div style='color:orange;'>Modelo de Biopsia no disponible.</div>",
                imagen,
            )

        # 1. Convertir a PIL si es array (Streamlit suele pasar numpy o PIL)
        if isinstance(imagen, np.ndarray):
            img_pil = Image.fromarray(imagen)
        else:
            img_pil = imagen

        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        # 3. Pre-procesamiento (PIL)
        # 4. Grad-CAM Inferencia
        # Obtenemos la capa objetivo de ResNet18 (Normalmente la última capa conv antes del pooling: layer4[-1])
        target_layer = modelo_instancia.model.layer4[-1]
        
        # heatmap_img ya viene como array OpenCV / RGB y listo
        heatmap_img, prob = generate_gradcam(modelo_instancia, img_pil, target_layer)

        # 5. Lógica de resultados
        es_benigno = prob >= 0.5
        resultado = "BENIGNO (NORMAL)" if es_benigno else "🚨 MALIGNO (ADENOCARCINOMA)"
        
        # Confianza
        confianza = prob if es_benigno else (1.0 - prob)
        color = "#28a745" if es_benigno else "#ff4b4b"
        subtitulo = (
            "Tejido dentro de los parámetros normales."
            if es_benigno
            else "Sospecha de malignidad. El mapa de calor indica zonas de interés."
        )

        # Redimensionado de la original para que queden igualitas al lado
        img_original_224 = cv2.resize(np.array(img_pil), (224, 224))

        html_vision = f"""
        <div style='background-color: #000; border: 3px solid {color}; padding: 25px; border-radius: 15px; text-align: center; color: white; font-family: sans-serif; margin-top: 15px;'>
            <h2 style='margin: 10px 0; color: {color}; font-size: 24px; font-weight: 900;'>{resultado}</h2>
            <div style='margin: 15px 0;'>
                <span style='font-size: 40px; font-weight: bold;'>{confianza * 100:.1f}%</span>
                <span style='font-size: 18px; color: #888;'> de confianza</span>
            </div>
            <div style='background-color: {color}22; border: 1px solid {color}; padding: 10px; border-radius: 8px;'>
                <p style='margin: 0; color: white; font-size: 14px;'>{subtitulo}</p>
            </div>
        </div>
        """
        # Devolvemos TRES cosas: el html_info, la imagen Grad-CAM y la Imagen Original
        return html_vision, heatmap_img, img_original_224

    except Exception as e:
        # En caso de error, devolvemos un array vacío como imagen o la original para que no rompa la triple asignación
        img_error = cv2.resize(np.array(img_pil), (224, 224)) if 'img_pil' in locals() else imagen
        return f"<div style='color:red;'>Error en inferencia de Biopsia: {str(e)}</div>", img_error, img_error