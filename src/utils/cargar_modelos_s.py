import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
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
        except:
            st.error(f"Error fatal en carga de modelo Keras: {e}")
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

# Instanciamos el modelo globalmente dentro del contexto de Streamlit
# modelo_vision_global = inicializar_entorno_y_modelo()


def predecir(
    modelo,
    selector,
    edad,
    genero,
    estadio,
    tumor,
    sangre,
    cea,
    fuma,
    alc,
    diab,
    fam,
    ibd,
    peso,
    altura,
):
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
        bmi_val = peso_v / (altura_m**2)
        if bmi_val >= 30:
            obesity_bmi = 2
        elif bmi_val >= 25:
            obesity_bmi = 1
        else:
            obesity_bmi = 0

        def es_positivo(valor):
            return valor in [True, 1, "1", "Sí", "si", "SÍ", "Si", "Yes", "yes"]

        factores = []
        if es_positivo(fuma):
            factores.append("Tabaquismo activo")
        if alc_v > 7:
            factores.append(f"Consumo de alcohol ({alc_v} u/semana)")
        if es_positivo(sangre):
            factores.append("Sangre detectable en heces (FOBT+)")
        if es_positivo(fam):
            factores.append("Antecedentes familiares")
        if es_positivo(ibd):
            factores.append("Enfermedad Inflamatoria Intestinal")
        if cea_v > 5:
            factores.append(f"Marcador CEA elevado ({cea_v} ng/mL)")

        gen_n = 0 if genero in ["Masculino", 0, "0", "M", "m"] else 1

        # 13 variables que espera el modelo ML RandomForest entrenado en ml_v2.py
        features = np.array(
            [
                [
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
                    peso_v,
                ]
            ]
        )

        if modelo is not None:
            prob = modelo.predict_proba(features)[0][1]
        else:
            prob = min((edad_v / 110) * 0.4 + (0.2 if es_positivo(sangre) else 0), 0.95)

        # Determinar color y recomendación
        necesita_colonoscopia = prob > 0.4 or es_positivo(sangre)
        rec_html = (
            """<div style='background-color: #330000; border: 2px solid #ff0000; padding: 15px; border-radius: 10px; margin-top: 15px;'><h3 style='color: #ff4b4b; margin: 0; font-size: 18px;'>🚨 RECOMENDACIÓN URGENTE</h3><p style='color: white; margin: 5px 0; font-weight: bold;'>SE REQUIERE COLONOSCOPIA DE DIAGNÓSTICO</p></div>"""
            if necesita_colonoscopia
            else """<div style='background-color: #002200; border: 1px solid #28a745; padding: 10px; border-radius: 10px; margin-top: 15px;'><p style='color: #28a745; margin: 0; font-size: 13px;'>✅ Seguimiento rutinario recomendado.</p></div>"""
        )

        color_borde = (
            "#ff4b4b" if prob > 0.6 else "#ffa500" if prob > 0.3 else "#28a745"
        )
        factores_html = "".join(
            [f"<li style='margin-bottom: 5px;'>{f}</li>" for f in factores]
        )
        explicacion_html = (
            f"<ul style='text-align: left; display: inline-block; color: white; font-size: 14px; margin: 0;'>{factores_html}</ul>"
            if factores
            else "<p style='color: #666;'>Sin factores externos detectados.</p>"
        )

        return f"""
        <div style='background-color: #000000; padding: 30px; border-radius: 15px; border: 3px solid {color_borde}; text-align: center; color: white; font-family: sans-serif;'>
            <h2 style='color: white; margin: 0; letter-spacing: 2px;'>INFORME MÉDICO ColonAI</h2>
            <hr style='border: 0; border-top: 1px solid #222; width: 100%; margin: 15px 0;'>
            <p style='font-size: 12px; color: #aaa; text-transform: uppercase;'>Probabilidad de Hallazgos Patológicos</p>
            <div style='font-size: 64px; font-weight: 900; color: {color_borde}; margin: 10px 0;'>{prob * 100:.1f}%</div>
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

        # 4. Lógica de resultados
        es_polipo = pred < 0.5
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
        return html_vision, img_res

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

        # 1. Transformaciones idénticas a las de validación del entrenamiento
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        # 2. Convertir a PIL si es array (Streamlit suele pasar numpy o PIL)
        if isinstance(imagen, np.ndarray):
            img_pil = Image.fromarray(imagen)
        else:
            img_pil = imagen

        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        # 3. Pre-procesamiento
        img_tensor = transform(img_pil).unsqueeze(
            0
        )  # Añadir dimensión Batch [1, 3, 224, 224]

        # 4. Inferencia
        with torch.no_grad():
            outputs = modelo_instancia(img_tensor)
            # Aplicamos Sigmoide para obtener probabilidad
            prob = torch.sigmoid(outputs).item()

        # 5. Lógica de resultados
        # Según datasets.ImageFolder: colon_aca -> 0 (Maligno), colon_n -> 1 (Benigno)
        # Si prob < 0.5 -> Clase 0 (Maligno)
        # Si prob >= 0.5 -> Clase 1 (Benigno)

        es_benigno = prob >= 0.5
        resultado = "BENIGNO (NORMAL)" if es_benigno else "🚨 MALIGNO (ADENOCARCINOMA)"

        # Confianza
        confianza = prob if es_benigno else (1.0 - prob)
        color = "#28a745" if es_benigno else "#ff4b4b"
        subtitulo = (
            "Tejido dentro de los parámetros normales."
            if es_benigno
            else "Sospecha de malignidad. Requiere confirmación oncológica."
        )

        # Redimensionado solo para visualización en el recuadro de Streamlit
        img_visual = cv2.resize(np.array(img_pil), (224, 224))

        html_vision = f"""
        <div style='background-color: #000; border: 3px solid {color}; padding: 25px; border-radius: 15px; text-align: center; color: white; font-family: sans-serif;'>
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
        return html_vision, img_visual

    except Exception as e:
        return (
            f"<div style='color:red;'>Error en inferencia de Biopsia: {str(e)}</div>",
            imagen,
        )
