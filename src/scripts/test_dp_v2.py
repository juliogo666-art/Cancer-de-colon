import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. RUTA DEL MODELO (Asegúrate de que sea la correcta)
MODEL_PATH = r"cancer de colon\prueba\modelos\dl\modelo_pro_agresivo.keras"

# 2. CARGAR MODELO
print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)


def predecir_imagen(img_path):
    # Leer y preparar imagen
    img = cv2.imread(img_path)
    if img is None:
        print("No se pudo leer la imagen")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (150, 150))

    # Pre-procesamiento (IGUAL al del entrenamiento)
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_array.astype(np.float32))

    # Predicción
    prediction = model.predict(img_preprocessed, verbose=0)[0][0]

    # Resultado
    # Clase 1 suele ser la mayoritaria (Sano), Clase 0 Hallazgo
    clase = "SANO" if prediction > 0.5 else "HALLAZGO/RIESGO"
    confianza = prediction if prediction > 0.5 else (1 - prediction)

    print(f"\n--- Resultado para: {os.path.basename(img_path)} ---")
    print(f"Diagnóstico: {clase}")
    print(f"Confianza: {confianza:.2%}")
    print(f"Valor crudo (0-1): {prediction:.4f}")


# 3. PRUEBA AQUÍ TUS IMÁGENES
# Pon aquí la ruta de una imagen que quieras probar
ruta_test = r"cancer de colon\prueba\dataset_colon_completo\polipo sano.jpg"
if os.path.exists(ruta_test):
    predecir_imagen(ruta_test)
else:
    print(f"Pon una ruta de imagen válida en 'ruta_test' para probar")
