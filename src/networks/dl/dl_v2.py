# Este es la tercera prueba y modelos de deep learning

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Asegúrate de que esta carpeta existe y tiene subcarpetas 'polipo' y 'sano'
PATH_DATOS = 'cancer de colon/prueba/Colonoscopic_processed/Colonoscopic_processed' 

# --- 2. FUNCIÓN OPENCV (El "Enfermero" que prepara la imagen) ---
def procesar_con_opencv(image):
    # La imagen llega como float32 (0 a 1) por el rescale, la pasamos a 0-255
    img = (image * 255).astype(np.uint8)
    
    # Modificación de color: Simular luz amarillenta de endoscopio antiguo
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,1] = hsv[:,:,1] * 1.2 # Aumentar saturación
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Tapar un trozo (Cutout): Ponemos un parche negro aleatorio
    h, w, _ = img.shape
    cv2.rectangle(img, (w//4, h//4), (w//2, h//2), (0,0,0), -1)
    
    return img.astype(np.float32) / 255.0

# --- 3. PREPARACIÓN DE DATOS (Generadores) ---
datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=procesar_con_opencv, # Aquí inyectamos OpenCV
    validation_split=0.2 # Usamos el 20% para testear
)

train_gen = datagen.flow_from_directory(
    PATH_DATOS, target_size=(150, 150), batch_size=16, 
    class_mode='binary', subset='training'
)

test_gen = datagen.flow_from_directory(
    PATH_DATOS, target_size=(150, 150), batch_size=16, 
    class_mode='binary', subset='validation'
)

# --- 4. MODELO CNN (El "Cerebro") ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Sigmoid porque es Binario (0 o 1)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 5. ENTRENAMIENTO Y MÉTRICAS ---
print("\n🚀 Iniciando entrenamiento...")
history = model.fit(train_gen, epochs=20, validation_data=test_gen)

# --- 6. VISUALIZACIÓN DE MÉTRICAS ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.title('Evolución de la Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.title('Evolución del Error (Loss)')
plt.legend()
plt.show()

# --- 7. PREDICCIÓN FINAL ---
def predecir_imagen(ruta):
    img = cv2.imread(ruta)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img, (150, 150)) / 255.0
    img_res = np.expand_dims(img_res, axis=0)
    
    prediccion = model.predict(img_res)
    if prediccion[0] > 0.5:
        print(f"🔍 Resultado para {ruta}: SANO (Prob: {prediccion[0][0]:.2f})")
    else:
        print(f"⚠️ Resultado para {ruta}: PÓLIPO DETECTADO (Prob: {prediccion[0][0]:.2f})")

# Ejemplo de uso (descomenta si tienes una imagen para probar):
# predecir_imagen('ruta_de_tu_imagen_nueva.jpg')