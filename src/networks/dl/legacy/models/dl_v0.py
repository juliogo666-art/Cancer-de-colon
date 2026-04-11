# Este es la primera prueba y modelos de deep learning

import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# --- 1. CONFIGURACIÓN ---
PATH_ORIGEN = '/cancer de colon/prueba/Colonoscopic_processed/Colonoscopic_processed'
PATH_DESTINO = './dataset_procesado'

# Crear carpeta de destino si no existe
if os.path.exists(PATH_DESTINO):
    shutil.rmtree(PATH_DESTINO) # Limpiar para no mezclar ejecuciones anteriores

# --- 2. PASO OPENCV: PROCESAR Y GUARDAR ---
print("🔧 Procesando imágenes con OpenCV...")

# Obtener solo las 3 primeras subcarpetas
subcarpetas = sorted([f for f in os.listdir(PATH_ORIGEN) if os.path.isdir(os.path.join(PATH_ORIGEN, f))])[:3]

for sub in subcarpetas:
    ruta_sub = os.path.join(PATH_ORIGEN, sub)
    # Clasificación simple: supongamos que las impares son pólipos y pares sanas 
    # (Ajusta esto según la lógica real de tu dataset)
    clase = "polipo" if int(sub) % 2 != 0 else "sano"
    
    ruta_salida = os.path.join(PATH_DESTINO, clase)
    os.makedirs(ruta_salida, exist_ok=True)
    
    for img_name in os.listdir(ruta_sub):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(ruta_sub, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                # --- OPERACIONES OPENCV ---
                # 1. Luz de endoscopio (HSV)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)
                img_mod = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                # 2. Parche negro (Cutout)
                h, w, _ = img_mod.shape
                cv2.rectangle(img_mod, (w//4, h//4), (w//2, h//2), (0,0,0), -1)
                
                # Guardar imagen procesada físicamente
                cv2.imwrite(os.path.join(ruta_salida, f"{sub}_{img_name}"), img_mod)

print(f"✅ Procesamiento completado. Imágenes guardadas en {PATH_DESTINO}")

# --- 3. ENTRENAMIENTO CON CNN ---
# Ahora que las imágenes están en el disco, las cargamos
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    PATH_DESTINO, target_size=(150, 150), batch_size=8, # Batch pequeño para pocas imágenes
    class_mode='binary', subset='training'
)

val_gen = datagen.flow_from_directory(
    PATH_DESTINO, target_size=(150, 150), batch_size=8,
    class_mode='binary', subset='validation'
)

# --- 4. ARQUITECTURA CNN ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 5. ENTRENAMIENTO ---
print("\n🚀 Iniciando entrenamiento con los datos procesados...")
history = model.fit(train_gen, epochs=10, validation_data=val_gen)

# (Aquí puedes añadir el código de gráficas que ya tenías)
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