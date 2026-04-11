# Este es la segunda prueba y modelos de deep learning

import tensorflow as tf
from keras import layers, models
# Para Keras 3 / TensorFlow moderno, se importa desde legacy
from keras.src.legacy.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
file_path = 'cancer de colon/prueba/dataset_colon_completo/dataset_colon_completo' # Asegúrate de que esta ruta es correcta

RUTA_DATASET = file_path + './dataset_limpio'
IMG_SIZE = (150, 150) # Tamaño al que reescalaremos las fotos
BATCH_SIZE = 32

# --- 2. FUNCIÓN OPENCV (Pre-procesado personalizado) ---
def mi_procesado_opencv(image):
    # La imagen llega como float32 (0 a 1). La pasamos a 0-255 para OpenCV
    img = (image * 255).astype(np.uint8)
    
    # Efecto OpenCV: Simular iluminación variable (Brillo aleatorio)
    valor = np.random.randint(-30, 30)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, valor)
    v = np.clip(v, 0, 255)
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Efecto OpenCV: Parche negro (Cutout) para que la IA no se fije en esquinas
    h, w, _ = img.shape
    cv2.rectangle(img, (0, 0), (w//5, h//5), (0,0,0), -1)
    
    # Devolvemos a float32 entre 0 y 1
    return img.astype(np.float32) / 255.0

# --- 3. GENERADORES DE DATOS ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # 20% para test
    preprocessing_function=mi_procesado_opencv # <--- Aquí inyectamos tu OpenCV
)

train_gen = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- 4. ARQUITECTURA DE LA CNN ---
model = models.Sequential([
    # Primera capa: detecta bordes y texturas simples
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    # Segunda capa: detecta formas más complejas (curvaturas de pólipos)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Tercera capa
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Aplanado y clasificación
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Evita que la IA memorice las fotos (overfitting)
    layers.Dense(1, activation='sigmoid') # 0 = Pólipo, 1 = Sano (o viceversa según el orden alfabético)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 5. ENTRENAMIENTO ---
print("🚀 Entrenando modelo...")
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen
)

# --- 6. GUARDAR EL MODELO ---
model.save('modelo_colonoscopia.h5')
print("✅ Modelo guardado como 'modelo_colonoscopia.h5'")