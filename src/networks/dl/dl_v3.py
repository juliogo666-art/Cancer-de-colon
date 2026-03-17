# Este es la cuarta prueba y modelos de deep learning

import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os

# --- 0. SILENCIAR LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. CONFIGURACIÓN ---
RUTA_BASE = r'cancer de colon\prueba\dataset_colon_completo\dataset_colon_completo'
SAVE_DIR = r'cancer de colon\prueba\modelos\dl'
os.makedirs(SAVE_DIR, exist_ok=True)

RUTA_DATASET = os.path.join(RUTA_BASE, 'dataset_limpio')
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# --- 2. PRE-PROCESADO (Añadimos ruido aleatorio para que no memorice pixeles) ---
def mi_procesado_opencv(image):
    img = (image * 255).astype(np.uint8)
    # CLAHE para resaltar texturas
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Gamma
    gamma = 1.2
    tabla = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, tabla)
    
    return img.astype(np.float32) / 255.0

# --- 3. GENERADORES (AUMENTO DE DATOS AGRESIVO) ---
# Si la IA ve la misma foto rotada, estirada y movida, no puede memorizarla fácilmente
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=90,      # Rotación total
    width_shift_range=0.2,  # Desplazamiento
    height_shift_range=0.2,
    shear_range=0.2,        # Deformación
    zoom_range=0.3,         # Zoom fuerte
    horizontal_flip=True,
    vertical_flip=True,     # En colonoscopia la cámara gira en todas direcciones
    fill_mode='reflect',
    preprocessing_function=mi_procesado_opencv
)

train_gen = datagen.flow_from_directory(RUTA_DATASET, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='training')
val_gen = datagen.flow_from_directory(RUTA_DATASET, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='validation')

# --- 4. MODELOS CON REGULARIZACIÓN FUERTE ---
def crear_modelo_a(input_tensor):
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_tensor)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x) # Capa extra para más detalle
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x) 
    return x

def crear_modelo_b(input_tensor):
    x = layers.SeparableConv2D(32, (3, 3), activation='relu')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.SeparableConv2D(64, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    return x

# Ensamblaje
# Ensamblaje (Igual que antes)
entrada = Input(shape=(150, 150, 3))
rama_a = crear_modelo_a(entrada)
rama_b = crear_modelo_b(entrada)
unido = layers.concatenate([rama_a, rama_b])
salida = layers.Dense(1, activation='sigmoid')(unido)

ensemble_model = models.Model(inputs=entrada, outputs=salida)

# Subimos un poco el learning rate para que salga del estancamiento
ensemble_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                     loss='binary_crossentropy', metrics=['accuracy'])

# --- 5. ENTRENAMIENTO CON MONITORIZACIÓN ---
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

pesos_clase = {0: 2.0, 1: 1.0} 

print("🚀 Entrenando con Balanceo de Clases...")
ensemble_model.fit(
    train_gen, 
    epochs=30, 
    validation_data=val_gen,
    class_weight=pesos_clase, # <--- ESTO ES LA LLAVE
    callbacks=[early_stop]
)

# --- 6. GUARDADO SEGURO ---
path_final = os.path.join(SAVE_DIR, 'modelo_ensemble_final.keras')
ensemble_model.save(path_final)
print(f"✅ Modelo guardado en: {path_final}")