# Modelo definitivo

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# --- 0. CONFIGURACIÓN INICIAL ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Silenciar warnings innecesarios

# Verificación de GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Entrenando con GPU: {gpus}")
else:
    print("⚠️ GPU no detectada, se usará la CPU.")

# --- 1. RUTAS Y PARÁMETROS ---
RUTA_BASE = r'cancer de colon\prueba\dataset_colon_completo\dataset_colon_completo'
RUTA_DATASET = os.path.join(RUTA_BASE, 'dataset_limpio')
SAVE_DIR = r'cancer de colon\prueba\modelos\dl'
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (150, 150)
BATCH_SIZE = 16  # Batch más pequeño ayuda a salir de estancamientos (mínimos locales)

# --- 2. GENERADORES (Usando Pre-procesamiento Nativo de MobileNet) ---
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Normaliza entre -1 y 1
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
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

# --- 3. CONSTRUCCIÓN DEL MODELO AJUSTADO ---
def crear_modelo_completo():
    input_layer = Input(shape=(150, 150, 3))

    # RAMA A: MobileNetV2 - VOLVEMOS A CONGELAR
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_layer)
    base_model.trainable = False # No lo toques, usa su conocimiento de Google
        
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) # Añadimos L2
    rama_a = layers.Dropout(0.6)(x) # Subimos dropout

    # RAMA B: CNN personalizada
    y = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    y = layers.BatchNormalization()(y) # Estabiliza la pérdida
    y = layers.MaxPooling2D(2, 2)(y)
    y = layers.GlobalAveragePooling2D()(y)
    rama_b = layers.Dense(64, activation='relu')(y)

    # UNIÓN
    combined = layers.concatenate([rama_a, rama_b])
    # Añadimos una capa extra de decisión con L2
    combined = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined)
    output_layer = layers.Dense(1, activation='sigmoid')(combined)

    return models.Model(inputs=input_layer, outputs=output_layer)

# --- 4. COMPILACIÓN (Subimos un pelín el Learning Rate) ---
ensemble_model = crear_modelo_completo()
ensemble_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # 0.0001
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# --- 5. PESOS MÁS SUAVES ---
pesos_clase = {0: 2.0, 1: 1.0}

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True,
    verbose=1
)

print("🚀 Iniciando Entrenamiento Maestro...")
history = ensemble_model.fit(
    train_gen, 
    epochs=50, 
    validation_data=val_gen,
    class_weight=pesos_clase,
    callbacks=[early_stop]
)

# --- 6. GUARDADO FINAL ---
path_final = os.path.join(SAVE_DIR, 'modelo_pro_agresivo.onnx')
ensemble_model.save(path_final)
print(f"\n✅ Proceso terminado. Modelo guardado en: {path_final}")