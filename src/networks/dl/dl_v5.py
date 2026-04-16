import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import numpy as np
import tf2onnx
import random

# --- 0. CONFIGURACIÓN DE SEMILLAS (REPRODUCIBILIDAD) ---
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# --- CONFIGURACIÓN ENTORNO ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Entrenando con GPU: {gpus}")
else:
    print("GPU no detectada, se usará la CPU.")

# --- 1. RUTAS Y PARÁMETROS ---
RUTA_BASE = r"cancer de colon\prueba\dataset_colon_completo\dataset_colon_completo"
RUTA_DATASET = os.path.join(RUTA_BASE, "dataset_limpio")
SAVE_DIR = r"artifacts\weights"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# --- 2. GENERADORES (Con Seed) ---
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_gen = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=SEED,
)

val_gen = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=SEED,
)


# --- 3. CONSTRUCCIÓN DEL MODELO ---
def crear_modelo_completo():
    """
    Crea un modelo de red neuronal de doble rama (ensemble interno):
    
    - RAMA A (MobileNetV2): Usa transfer learning de un modelo preentrenado 
      en ImageNet para extraer características generales de alto nivel (bordes, texturas).
      Se congela (trainable=False) para evitar sobreajuste y acelerar entrenamiento.
      
    - RAMA B (CNN Personalizada): Red convolucional simple entrenada desde cero,
      diseñada para captar características específicas de las imágenes médicas
      (ej. patrones de color o morfología de los pólipos).
      
    Las características de ambas ramas se concatenan y procesan juntas
    por una red densa final con regularización L2 y alta tasa de Dropout (0.6)
    para minimizar el sobreajuste.
    """
    input_layer = Input(shape=(150, 150, 3))

    # RAMA A: MobileNetV2
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_tensor=input_layer
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    rama_a = layers.Dropout(0.6)(x)

    # RAMA B: CNN personalizada
    y = layers.Conv2D(32, (3, 3), activation="relu")(input_layer)
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling2D(2, 2)(y)
    y = layers.GlobalAveragePooling2D()(y)
    rama_b = layers.Dense(64, activation="relu")(y)

    # UNIÓN
    combined = layers.concatenate([rama_a, rama_b])
    combined = layers.Dense(
        32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(combined)
    output_layer = layers.Dense(1, activation="sigmoid")(combined)

    return models.Model(inputs=input_layer, outputs=output_layer)


# --- 4. COMPILACIÓN ---
ensemble_model = crear_modelo_completo()
ensemble_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", "precision", "recall"],
)

# --- 5. PESOS Y ENTRENAMIENTO ---
pesos_clase = {0: 2.0, 1: 1.0}

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
)

print("Iniciando Entrenamiento Maestro...")
history = ensemble_model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen,
    class_weight=pesos_clase,
    callbacks=[early_stop],
)

# --- 6. GUARDADO FINAL ---
path_keras = os.path.join(SAVE_DIR, "modelo_pro_agresivo.keras")
ensemble_model.save(path_keras)

# Convertir a ONNX
onnx_model, _ = tf2onnx.convert.from_keras(ensemble_model)

os.makedirs(r"artifacts\exports", exist_ok=True)
path_onnx = os.path.join(r"artifacts\exports", "modelo_pro_agresivo.onnx")
with open(path_onnx, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"\nProceso terminado. Modelo guardado en: {path_onnx}")
