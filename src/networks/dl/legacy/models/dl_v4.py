import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import os
import numpy as np
import random

# --- 0. CONFIGURACIÓN DE SEMILLA (REPRODUCTIBILIDAD) ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

IMG_SIZE = (224, 224) # ResNet prefiere 224x224
BATCH_SIZE = 32

# --- 2. GENERADORES ---
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    RUTA_DATASET, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary', 
    subset='training',
    seed=SEED
)

val_gen = datagen.flow_from_directory(
    RUTA_DATASET, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary', 
    subset='validation',
    seed=SEED,
    shuffle=False
)

# --- 3. CÁLCULO AUTOMÁTICO DE PESOS DE CLASE (DESBALANCES) ---
from sklearn.utils import class_weight

labels = train_gen.classes
unique_labels = np.unique(labels)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=unique_labels,
    y=labels
)
pesos_clase = dict(zip(unique_labels, weights))

print(f"📊 Distribución de clases: {np.bincount(labels)}")
print(f"⚖️ Pesos calculados automáticamente: {pesos_clase}")

# --- 4. CONSTRUCCIÓN DEL MODELO ENSEMBLE CON RESNET ---
def crear_modelo_resnet_ensemble():
    input_layer = Input(shape=(224, 224, 3))

    # RAMA A: ResNet50V2
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_tensor=input_layer)
    base_model.trainable = False # Congelamos para no destruir pesos de imagenet
        
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    rama_a = layers.Dropout(0.5)(x)

    # RAMA B: CNN personalizada para texturas locales
    y = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    y = layers.MaxPooling2D(2, 2)(y)
    y = layers.Conv2D(64, (3, 3), activation='relu')(y)
    y = layers.GlobalAveragePooling2D()(y)
    rama_b = layers.Dense(128, activation='relu')(y)

    # UNIÓN (Ensemble)
    combined = layers.concatenate([rama_a, rama_b])
    combined = layers.Dense(64, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)
    output_layer = layers.Dense(1, activation='sigmoid')(combined)

    return models.Model(inputs=input_layer, outputs=output_layer)

# --- 5. COMPILACIÓN Y ENTRENAMIENTO ---
modelo = crear_modelo_resnet_ensemble()
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.Recall()] # Añadimos Recall por ser tema médico
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
]

print("\n🚀 Iniciando Entrenamiento con ResNet Ensemble...")
history = modelo.fit(
    train_gen, 
    epochs=50, 
    validation_data=val_gen,
    class_weight=pesos_clase,
    callbacks=callbacks
)

# --- 6. GUARDADO ---
path_final = os.path.join(SAVE_DIR, 'modelo_resnet_ensemble.keras')
modelo.save(path_final)
print(f"\n✅ Proceso terminado. Modelo guardado en: {path_final}")