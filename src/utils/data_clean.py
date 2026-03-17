import pandas as pd
import os
import shutil
from tqdm import tqdm

# --- 1. CONFIGURACIÓN DE RUTAS ---
file_path = '/cancer de colon/prueba/dataset_colon_completo/dataset_colon_completo' # Asegúrate de que esta ruta es correcta
RUTA_CSV_ORIGINAL = file_path + '/metadata.csv' # El CSV original con metadatos
# Ruta donde están tus imágenes descargadas (ajusta según tu PC)
RUTA_IMAGENES_ORIGEN = file_path + '/data' 
# Carpeta nueva donde irá todo lo limpio
CARPETA_DESTINO = file_path + './dataset_limpio'

# --- 2. LIMPIEZA DEL CSV ---
print("🧹 Leyendo y limpiando metadatos...")
df = pd.read_csv(RUTA_CSV_ORIGINAL)

def determinar_clase(texto):
    texto = str(texto).lower()
    # Si dice que no hay pólipos, es sano
    if "no polyps" in texto:
        return "sano"
    # Si menciona que contiene un pólipo o hay 1 o más pólipos
    if "containing a polyp" in texto or "1 polyp" in texto or "2 polyps" in texto:
        return "polipo"
    return None

# Aplicamos la lógica a cada fila
df['clase_detectada'] = df['text'].apply(determinar_clase)

# Consolidamos: Una imagen puede tener muchas filas. 
# Si en alguna fila dice 'polipo', la imagen completa se marca como 'polipo'.
resumen = df.groupby('file_name')['clase_detectada'].apply(
    lambda x: 'polipo' if 'polipo' in x.values else 'sano'
).reset_index()

# Guardar el nuevo CSV limpio en la carpeta de destino
os.makedirs(CARPETA_DESTINO, exist_ok=True)
resumen.to_csv(os.path.join(CARPETA_DESTINO, 'metadata_limpio.csv'), index=False)
print(f"✅ CSV limpio guardado. Total imágenes únicas: {len(resumen)}")

# --- 3. COPIA Y ORGANIZACIÓN DE IMÁGENES ---
print("📂 Iniciando copia y clasificación de imágenes...")

# Creamos las carpetas para la CNN
os.makedirs(os.path.join(CARPETA_DESTINO, 'polipo'), exist_ok=True)
os.makedirs(os.path.join(CARPETA_DESTINO, 'sano'), exist_ok=True)

errores = 0
copiadas = 0

for _, fila in tqdm(resumen.iterrows(), total=len(resumen)):
    # file_name suele ser 'data/008605.png', sacamos solo el nombre del archivo
    nombre_archivo = os.path.basename(fila['file_name'])
    clase = fila['clase_detectada']
    
    # Construimos rutas de origen y destino
    ruta_origen = os.path.join(RUTA_IMAGENES_ORIGEN, nombre_archivo)
    ruta_final = os.path.join(CARPETA_DESTINO, clase, nombre_archivo)
    
    try:
        # Hacemos una COPIA física del archivo
        if os.path.exists(ruta_origen):
            shutil.copy2(ruta_origen, ruta_final)
            copiadas += 1
        else:
            # Si no está en 'data', probamos buscarlo sin el prefijo
            # A veces el CSV dice 'data/xxx.png' pero el archivo es 'xxx.png'
            errores += 1
    except Exception as e:
        errores += 1

print(f"\n✨ ¡Proceso terminado!")
print(f"📸 Imágenes copiadas con éxito: {copiadas}")
print(f"⚠️ Imágenes no encontradas u errores: {errores}")
print(f"📂 Tu dataset listo para la CNN está en: {CARPETA_DESTINO}")