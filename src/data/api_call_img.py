import os
import shutil
import requests
import zipfile
from tqdm import tqdm


def prueba_kaggle():
    # 1. Definir rutas (basadas en la estructura de ese dataset)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    destino_raw = os.path.join(base_dir, "src", "data", "raw")
    os.makedirs(destino_raw, exist_ok=True)

    # 2. Ejecutar la descarga desde Python usando Kaggle API
    # Lo descargamos directamente en la carpeta raw
    os.system(
        f"kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images --path '{destino_raw}' --unzip"
    )

    # La carpeta descomprimida suele llamarse 'lung_colon_image_set'
    base_path = os.path.join(destino_raw, "lung_colon_image_set")
    colon_path = os.path.join(base_path, "colon_image_sets")
    lung_path = os.path.join(base_path, "lung_image_sets")

    # Limpieza: Borrar la parte de pulmón (Lung) si existe
    if os.path.exists(lung_path):
        print("Borrando datos de pulmón para optimizar espacio...")
        shutil.rmtree(lung_path)
        print("¡Listo! Solo quedan las imágenes de colon.")
    else:
        print("La carpeta de pulmón no se encontró o ya fue eliminada.")

    # Mover colon_image_sets directamente a raw y borrar la carpeta padre vacía
    if os.path.exists(colon_path):
        target_colon = os.path.join(destino_raw, "colon_image_sets")
        if os.path.exists(target_colon):
            shutil.rmtree(target_colon)  # Limpiar si ya existe
        shutil.move(colon_path, destino_raw)
        shutil.rmtree(base_path)
        print(f"Imágenes de Kaggle organizadas en: {target_colon}")


def descargar_zip_huggingface():
    # URL directa del archivo ZIP en Hugging Face
    url = "https://huggingface.co/datasets/ZhenbinWang/Colonoscopic_processed/resolve/main/Colonoscopic_processed.zip"

    # Rutas para este archivo
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    destino_raw = os.path.join(base_dir, "src", "data", "raw")
    os.makedirs(destino_raw, exist_ok=True)

    archivo_zip = os.path.join(destino_raw, "Colonoscopic_processed.zip")
    carpeta_destino = os.path.join(destino_raw, "colonoscopic_processed")

    print("Iniciando descarga desde Hugging Face...")

    # 1. Descarga del archivo con barra de progreso
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(archivo_zip, "wb") as f,
        tqdm(
            desc="Colonoscopic_processed.zip",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    # 2. Descompresión
    print(f"Descomprimiendo en '{carpeta_destino}'...")
    with zipfile.ZipFile(archivo_zip, "r") as zip_ref:
        zip_ref.extractall(carpeta_destino)

    # 3. Limpieza (borrar el zip para ahorrar espacio)
    os.remove(archivo_zip)
    print(f"Proceso completado con éxito. Imágenes en: {carpeta_destino}")


def listar_contenido_descomprimido():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    destino_raw = os.path.join(base_dir, "src", "data", "raw")
    carpetas_a_revisar = [
        os.path.join(destino_raw, "colon_image_sets"),
        os.path.join(destino_raw, "colonoscopic_processed"),
    ]

    for carpeta_destino in carpetas_a_revisar:
        if not os.path.exists(carpeta_destino):
            print(f"La carpeta {carpeta_destino} no existe.")
            continue

        print(f"\nExplorando contenido en: {os.path.abspath(carpeta_destino)}")

        # Listar todo lo que hay dentro (incluyendo subcarpetas)
        for root, dirs, files in os.walk(carpeta_destino):
            nivel = root.replace(carpeta_destino, "").count(os.sep)
            indentacion = " " * 4 * (nivel)
            print(f"{indentacion}{os.path.basename(root)}/")
            sub_indent = " " * 4 * (nivel + 1)

            # Mostramos solo los primeros 3 archivos de cada carpeta para no saturar
            for f in files[:3]:
                print(f"{sub_indent}{f}")
            if len(files) > 3:
                print(f"{sub_indent}... y {len(files) - 3} archivos más.")


if __name__ == "__main__":
    prueba_kaggle()
