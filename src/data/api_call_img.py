"""
api_call_img.py — Descarga de datasets de imágenes de colonoscopia.

Proporciona funciones para descargar datasets desde Kaggle y HuggingFace.
"""

import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Directorio base relativo a la ubicación de este archivo
BASE_DIR = Path(__file__).parent


def prueba_kaggle(output_dir=None):
    """
    Descarga el dataset de imágenes histopatológicas de pulmón y colon desde Kaggle
    y elimina la parte de pulmón para ahorrar espacio.

    Parameters
    ----------
    output_dir : str, optional
        Directorio donde descargar. Por defecto: src/data/raw/
    """
    if output_dir is None:
        output_dir = str(BASE_DIR / "raw")

    # 1. Ejecutar la descarga desde Kaggle CLI
    os.system(
        f"kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images "
        f"--unzip -p {output_dir}"
    )

    # 2. Definir rutas
    base_path = os.path.join(output_dir, "lung_colon_image_set")
    colon_path = os.path.join(base_path, "colon_image_sets")
    lung_path = os.path.join(base_path, "lung_image_sets")

    # 3. Limpieza: Borrar la parte de pulmón (Lung) si existe
    if os.path.exists(lung_path):
        print("Borrando datos de pulmón para optimizar espacio...")
        shutil.rmtree(lung_path)
        print("¡Listo! Solo quedan las imágenes de colon.")
    else:
        print("La carpeta de pulmón no se encontró o ya fue eliminada.")


def descargar_zip_huggingface(output_dir=None):
    """
    Descarga y descomprime el dataset de colonoscopia desde HuggingFace.

    Parameters
    ----------
    output_dir : str, optional
        Directorio donde guardar. Por defecto: src/data/raw/
    """
    if output_dir is None:
        output_dir = str(BASE_DIR / "raw")

    url = "https://huggingface.co/datasets/ZhenbinWang/Colonoscopic_processed/resolve/main/Colonoscopic_processed.zip"
    archivo_zip = os.path.join(output_dir, "Colonoscopic_processed.zip")
    carpeta_destino = os.path.join(output_dir, "dataset_colonoscopia")

    os.makedirs(output_dir, exist_ok=True)

    print("Iniciando descarga desde Hugging Face...")

    # 1. Descarga con barra de progreso
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(archivo_zip, "wb") as f,
        tqdm(
            desc="Descargando",
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
    print("Proceso completado con éxito.")


def listar_contenido_descomprimido(carpeta_destino=None):
    """Lista el contenido de la carpeta del dataset descomprimido."""
    if carpeta_destino is None:
        carpeta_destino = str(BASE_DIR / "raw" / "dataset_colonoscopia")

    if not os.path.exists(carpeta_destino):
        print(f"La carpeta {carpeta_destino} no existe.")
        return

    print(f"Explorando contenido en: {os.path.abspath(carpeta_destino)}")

    # Listar todo lo que hay dentro (incluyendo subcarpetas)
    for root, dirs, files in os.walk(carpeta_destino):
        nivel = root.replace(carpeta_destino, "").count(os.sep)
        indentacion = " " * 4 * (nivel)
        print(f"{indentacion}📁 {os.path.basename(root)}/")
        sub_indent = " " * 4 * (nivel + 1)

        # Mostramos solo los primeros 3 archivos de cada carpeta para no saturar
        for f in files[:3]:
            print(f"{sub_indent}📄 {f}")
        if len(files) > 3:
            print(f"{sub_indent}... y {len(files) - 3} archivos más.")


if __name__ == "__main__":
    prueba_kaggle()
