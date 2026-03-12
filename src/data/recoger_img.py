import os
import shutil

name = "your_kaggle_username"
api = "your_kaggle_api_key"
url = "andrewmvd/lung-and-colon-cancer-histopathological-images"

# 1. Ejecutar la descarga desde Python
os.system("uv add kaggle")
os.system(f"$env:KAGGLE_USERNAME={name}")
os.system(f"$env:KAGGLE_KEY={api}")
os.system(f"kaggle datasets download -d {url} --unzip")

# # 2. Definir rutas (basadas en la estructura de ese dataset)
# # La carpeta descomprimida suele llamarse 'lung_colon_image_set'
# base_path = 'lung_colon_image_set'
# colon_path = os.path.join(base_path, 'colon_image_sets')
# lung_path = os.path.join(base_path, 'lung_image_sets')

# # 3. Limpieza: Borrar la parte de pulmón (Lung) si existe
# if os.path.exists(lung_path):
#     print("Borrando datos de pulmón para optimizar espacio...")
#     shutil.rmtree(lung_path)
#     print("¡Listo! Solo quedan las imágenes de colon.")
# else:
#     print("La carpeta de pulmón no se encontró o ya fue eliminada.")