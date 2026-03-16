# Plan de Implementación: Correcciones del Proyecto Cáncer de Colón

Corregir los 21 hallazgos de la revisión del proyecto, desde bugs críticos hasta mejoras de calidad.

## Proposed Changes

### A. Correcciones Críticas (Items 1-4)

---

#### [MODIFY] [sintetiza_historiales.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/scripts/sintetiza_historiales.py)

- Envolver las líneas 1-145 (generación de 5000 pacientes) dentro de una función `generar_datos_sinteticos(output_dir)`
- Mover la llamada a `to_csv` dentro de la función
- Proteger con `if __name__ == "__main__"` para que solo se ejecute explícitamente
- Mover `import random` al top del archivo (item 18)
- La semilla se fijará dentro de la función, no a nivel de módulo

---

#### [MODIFY] [testeos.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/models/testeos.py)

- Convertir en función `entrenar_random_forest(df)` que recibe un DataFrame
- Añadir `confusion_matrix` visual con seaborn
- Guardar el modelo con `joblib`
- Añadir `if __name__ == "__main__"` con carga de CSV de ejemplo
- Esto cubre items 2 y 9

---

#### [MODIFY] [.gitignore](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/.gitignore)

- Añadir `*.pth` para excluir pesos de modelos PyTorch
- Añadir `*.pkl` y `*.joblib` para modelos scikit-learn

---

#### [MODIFY] [modelo_busca_polipos_Clas.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/models/modelo_busca_polipos_Clas.py)

- Añadir split train/val/test (70/15/15) usando `torch.utils.data.random_split`
- Crear `transform_val` sin augmentation para validación y test
- Añadir bucle de evaluación por epoch con métricas (accuracy, loss) sobre val
- Añadir evaluación final sobre test con precision, recall, F1
- Mover `import random` al top del archivo (item 18)
- Esto cubre items 4, 5, 18

---

### B. Mejoras Importantes (Items 6-8)

---

#### [DELETE] [2_data_cleaning.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/utils/2_data_cleaning.py)

- Funciones vacías sin implementar, eliminarlo para no confundir

---

#### [MODIFY] [data_cleaning.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/scripts/data_cleaning.py)

- Separar la lógica de `to_csv` del proceso de limpieza: las funciones de limpieza devuelven el DataFrame limpio, y una función separada `guardar_csv()` se encarga del guardado
- Mejorar `fillna`: documentar con comentarios por qué se usa 0 y en qué casos podría ser problemático (item 20)

---

#### [MODIFY] [api_call_img.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/data/api_call_img.py)

- Usar `pathlib.Path(__file__).parent` en vez de rutas relativas hardcodeadas (item 7)

---

### C. Mejoras de Estructura (Items 10-16)

---

#### [MODIFY] [main.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/main.py)

- Añadir `argparse` con subcomandos: [eda](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/scripts/eda.py#32-175), `train-polyps`, `train-rf`, `generate-data`
- Mantener [eda](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/scripts/eda.py#32-175) como acción por defecto para no romper el flujo actual

---

#### [MODIFY] [README.md](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/README.md)

- Corregir formato Markdown (los `\#\#` escapados)
- Añadir secciones: Instalación, Uso, Estructura del proyecto, Requisitos

---

#### [DELETE] [rules_cleaning.yaml](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/config/rules_cleaning.yaml)

- Archivo vacío, eliminarlo

---

#### [MODIFY] [modelo_busca_polipos_Segment.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/models/modelo_busca_polipos_Segment.py)

- Añadir un TODO claro documentando el plan de implementación futuro, o eliminar si no hay planes

---

#### Archivos [__init__.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/utils/__init__.py) (item 10)

- Añadir imports y `__all__` a [src/models/__init__.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/models/__init__.py) y [src/scripts/__init__.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/scripts/__init__.py)

---

#### [MODIFY] [ClientsData.R](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/utils/ClientsData.R)

- Añadir comentario explicando su propósito, o eliminarlo si no es relevante (lo documentaré como "legacy/experimental")

---

### D. Calidad de Código (Items 17, 19, 21)

---

#### [MODIFY] [eda_visualization.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/utils/eda_visualization.py)

- Añadir `plt.close(fig)` después de cada `st.pyplot(fig)` para evitar memory leaks

---

#### [MODIFY] [.env.sample](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/.env.sample)

- Corregir formato: `HF_TOKEN=your_token_here` sin espacios ni comillas

---

## Verification Plan

### Verificación automática
1. **Compilación del modelo**: Ejecutar `python -c "from src.models.modelo_busca_polipos_Clas import PolypDetector; print('OK')"` para verificar que los imports no rompen nada
2. **Imports limpios**: Ejecutar `python -c "from src.scripts.sintetiza_historiales import sintetizar_historiales; print('Import OK')"` — debe importar SIN generar datos ni escribir CSVs
3. **Import de testeos**: Ejecutar `python -c "from src.models.testeos import entrenar_random_forest; print('OK')"` — debe importar sin error
4. **Streamlit**: Ejecutar `streamlit run main.py` y verificar visualmente que las 5 secciones del menú funcionan

### Verificación manual (usuario)
1. Abrir la app de Streamlit y navegar por las 5 secciones para confirmar que todo sigue funcionando
2. Verificar en Git que [polyp_resnet18.pth](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/polyp_resnet18.pth) ya no aparece como tracked con `git status`
