# Revision Completa y Plan de Implementacion - ColonAI
## (Actualizado tras commit 8251829 del companero)

---

## Analisis del Commit del Companero (8251829 - "anadir el shap")

### Archivos modificados (9 ficheros)

| Archivo | Cambio | Valoracion |
|---------|--------|-----------|
| `src/frontend/app.py` | Reescritura de carga del ensemble SHAP | CORRIGE bug, INTRODUCE otro |
| `src/utils/gradcam_utils.py` | SHAP ahora soporta lista de modelos (ensemble) | BUENO, pero titulo sigue invisible |
| `src/networks/dl/dl_v5.py` | BATCH_SIZE 16->32, anade precision/recall | BUENO, con matiz |
| `src/tracking/predictions.csv` | 3 nuevos registros de prediccion | Sin impacto en codigo |
| 4x confusion_matrix_*.png | Regeneradas (tamanos ligeramente distintos) | Sin impacto en codigo |
| `src/data/clean/nuevos_pacientes_5000.csv` | Datos regenerados | Sin impacto en codigo |

---

### Cambio 1: app.py - Carga del ensemble SHAP

**Lo que hizo el companero:**
- Cambio `pickle.load()` por `joblib.load()` (correcto para sklearn/XGBoost/LightGBM)
- Usa la ruta centralizada `settings.MODEL_ML_FINAL_PATH` en vez de ruta hardcodeada
- Anade comprobacion con `os.path.exists()` antes de cargar
- **ELIMINO** el bloque suelto de las lineas 902-904 que cargaba el pickle fuera de contexto

**Bugs del plan original que CORRIGE:**
- ~~Bug B (Fase 1.1) - Carga incondicional del pickle al final~~ --> **CORREGIDO**

**Bugs del plan original que siguen PENDIENTES:**
- Bug A (codigo duplicado en lineas 54-57): `directorio_actual` y `directorio_raiz` siguen definidos dos veces
- Bug C (`set_page_config` despues de widgets): sigue sin corregir

**Nuevo problema introducido:**
> [!WARNING]
> El archivo ya no tiene newline al final (el diff muestra `\ No newline at end of file`). Esto puede causar warnings en linters y problemas con algunas herramientas de diff/merge.

---

### Cambio 2: gradcam_utils.py - SHAP con ensemble

**Lo que hizo el companero:**
- Creo la funcion auxiliar `_extract_raw_shap()` que gestiona distintos formatos de salida SHAP
- Anade soporte para cuando `modelo` es una lista/tupla (el ensemble son 3 modelos: RF + XGBoost + LightGBM)
- Itera sobre cada modelo del ensemble, calcula SHAP individual y **promedia** los valores
- Tiene fallback si algun modelo del ensemble falla

**Valoracion: BUEN cambio**, resuelve un problema real. El ensemble es una lista de 3 modelos y antes se pasaba directamente a `TreeExplainer` lo cual fallaba.

**Problemas que persisten:**
> [!WARNING]
> **El titulo SHAP sigue siendo invisible** (linea 274):
> ```python
> plt.title("Factores que definen tu perfil de riesgo", color="white")
> ```
> Fondo blanco + texto blanco = titulo que no se ve. Esto sigue en el plan como Fase 1.5.

**Mejoras recomendadas al codigo nuevo:**
1. La funcion `_extract_raw_shap()` esta definida **dentro** del try/except como funcion anidada. Es mejor extraerla al nivel del modulo para reutilizarla y testearla.
2. El `except Exception` del fallback (linea 214) es demasiado generico y silencia errores que podrian ser utiles para debug. Seria mejor logear que tipo de error ocurrio.
3. Falta docstring en `_extract_raw_shap()`.

---

### Cambio 3: dl_v5.py - Batch size y metricas

**Lo que hizo el companero:**
```diff
- BATCH_SIZE = 16
+ BATCH_SIZE = 32
```
```diff
- metrics=["accuracy"],
+ metrics=["accuracy", "precision", "recall"],
```

**Valoracion:**
- **BATCH_SIZE 32**: Correcto. Con el data augmentation agresivo que tiene (rotacion 30, shift, zoom, flip), un batch de 32 es mas estable para el calculo dei gradiente que uno de 16. Ademas los modelos de biopsia ya usaban 32.
- **Precision y Recall**: Excelente para monitorizacion durante entrenamiento. Sin embargo:

> [!IMPORTANT]
> El `precision` y `recall` de Keras por defecto **solo funcionan bien en clasificacion binaria** con umbral 0.5, que es exactamente este caso (polipo vs sano). Correcto para este uso.
> 
> Sin embargo, estas metricas **NO se estan registrando** en ningun tracker despues del entrenamiento. Solo se ven durante el `fit()` y luego se pierden. Esto sigue siendo parte de la Fase 4/5 del plan.

---

### Cambio 4: predictions.csv - Nuevos registros

3 predicciones de prueba del 15 de abril. Detalles notables:
- Todas tienen `patient_id` vacio -> Se hicieron sin cargar un paciente primero (modo manual)
- Todas con exactamente los mismos features y resultado `Low` con confianza 0.9996
- Parece una prueba repetida del companero para verificar que SHAP funcionaba

**Sin impacto en codigo.** Solo datos de auditoria.

---

## Impacto en el Plan Original - Resumen

| Fase Original | Estado tras el commit | Accion |
|--------------|----------------------|--------|
| **1.1 Bug B** - Carga duplicada ensemble | **CORREGIDO** por el companero | Tachar del plan |
| **1.1 Bug A** - Codigo duplicado rutas | **SIGUE PENDIENTE** | Mantener |
| **1.1 Bug C** - set_page_config mal ubicado | **SIGUE PENDIENTE** | Mantener |
| **1.2** - CSV_RISK_PATH inexistente | **SIGUE PENDIENTE** | Mantener |
| **1.3** - Cleanup incompleto lifespan | **SIGUE PENDIENTE** | Mantener |
| **1.4** - BMI duplicado | **SIGUE PENDIENTE** | Mantener |
| **1.5** - SHAP titulo blanco | **SIGUE PENDIENTE** | Mantener |
| **1.NEW** - Falta newline al final de app.py | **NUEVO** (introducido por el commit) | Anadir al plan |
| **1.NEW** - _extract_raw_shap sin docstring | **NUEVO** (introducido por el commit) | Anadir a Fase 6 |
| **Fases 2-7** | Sin cambios | Mantener integramente |

---

## FASE 1: Bugs Criticos (ACTUALIZADA)

### 1.1 [FIX] Codigo duplicado de rutas - app.py (PENDIENTE)

En [app.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/frontend/app.py) las lineas 30-32 y 54-57 definen las mismas variables:
```python
# Lineas 30-32 (primera vez)
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.dirname(os.path.dirname(directorio_actual))
sys.path.append(directorio_raiz)

# Lineas 54-57 (duplicado exacto)
directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.dirname(os.path.dirname(directorio_actual))
sys.path.append(directorio_raiz)
```
**Accion:** Eliminar las lineas 54-57 (bloque duplicado) y el comentario duplicado de las lineas 54-55.

### ~~1.1b [FIX] Carga incondicional del pickle~~ CORREGIDO POR COMPANERO

~~El compañero elimino las lineas 902-904 y uso `joblib.load()` con ruta `settings`.~~

### 1.1c [FIX] `set_page_config` mal ubicado - app.py (PENDIENTE)

`st.set_page_config()` (linea 96) se ejecuta despues de `st.columns()` (linea 83) y `st.selectbox()` (linea 85). Debe ser la primera llamada a Streamlit.

**Accion:** Mover `st.set_page_config()` a la linea 14, justo despues de `import streamlit as st`.

### 1.2 [FIX] `settings.CSV_RISK_PATH` no existe - main.py (PENDIENTE)

En [main.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/main.py) linea 95 se referencia `settings.CSV_RISK_PATH` que no existe en `settings.py`.

**Accion:** Cambiar a `settings.CSV_MASTER_PATH` o crear la propiedad `CSV_RISK_PATH` en settings apuntando al CSV de datos limpios.

### 1.3 [FIX] Cleanup incompleto en lifespan - dependencies.py (PENDIENTE)

En [dependencies.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/api/dependencies.py) linea 200, falta liberar `modelo_ml_triage` en el cleanup.

### 1.4 [FIX] Asignacion duplicada de BMI - app.py (PENDIENTE)

Lineas 563-564: `df_master.at[idx, "BMI"] = val_bmi` aparece dos veces seguidas.

### 1.5 [FIX] SHAP - Titulo blanco sobre fondo blanco - gradcam_utils.py (PENDIENTE)

Linea 274: `plt.title("...", color="white")` sobre fondo `#ffffff`. Sigue sin corregir por el companero.

**Accion:** Cambiar a `color="#1e293b"` (gris oscuro, coherente con el diseno del frontend).

### 1.6 [NEW] Falta newline al final de app.py (NUEVO)

El commit del companero dejo el archivo sin salto de linea final. Anadir `\n` al final.

---

## FASE 2: Archivos Incompletos y Redundantes (SIN CAMBIOS)

### 2.1 Archivos a mover a `_archive/`

| Archivo | Razon |
|---------|-------|
| `modelo_busca_polipos_Segment.py` | Solo tiene un TODO, 0 codigo |
| `src/frontend/legacy/` (2 archivos) | Frontend antiguo |
| `src/utils/legacy/` (1 archivo) | Dependencies antiguo |
| `src/models/ml/legacy/` (3 archivos) | ML iteraciones v0-v2 |
| `src/scripts/test_dp_v2.py` | Script de prueba antiguo |
| `src/scripts/test_ml_v0.py` | Script de prueba antiguo |

### 2.2 Funcion `determinar_clase()` rota en data_cleaning.py (SIN CAMBIOS)

Sigue teniendo rutas hardcodeadas, recursion infinita potencial y emojis.

---

## FASE 3: Modelos de Biopsia (SIN CAMBIOS)

**Recomendacion:** Usar **ResNet18 (v0)** como principal. Ya integrado con la API. Conservar DenseNet121 (v1) como alternativa documentada. Mejorar comentarios en DenseNet.

---

## FASE 4: Registro de Resultados de Modelos (SIN CAMBIOS)

`ExperimentTracker` sigue sin usarse en ningun script de entrenamiento. Tres acciones:
1. Integrar tracker en `ml_v3.py` y `ml_v4_dual.py`
2. Integrar tracker en `modelo_biopsia_v0.py` y `modelo_busca_polipos_Clas.py`
3. Registrar tambien las metricas nuevas de dl_v5.py (precision/recall que anadio el companero)

---

## FASE 5: Evaluacion de Overfitting / Underfitting (SIN CAMBIOS)

Tres acciones:
1. Registrar historial de loss por epoca en el tracker
2. Generar graficos diagnosticos automaticos (train vs val loss)
3. Guardar graficos como imagen en `artifacts/` para documentacion

---

## FASE 6: Calidad de Codigo y Legibilidad (ACTUALIZADA)

### 6.1 Variables con nombres poco claros (SIN CAMBIOS)

### 6.2 Comentarios faltantes (ACTUALIZADO)

Archivos que necesitan mas comentarios:
- **ml_v4_dual.py**: 0 comentarios explicativos
- **DenseNet v1**: Falta documentar TransformSubset y evaluate_model
- **dl_v5.py**: Falta explicar la arquitectura de doble rama
- **gradcam_utils.py**: La nueva funcion `_extract_raw_shap()` (del companero) no tiene docstring. Debe tener uno explicando los formatos que maneja.

### 6.3 Iconos/Emojis a eliminar (SIN CAMBIOS)

| Archivo | Linea | Contenido |
|---------|-------|-----------|
| app.py (SHAP) | ~665 | Emoji en heading SHAP |
| main_api.py | 189, 192, 194 | Emojis en recomendaciones |
| data_cleaning.py | 287, 310, 313 | Emojis en prints |
| eda_visualization.py | 678, 681, 691, 700, 707 | Emojis en headings |

---

## FASE 7: UI/UX - Fondos, Contrastes y Colores (SIN CAMBIOS)

- SHAP titulo invisible (Fase 1.5)
- Tabs con contraste justo (#64748b sobre #f8fafc, ratio 4.45:1 -> mejorar a #475569)

---

## Resumen de Prioridades (ACTUALIZADO)

| Fase | Esfuerzo | Impacto en Rubrica | Estado |
|------|---------|-------------------|--------|
| 1. Bugs criticos | Bajo (25 min) | ALTO | 1/6 corregido, 5 pendientes + 1 nuevo |
| 2. Limpieza archivos | Bajo (15 min) | MEDIO | Pendiente |
| 3. Modelos biopsia | Medio (1h) | MEDIO | Pendiente |
| 4. Registro resultados | Medio (2h) | ALTO | Pendiente |
| 5. Evaluacion overfitting | Alto (3h) | ALTO | Pendiente |
| 6. Calidad codigo | Medio (1.5h) | ALTO | Pendiente (+1 item nuevo) |
| 7. UI/UX contraste | Bajo (30 min) | MEDIO | Pendiente |

## Open Questions

> [!IMPORTANT]
> 1. Quieres que **eliminemos** los archivos legacy o los movamos a una carpeta `_archive/`?
> 2. El modelo de segmentacion (`modelo_busca_polipos_Segment.py`) esta vacio. Quieres que lo **implementemos** o lo **eliminamos**?
> 3. Para el registro de resultados de modelos (Fase 4-5): prefieres graficos de overfitting **automaticos al entrenar** o bajo demanda con un comando?

## Plan de Verificacion

### Tests automaticos
```bash
pytest src/test/ -v --tb=short
```

### Verificacion manual
1. `python main.py api` -> Verificar 4 modelos cargados
2. `python main.py frontend` -> Sin errores de carga
3. Buscar paciente por DNI -> Formulario se rellena
4. Calcular riesgo (IA) -> SHAP visible con titulo legible
5. Subir colonoscopia -> Diagnostico + Grad-CAM
6. Subir biopsia -> Diagnostico + Grad-CAM
