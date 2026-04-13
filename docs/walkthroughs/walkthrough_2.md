# Walkthrough — Sesión 8 de Abril (noche)

## Resumen

Se han creado **8 archivos nuevos** y modificado **2 existentes** para completar los componentes 6-7 del plan de implementación.

---

## Archivos Creados

### 1. Frontend Unificado
| Archivo | Líneas | Descripción |
|---------|--------|-------------|
| [app.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/frontend/app.py) | ~420 | Fusión de `app_s_busqueda.py` + `app_s.py` |

**Lo que tomó de cada uno:**
- De `app_s_busqueda.py`: formulario completo con todos los campos, búsqueda por DNI/NUSS, botones de CRUD
- De `app_s.py`: predicción ML con SHAP, análisis de biopsias (PyTorch), radio button para tipo de imagen, layout lado a lado con Grad-CAM
- **Nuevo**: Usa `settings.py` para rutas centralizadas

### 2. Configuración Centralizada
| Archivo | Líneas | Descripción |
|---------|--------|-------------|
| [settings.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/config/settings.py) | ~170 | Todas las rutas y constantes del proyecto |

### 3. Tests Reales con Pytest
| Archivo | Tests | Descripción |
|---------|-------|-------------|
| [test_metrics.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/test/test_metrics.py) | 17 | Tests de las 6 métricas clínicas |
| [test_schemas.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/test/test_schemas.py) | 12 | Tests de validación Pydantic |
| [test_api.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/test/test_api.py) | 6 | Tests de integración de la API |

**Resultado**: ✅ 29/29 tests pasados

### 4. Documentación
| Archivo | Contenido |
|---------|-----------|
| [docs/arquitectura.md](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/docs/arquitectura.md) | Estructura del proyecto, flujo de datos |
| [docs/modelos.md](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/docs/modelos.md) | Los 3 modelos: ML, CNN, ResNet18 con sus inputs/outputs |
| [docs/api_endpoints.md](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/docs/api_endpoints.md) | Documentación de todos los endpoints |

---

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| [main.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/main.py) | Reescrito con 6 comandos: `api`, `frontend`, `eda`, `train-ml`, `test`, `generate-data` |
| [patient_schemas.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/schemas/patient_schemas.py) | Fix: `class Config` → `model_config = ConfigDict()` (Pydantic V2) |

---

## Cómo Arrancar

```bash
# Ejecutar el frontend
streamlit run src/frontend/app.py

# O desde main.py
python main.py frontend

# Arrancar la API
python main.py api

# Ejecutar tests
python main.py test

# Entrenar modelo ML
python main.py train-ml
```

---

## Pendientes para próxima sesión

**Hechos 1 y 2**
3. **Conectar predicción con logger**: Que `app.py` llame a `PredictionLogger` después de cada predicción
4. **Integrar API con frontend**: Opcionalmente hacer que Streamlit haga HTTP requests a la API en vez de cargar modelos directamente
