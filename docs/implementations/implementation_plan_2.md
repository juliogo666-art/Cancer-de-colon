# Plan de Implementación — Fase Final: Internacionalización y Arquitectura API

Este plan ha sido refinado con base en las directrices del equipo para asegurar que la aplicación sea multilingüe, escalable y trazable.

## Goal Description & Respuestas Abiertas

**Sobre el Idioma (i18n)**:
Implementaremos un sistema de internacionalización dinámico. Utilizaremos **un solo archivo YAML** llamado `translations.yaml` en la carpeta `src/config/`. Este enfoque es el mejor porque centraliza todos los textos y permite añadir idiomas futuros súper rápido (usando estructuras como `es: { title: "Hola" }, en: { title: "Hello" }`). El frontend añadirá un desplegable o un par de botones en el sidebar para cambiar el idioma en caliente (hot reload).

**¿Por qué desacoplar el Frontend y la API? (Llamadas HTTP vs Imports Locales)**:
1. **Rendimiento:** Cargar un modelo de Machine Learning y, en especial, modelos profundos de Deep Learning (CNNs), exige mucha Memoria RAM/VRAM. Si Streamlit carga estos modelos internamente cada vez que un usuario entra o recarga, la app pesará muchísimo y colapsará.
2. **Escalabilidad:** Si el modelo está en una API externa (FastAPI), este se carga *una sola vez en memoria* al arrancar el servidor. Cuando Streamlit le hace un `requests.post()`, el modelo ya está encendido, recibe los datos instantáneamente, infiere y devuelve el resultado en milisegundos.
3. **Versatilidad:** Si el día de mañana tu equipo crea una App en iOS/Android, esa nueva app no puede correr código en Python; ¿solución? Se conectará a vuestra API directamente sin pasar por Streamlit.

**Política de Conservación (Legacy)**:
Ningún script clásico se borrará. Serán confinados a la ruta `src/legacy/` o similar para que estén a la mano en caso de auditorías, fallos, o necesidad de código retrospectivo.

---

## Proposed Changes

### 1. Internacionalización con YAML (Frontend)

El frontend requiere un selector de idiomas y la externalización de textos.

#### [NEW] [translations.yaml](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/config/translations.yaml)
- **Contenido:** Diccionario estructurado conteniendo un árbol de literales para `es` (Español) y `en` (Inglés) describiendo los nombres de variables, modales y botones en formato humano (ej. "nivel_obesidad", "calcular_riesgo").

#### [MODIFY] [app.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/frontend/app.py)
- **Desplegable de Idioma:** En el menú del layout, permitirá cambiar entre "ES" y "EN".
- **Variables Intuitivas:** Renombraremos todas las variables al castellano (por ejemplo: `archivo_yaml`, `texto_boton_calcular`) y añadiremos amplios comentarios explicando qué hace cada bloque (tal cual has pedido).

---

### 2. Desacoplamiento e Integración Frontend - API

Hacer que Streamlit "hable" vía red (REST) con los endpoints, en lugar de invocar las funciones complejas embebidas.

#### [MODIFY] [app.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/frontend/app.py)
- Reemplazar las funciones `predecir()`, `colonos()` y `biopsias()` por funciones que hacen peticiones HTTP asíncronas / síncronas contra `http://localhost:8000/api/v1/...` donde está la API.
- Re-escribir usando comentarios claros.

#### [MOVE] [cargar_modelos_s.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/utils/cargar_modelos_s.py) -> [legacy/cargar_modelos_s.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/legacy/cargar_modelos_s.py)
- Ya no se usa directamente por el frontend. Se mueve a la carpeta `legacy/` al igual que otros archivos no utilizados para conservarlos.

#### [MODIFY] [main_api.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/api/main_api.py)
- Afinar los esquemas Pydantic para garantizar que la API procese exitosamente lo que recibe del Frontend `app.py`.

---

### 3. Trazabilidad e Integración (Backend API Logger)

Aplicar el registro exhaustivo sobre cada solicitud. 

#### [MODIFY] [main_api.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/api/main_api.py)
- **Inyección del Logger:** Inyectar clase `PredictionLogger` dentro del endpoint de riesgo clínico.
- **Flujo:** La API recibe los datos de paciente desde app.py -> predice -> el PredictionLogger escribe el registro (ID y su evolución) -> se envía la respuesta de vuelta al `app.py`.

#### [MODIFY] [prediction_logger.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%202%20-%20Cancer%20colon/src/tracking/prediction_logger.py)
- Asegurar que escriba con un formato limpio en CSV e incluya timestamp. El código será comentado paso a paso detallando cómo y por qué se guarda la evolución histórica del paciente.

---

## Verificación
Todo esto se validará levantando tanto la API (`python main.py api`) como el Fronend (`python main.py frontend`) simulando el escenario productivo final completo y validando que el logger genera las trazas.
