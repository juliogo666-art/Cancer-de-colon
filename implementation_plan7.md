# Plan de Auditoría y Cumplimiento de Rúbricas (Proyecto Cáncer de Colon)

El objetivo de este plan es añadir los componentes documentales y técnicos que garanticen el **Nivel 5** en todas las evidencias requeridas por el documento `P2_Enunciado_text.txt`, con especial foco en `M3RA1, M3RA2, M3RA3, y M3RA4`.

## User Review Required

> [!IMPORTANT]
> Revisa este plan para confirmar si deseas que implemente **todas** estas mejoras o si prefieres omitir alguna. Todas ellas van directas a mejorar la calificación del proyecto aportando las evidencias que exigen las rúbricas.

## Proposed Changes

A continuación se detalla la hoja de ruta para completar la auditoría del proyecto. Se ha dividido por componentes lógicos.

---

### Componente 1: Testing y Evidencias de Cobertura (M3RA4, M3RA3)

Configuraremos el entorno de pruebas para poder extraer fácilmente un informe de cobertura (Coverage Report) que demuestre de forma técnica y numérica la fiabilidad del código. Te servirá para adjuntarlo como captura en la documentación final.

#### [NEW] [pytest.ini](file:///pytest.ini)
- Creación de un archivo de configuración de pytest en la raíz para establecer las carpetas de pruebas y las métricas de cobertura necesarias.

#### [NEW] [run_tests.ps1](file:///run_tests.ps1)
- Pequeño script en PowerShell para que puedas lanzar todos los tests automatizados (`pytest --cov=src --cov-report=html`) con un solo clic. Esto generará la carpeta `htmlcov/` con los resultados visuales de tus pruebas.

#### [MODIFY] [requirements.txt](file:///src/requirements.txt)
- Añadiremos librerías necesarias como `pytest-cov` de forma explícita.

---

### Componente 2: Evidencias de Análisis y Rendimiento (M3RA4, M3RA3)

La rúbrica demanda "Análisis de la precisión y rendimiento utilizando conjuntos de datos". Vamos a generar de forma automática unas gráficas para que puedas pegarlas en tu trabajo escrito.

#### [NEW] [generate_metrics_plots.py](file:///src/scripts/generate_metrics_plots.py)
- Se creará un script dedicado a generar imágenes PNG (Curvas ROC, Matrices de Confusión) leyendo los resultados generados en las predicciones anteriores o simulando una pasada de evaluación final.
- **Salida esperada:** Una carpeta nueva `artifacts/plots/` con ficheros como `roc_curve.png`, `confusion_matrix.png`, etc.

---

### Componente 3: Documentación Técnica de Requisitos y Casos de Uso (M3RA4, M3RA1, M3RA2)

Es crucial tener un documento explícito al que referenciar en tu entrega.

#### [NEW] [requisitos_y_casos_uso.md](file:///docs/requisitos_y_casos_uso.md)
- Este documento incluirá:
  - **Requisitos Funcionales:** Comportamiento esperado de la API, flujos del Frontend y gestión del CSV.
  - **Requisitos No Funcionales:** Requerimientos de arquitectura (desacople), tiempos de carga de modelos CNN, y escalabilidad.
  - **Casos de Uso Principales:** Descripción paso a paso de los perfiles que usarían tu sistema y diagrama conceptual en texto.

---

### Componente 4: Robustez y Refactorización Menor

Detalles de código que suman profesionalidad al proyecto.

#### [MODIFY] [main_api.py](file:///src/api/main_api.py)
- **Mejora:** Validación del tamaño del archivo. Se validará que las imágenes subidas a los *endpoints* de colonoscopia y biopsia no superen un tamaño razonable (p.ej. 5MB) devolviendo un código de error `413 Payload Too Large`. Ayuda a la validación de robustez.

#### [MODIFY] [training_pipeline.py](file:///src/pipelines/training_pipeline.py)
- **Mejora:** Inclusión de la biblioteca estándar `logging` para sustituir los simples `print()`, proveyendo logs temporizados `[INFO] [2026-...] Modelo entrenado`.

## Open Questions

> [!NOTE]
> Por favor, confírmame:
> 1. ¿Utilizas Windows para tus ejecuciones locales? (He propuesto `.ps1` para PowerShell, indícame si prefieres un `.bat` clásico).
> 2. Una vez genere todo esto, te quedará pendiente grabar el **Video Demostración** y hacer un **Manual de Usuario** (en inglés). ¿Deseas que te redacte también la base del manual de usuario en la capeta `docs/`?

## Verification Plan

### Automated Tests
- Correremos el nuevo `run_tests.ps1` en tu entorno (aprobado por el usuario), lo que abrirá / generará el reporte `.html` de cobertura en `htmlcov/index.html`.
  
### Manual Verification
- Comprobaremos la creación de gráficas visuales ejecutando el script `generate_metrics_plots.py`.
- Leeremos la estructura de la nueva documentación para corroborar que cumple punto por punto con los descriptores de "Nivel 5" en los RA3 y RA4.
