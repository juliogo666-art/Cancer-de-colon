# Implementación de Pipeline ML Dual y Recomendaciones Clínicas

El razonamiento es biomecánicamente perfecto. Si evaluamos a un paciente que entra por la puerta sin analíticas, y le forzamos valores `FOBT=0` y `CEA=0`, la IA entenderá falsamente que "está 100% sano" según los marcadores tumorales, aplastando su riesgo real por estilo de vida.

## Solución: Arquitectura de "Dos Fases" (Triaje vs Clínico)
Vamos a estructurar el sistema para que soporte dos "cerebros" dependiendo del estado del paciente.

## Proposed Changes

### 1. Modelos de Machine Learning (`src/models/ml`)
---
#### [NEW] `src/models/ml/ml_v4_dual.py`
Crearé un nuevo programa unificado de entrenamiento que generará 2 modelos simultáneamente desde tu `cancer_risk_final.csv`:
- `lgbm_clinico.pkl`: Entrenado con 11 variables (para cuando SÍ hay analíticas).
- `lgbm_triage.pkl`: Entrenado solo con 9 variables de estilo de vida/demográficas. Excluye FOBT y CEA. Aprenderá a calcular el riesgo basándose únicamente en el entorno del paciente.

### 2. Backend FastAPI (`src/api`)
---
#### [MODIFY] `dependencies.py`
- Cargará en la memoria del servidor ambós modelos: `modelo_ml_clinico` y `modelo_ml_triage`.

#### [MODIFY] `main_api.py` (Endpoint `/predict/risk`)
- Aceptará un estado "Desconocido" / `-1` para `fobt_resultado` y `cea_level`.
- **Enrutador Dinámico:** 
  - Si `fobt_resultado == -1`, utilizará el modelo de Triaje (9 variables). Añadirá al JSON de respuesta el campo: `recommendation: "Es necesario realizar analítica de sangre (CEA) y muestra de heces (FOBT)."`
  - Si existen resultados, usará el modelo Clínico Completo. Si la predicción resulta `Medium` o `High`, añadirá: `recommendation: "Riesgo elevado confirmado por analíticas. Se recomienda derivación para COLONOSCOPIA."`

### 3. Frontend Streamlit (`src/frontend/app.py`)
---
#### [MODIFY] `app.py`
- Debajo del "Historial Familiar", añadiré nuevos controles interactivos para que el médico lea y anote las analíticas corporales:
  - **FOBT (Sangre en Heces):** Desplegable con opciones `["Desconocido", "Negativo", "Positivo"]`.
  - **Nivel CEA (ng/mL):** Selector Numérico. Si no se ha hecho, se puede dejar una casilla vacía o representar con `-1.0`.
- Cuando el doctor pulse "CALCULAR RIESGO (IA)" y el paciente sea `Desconocido`, el frontend enviará los `-1` a la API, disparando la red neuronal de triaje.
- Añadiré una sección visual debajo de "Riesgo Global" que muestre en pantalla la **Recomendación Clínica Textual** que devuelva el servidor.
- La lógica del botón de guardar adaptará el `FOBT_Resultado` de guardado a `Desconocido` para los pacientes "nuevos" (en lugar de forzar un Negativo).

## User Review Required
> [!IMPORTANT]
> ¿Estás de acuerdo con el concepto de añadir el estado `Desconocido` a la base de datos para pacientes vírgenes de analíticas (mapeado como `-1`) para que el sistema active automáticamente la vía rápida de *Triaje*?

## Verification Plan
1. Ejecutar el script `ml_v4_dual.py` para generar ambos pesos.
2. Reiniciar el clúster de Uvicorn/FastAPI.
3. Buscar tu paciente `5000` (Adam, el nuevo) en la web.
4. Con CEA en desconocido, dar a Calcular: debería evaluar un riesgo basándose en que Adam fuma, bebe y es obeso, dándole un "High Risk" sugerido y una recomendación de analizar sus heces.
5. Editar la web para añadir sus resultados de heces (Positivo), y al dar a Calcular, el modelo Clínico debería coger el timón y recomendar enviar a Adam a la cama de colonoscopia.
