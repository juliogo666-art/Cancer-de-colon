# Resumen de Integración Clínico-Triaje

Se ha desplegado en todo el software el modelo híbrido (clínico-triaje) para poder atender a pacientes estén o no probados contra perfiles analíticos.

## 1. El Cerebro (Modelos ML)
El script de entrenamiento en la trastienda ha sido sustituido por `ml_v4_dual.py`. Cuando lo he activado, ha destilado de los 5000 registros dos subredes de memoria LightGBM independientes:
1. `lgbm_clinico.pkl`: El modelo completo que se sabe *todo* sobre los riesgos con 11 variables y una precisión masiva del **95.6%**.
2. `lgbm_triage.pkl`: Un modelo entrenado en cuarentena excluyendo intencionalmente el FOBT y el CEA. Su precisión bajó a un respetable **88.9%**, que es brutal teniendo en cuenta que no sabe mirar la sangre del paciente.

## 2. El Router (Backend - FastAPI)
Al arrancar el servidor `main_api.py` a través de sus dependencias, ahora detecta y sube a RAM ambas "mentes" de manera conjunta (`4/4 modelos cargados correctamente`).
El router `/predict/risk` vigila la carga útil (Payload) que llega desde la página web:
- Si detecta un paciente en donde `fobt_resultado` viene marcado como desconocido (`-1`), intercepta el tráfico y se lo envía al modelo `lgbm_triage`. 
- Además, incrusta activamente una "Recomendación" avisando al médico de que "es urgente mandar a hacer una analítica y muestra heces y volver luego".

## 3. El Diseño Visual (Frontend)
Debajo del Historial Familiar verás una nueva fila llamada **Analíticas Confirmadas (Dejar en Blanco si Triaje)**:
- Un dropdown para seleccionar: `Desconocido (-) / Negativo / Positivo` para las Heces.
- Un cuadro de texto en blanco para poner el valor del marcador si es que existe.
- Si le das a _Calcular_, la **caja de Diagnóstico Verde** se va a expandir y vas a ver aparecer un **Mensaje de Alerta en Texto Rojo** dictado por la Inteligencia Artificial (por ejemplo: `🚨 Riesgo elevado confirmado por marcadores analíticos. Se recomienda derivación urgente para COLONOSCOPIA`). 

> [!TIP]
> Dado que la arquitectura del backend "uvicorn" tiene hot-reload en `main.py`, tu máquina ya ha auto-cargado todos los 4 pesos neuronales nuevos. ¡Ya puedes refrescar el navegador y ver tu nuevo módulo clínico en funcionamiento!
