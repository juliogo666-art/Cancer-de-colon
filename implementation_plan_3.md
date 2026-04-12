# Unificación de Base de Datos de Pacientes

El sistema actual sufre de fragmentación y desincronización de datos al operar sobre 3 archivos independientes en `src/data/clean/`:
- `nuevos_pacientes_5000.csv` (Usado por Streamlit para DNI, Nombres, Edad, Género)
- `cancer_risk_clean.csv` (Usado por Streamlit para variables clínicas, pero sobreescribe Edad y Género)
- `cancer_risk_final.csv` (Usado por el backend FastAPI, no tiene Edad ni Género pero sí tiene FOBT y CEA)

## Objetivo
Consolidar toda la información de los pacientes en un único archivo maestro y actualizar todo el código del repositorio (Frontend, Backend y Ajustes) para que lean/escriban **exclusivamente** de este archivo garantizando integridad.

## User Review Required
> [!IMPORTANT]
> - **Ruta Única:** El CSV final unificado lo alojaremos en `src/data/ready/pacientes_master.csv`.
> - **Género:** Existen valores numéricos (0/1) en *clean* y strings (Female/Male) en *nuevos_5000*. El nuevo dataset maestro usará de base el string (Female/Male) para que Streamlit pueda renderizarlo directamente con facilidad en el formulario y mantendremos consistencia visual. Para el modelo de IA esto es indiferente porque el modelo LightGBM **no utiliza el género ni la edad** en sus 11 features de entrenamiento según está codificado `ml_v3.py`.
> - ¿Estás de acuerdo con este enfoque?

## Proposed Changes

### Archivos de Datos y Scripts Utilitarios
---
#### [NEW] `src/data/ready/pacientes_master.csv`
Crearemos un script de usar y tirar (`scratch/unify_csv.py`) que leerá los 3 archivos CSV actuales, hará un cruce lógico por `Patient_ID` y tomará:
- DNI, Nombre, Apellidos, Ciudad, Edad, Gender (Female/Male) de `nuevos_pacientes_5000.csv`.
- Variables clínicas (Smoking, Obesity, etc.) y `FOBT_Resultado_n`, `CEA_Level_ng_mL` de `cancer_risk_final.csv`.
Esto generará el dataset maestro consolidado en tu nueva carpeta `ready/`.

#### [MODIFY] `src/config/settings.py`
Sustituiremos las variables independientes de los distintos CSV por una única ruta maestra:
- [DELETE] `CSV_RISK_PATH`, `CSV_RISK_CLEAN_PATH`, `CSV_PACIENTES_5000_PATH`
- [NEW] `CSV_MASTER_PATH = os.path.join(RAIZ_DEL_PROYECTO, "src", "data", "ready", "pacientes_master.csv")`

### Backend y Frontend
---
#### [MODIFY] `src/api/main_api.py`
Actualizar todos los sub-endpoints gestionados por la API (`GET /api/v1/patients`, `POST`, `PUT`) para que lean, reescriban y mantengan todos los campos extendidos usando el `CSV_MASTER_PATH`.

#### [MODIFY] `src/frontend/app.py`
Refactorizar de arriba abajo la gestión documental de Streamlit:
1. Eliminar la lectura doble de datasets. Ahora solo cargará `df_master = pd.read_csv(settings.CSV_MASTER_PATH)`.
2. Actualizar las funciones de búsqueda `buscar_paciente_por_documento()` para trabajar sobre el DataFrame único.
3. Actualizar la inyección de parámetros del clínico en el formulario (porcentaje de tabaquismo, etc.) asegurando que se extraen y se sobreescriben correctamente sobre ese mismo gran DataFrame único que luego es guardado por pandas con `to_csv(settings.CSV_MASTER_PATH)`.

## Open Questions
- ¿Los archivos de la carpeta `src/data/clean/` seguirán existiendo como historial base, o preferirías que añada al plan la eliminación técnica de dichos archivos para no dejar basura una vez creado el máster en `ready/`?

## Verification Plan
1. Se ejecutará el script de consolidación unificando los miles de registros en 1 solo CSV.
2. Comprobaremos con PowerShell la estructura de la cabecera en el nuevo `pacientes_master.csv` para ver que tiene todo (DNI, Nombres, Edad, Valores Clínicos, Pruebas y Riesgos).
3. Levantaré la API y haré un request de prueba (`curl` o requests) buscando un paciente específico, confirmando el volcado unificado.
4. (Requerida manual) El usuario procederá a utilizar Streamlit, buscará un DNI al azar y comprobará que el Género y la Edad ahora son consistentes en toda la app.
