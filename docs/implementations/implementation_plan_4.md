# Implementación "Nuevo Paciente"

Ahora que la base de datos es unificada y robusta (`pacientes_master.csv`), la creación de nuevos pacientes es predecible y segura. 

## Objetivo
Añadir un botón en la interfaz de Streamlit que habilite un modo de "Creación", permitiendo al doctor introducir por primera vez los datos personales (DNI, Nombre, etc.) junto con los clínicos, evaluar el riesgo inicial y guardarlo en la base de datos de manera atómica.

## User Review Required
> [!IMPORTANT]
> El diseño actual de la UI solo mostraba el Nombre y DNI como "Solo Lectura" en una caja verde si el paciente ya existía. En el modo "Nuevo Paciente", proponemos que esa caja verde se sustituya (o se acompañe) por una serie de cajas de texto `st.text_input` para que el médico pueda escribir el DNI, Nombre y Apellidos manualmente. ¿Te parece bien este cambio visual?

## Proposed Changes

### Frontend (`src/frontend/app.py`)
---
#### [MODIFY] `app.py - Sección de Búsqueda`
Añadimos el botón `Nuevo Paciente` antes de Búsqueda y Cargar.
- En el `st.session_state` crearemos un booleano `"modo_nuevo_paciente": False`.
- Al pulsar el botón, esto pasa a `True`, se limpian los campos actuales (edad, tabaco, bmi a 0), y se oculta el error de "Paciente no encontrado".

#### [MODIFY] `app.py - Sección Fila Paciente (Mitad Superior)`
Modificaremos la representación de la ficha del paciente:
- Si `"modo_nuevo_paciente"` es `False` y hay datos: Muestra la caja HTML verde de solo lectura (como está ahora).
- Si `"modo_nuevo_paciente"` es `True`: Muestra campos interactivos para DNI, Nombre, Apellido 1 y Apellido 2 usando columnas de Streamlit.

#### [MODIFY] `app.py - Botón CALCULAR`
No requiere apenas cambios. El modelo LightGBM solo evalúa parámetros clínicos, así que puede calcular el riesgo esté guardado el paciente o no (le pasaremos un ID temporal ficticio o None en la petición JSON de FASTAPI).

#### [MODIFY] `app.py - Botón GUARDAR`
Si el paciente es nuevo, la lógica de guardado hará lo siguiente:
1. Buscará el `Patient_ID` más alto actual en `df_master` y le sumará 1 (`max() + 1`).
2. Creará un registro (`dict`) de 27 columnas (mapeando correctamente el DNI, Nombre, los campos en blanco por defecto para FOBT y CEA, y los datos numéricos clínicos que puso el doctor).
3. Añadirá ese diccionario a `df_master` vía `pd.concat()`.
4. Lo guardará en `pacientes_master.csv`.
5. Cambiará el `"modo_nuevo_paciente"` a `False` para transformarlo en un paciente "Cargado" normal visualmente (modo solo lectura).

## Verification Plan
1. Iniciaremos Streamlit en el puerto por defecto.
2. Comprobaremos que "Nuevo Paciente" limpia el formulario.
3. Crearemos a un "John Doe" con DNI inventado.
4. Ejecutaremos el cálculo (para ver que recibe un "Low" de riesgo, por ejemplo).
5. Guardaremos a John Doe.
6. Refrescaremos la app y buscaremos ese DNI para confirmar la inserción de estado y persistencia de memoria.
