# Implementación: Flujo "Nuevo Paciente"

He completado el diseño del flujo de nuevo registro tal y como planteaste en las 4 secuencias de un entorno clínico real. Ahora los médicos pueden "dar de alta" a pacientes vírgenes sin depender de bases de datos externas de simulación.

## Detalles de la Implementación

### 1. Interfaz Adaptativa (Modo Creación)
- Al pulsar el nuevo botón verde **"🟢 Nuevo Paciente"**, toda la memoria de la sesión hace un *reset*.
- La clásica ficha "verde" de paciente (arriba a la derecha) **se transforma visualmente en un sub-formulario de identidad** (Cuatro cajas para DNI, Nombre, Primer y Segundo apellido).
- El resto de métricas y variables clínicas se quedan a 0 o en su valor neutral, a la espera de que el facultativo las introduzca conforme a la anamnesis en la consulta.

### 2. Guardado Atómico Inteligente
- Si decides pulsar "Calcular", viajará al backend perfectamente (LightGBM lo evaluará en caliente y escupirá tu recomendación).
- Al hacer clic sobre "Guardar", el sistema no machaca a otro paciente:
	1. Escanea el archivo general `pacientes_master.csv`.
	2. Identifica el `Patient_ID` más alto y autogenera el siguiente consecutivo (e.g., 5001, 5002...).
	3. Recupera los campos de Nombre y DNI del entorno dinámico y los une con las métricas clínicas en tu nuevo dataframe consolidado de 27-Features.
	4. Realiza el guardado final en `src/data/ready/pacientes_master.csv`.

### 3. Transición Transparente
- Instantáneamente después de guardarse, el Streamlit oculta mágicamente las "cajas" de DNI/Nombre y vuelve a renderizar el modo "Ficha Verde", asumiendo en memoria que ahora estás editando a este mismo paciente que acabas de meter. 

> [!TIP]
> Prueba a recargar la página y pulsar el botón **"🟢 Nuevo Paciente"** para ver los cambios visuales y crear una persona de prueba.
