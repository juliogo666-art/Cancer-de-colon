# Contexto

Estos dos conjuntos de datos consisten en un grupo de pacientes con cáncer colorrectal, que han sido operados para extirpar su tumor.

El conjunto de datos contiene datos de pacientes y otro sus respectivos niveles de expresión génica.

# Contenido

El conjunto de datos "Paciente con Cáncer Colorrectal Data.csv" consiste en el grupo de información sobre pacientes con cáncer colorrectal

El conjunto de datos de pacientes consta de las siguientes variables:

- Edad: en el momento del diagnóstico (en años)
- Etapa de Dukes: A a D (desarrollo/progresión de la enfermedad)
- Género: Hombre o mujer
- Localización: izquierda, derecha, colon o recto
- DFS: Supervivencia libre de enfermedad, meses (supervivencia sin que la enfermedad regrese)
- Evento DFS: 0 o 1 (con 1 = evento)
- Adj_Radio: Si el paciente también recibió radioterapia
- Adj_Chem: Si el paciente también recibió quimioterapia

El conjunto de datos de expresión génica "Colorrectal Cancer Gene Expression Data.csv" comprende los niveles de expresión génica para el mismo conjunto de pacientes.

Estos datos han sido preprocesados y transformados en log2. No necesitas hacer más transformaciones en los datos.