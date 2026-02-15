# Descripción del proyecto

#######################################################################################

El clinico necesita desarrollar un programa que al entregarle los datos del usuario analiza la información y predice si tienes o no cancer de colon.

\#\# Quén tenemos

1. Datos del paciente
   1. Historial medico
      1. Analitica general
      2. electro cardiograma
      3. tensión  
      4. .....
   2. Video del colon
2. Pruebas y resultados de diagnostico general
   1. Imagenes de polipos cancerigenos y sin
   2. Historial médico ficticio del paciente a tratar: datos vitales e información de carácter hereditario.  
   3. Pruebas y resultados de test de diagnóstico general  
   4. Pruebas y resultados de test de diagnósticos tumorales

Comparar imagenes con y sin pólipos

compara con datos históricos de los pacientes con pacientes que tienen cáncer mediante una correlación y verificar si necesita una colonoscopia o no, y mirar si este paciente tendrá una 3 fase o no.

tener en consideracion si toma medicacion, si fuma, bebe, un poco sus habitos

\#\# Cuál es el flujo

1. INPUTS
   1. Datos del paciente.  
   2. Imagenes/video colonoscopia.  
2. Esto va a una DB  
   1. Consulta los datos de pólipos o relacionados con estos 
   2. Saca los datos necesarios.  
3. Vectorizados la información  
4. Inferencia del modelo.  
