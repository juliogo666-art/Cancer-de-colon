# Requisitos y Casos de Uso del Simulador de Diagnóstico - Proyecto ColonAI

Este documento satisface las **evidencias de nivel 5** correspondientes a:
- *Documentación: Descripción detallada de los requisitos funcionales y no funcionales del simulador (M3RA4, M3RA3)*.
- *Casos de uso y escenarios de usuario (M3RA4)*.

---

## 1. Requisitos Funcionales

1. **RF01 - Predicción Clínica (Modelo Machine Learning):** El sistema debe permitir la introducción de un historial médico compuesto de 9 factores clínicos (tabaco, dieta, factores genéticos, índice de masa corporal, etc.) más 2 analíticas opcionales (FOBT y Marcador CEA) y devolver un nivel de riesgo clasificado en tres grados: Bajo, Medio, Alto.
2. **RF02 - Sistema de Triaje:** En caso de no proporcionar el análisis de sangre (CEA) o heces (FOBT), el sistema debe correr un modelo de Machine Learning parcial que sirva de *"Triaje"*, recomendando al usuario que se realice dichas pruebas analíticas para un diagnóstico confirmatorio.
3. **RF03 - Análisis Morfológico de Imágenes (Deep Learning):** El sistema debe contar con un submódulo capaz de escanear y analizar imágenes (ej. Radiografías o Colonoscopias) indicando si se visualiza la presencia de un pólipo.
4. **RF04 - Gestión de Base de Datos Clínicos:** El sistema debe permitir guardar pacientes evaluados y consultar sus historiales retrospectivamente usando un Dashboard y búsquedas sobre su Documento de Identidad (DNI) o su Número de Seguridad Social (NUSS).
5. **RF05 - Explicabilidad y Transparencia Clínica:** El sistema debe proporcionar un entorno de justificación algorítmica donde, mediante SHAP (Shapley Additive exPlanations) y/o Grad-CAM, se indique al facultativo qué valores exactos están influyendo en el diagnóstico.

---

## 2. Requisitos No Funcionales

1. **RNF01 - Tiempos de Respuesta de la Inferencia:** Las solicitudes de inferencia realizadas a las redes neuronales y algoritmos de ensamblaje (XGBoost/LightGBM) alojados en el backend no deben exceder de un tiempo de respuesta de 3.0 segundos en la carga nominal, para asegurar una Experiencia de Usuario óptima (UI/UX).
2. **RNF02 - Desacoplamiento Arquitectónico (API REST):** La interfaz visual (Frontend) debe estar arquitectónicamente divorciada e independiente del motor de cálculo (Backend) mediante protocolos HTTP/REST, permitiendo desplegar la capa de inteligencia en servidores GPU y la interfaz en Edge o Web (Streamlit/React).
3. **RNF03 - Internacionalización (i18n):** Todo el entorno interfaz debe poder bascular en caliente (hot-swap) entre idiomas soportados (Inglés y Español) sin que la sesión del facultativo se reinicie, usando un esquema `.YAML` centralizado.
4. **RNF04 - Tratamiento Estacionario (Fusión de Datos):** La base de datos debe almacenar exclusivamente información cruda ya estandarizada e inferida (0 y 1 categóricos), separando cualquier técnica de *oversampling* de la vista de uso.
5. **RNF05 - Robustez contra Subidas Erróneas:** El endpoint de la API dedicado a análisis de *biopsias* deberá estar protegido contra tamaños de imagen colosales o ficheros de extensiones corruptas limitando las subidas.

---

## 3. Casos de Uso y Escenarios de Usuario

### 3.1 Actores del Sistema
- **Médico General (Atención Primaria):** Solicita el triaje clínico inicial rellenando el formulario de hábitos (dieta, fumar, genética) del paciente.
- **Especialista (Oncólogo/Gastroenterólogo):** Adjunta y analiza los resultados de laboratorio completos (CEA, sangre oculta) e imágenes médicas a procesar por las Redes Neuronales.
- **Administrador de Datos / Investigador:** Requiere la exportación del CSV y control sobre las métricas (Logs) del sistema para futuros entrenamientos del modelo.

### 3.2 Diagrama Conceptual de Caso de Uso (Texto)

```
        +-------------------------+
        |   Atención Primaria     |
        +-------------------------+
                 | (1) Rellena Formulario Clínico (Fase 1)
                 v
   +------------------------------------+
   |   SISTEMA SIMULADOR (Modo Triaje)  |
   +------------------------------------+
                 | (2) Genera Score de Riesgo
                 | -> Si Alto: Recomienda Analítica + Colonoscopia
                 v
        +-------------------------+
        |  Gastroenterólogo (ESP) |
        +-------------------------+
                 | (3) Introduce Datos FOBT y CEA
                 | (4) Sube Imagen de la Colonoscopia
                 v
   +------------------------------------+
   |       SISTEMA SIMULADOR            |
   |      (Modo Cáncer + ResNet)        |
   +------------------------------------+
                 | (5) Confirmación Diagnóstica Final
                 | (6) Genera Explicación SHAP / Mapa de Calor
                 v
        +-------------------------+
        |        Paciente         |
        +-------------------------+
```

### 3.3 Escenarios de Usuario (Paso a Paso)

#### Escenario 1: Triaje de Paciente Sano con malos hábitos
- **Descripción:** Un paciente varón de 45 años, fumador y sedentario asiste a su médico.
- **Paso 1:** El Médico busca al paciente usando su DNI. Aparece vacío porque no tiene datos previos. Registra sus datos básicos, indicando mala dieta y tabaquismo (rango 8/10). No tiene analíticas.
- **Paso 2:** El Médico pulsa en "Calcular Riesgo (Asistente de IA)". 
- **Paso 3:** El algoritmo procesa los datos y determina un Riesgo *Medio*. 
- **Paso 4:** El sistema advierte en pantalla: *"Es necesario realizar analítica de sangre (CEA) y muestra de heces (FOBT) para confirmar diagnóstico"*. El Doctor programa dichas pruebas.

#### Escenario 2: Análisis Confirmatorio por Especialista (Positivo)
- **Descripción:** El paciente anterior vuelve con una colonoscopia de la cual un pólipo es extraıdo, además de una prueba confirmada positiva de cáncer (CEA > 6 ng/mL).
- **Paso 1:** El gastroenterólogo carga al paciente y anota CEA=6.2 ng/mL, Sangre Oculta = "Positivo".
- **Paso 2:** Se recalculan los valores con el botón "Calcular" y el riesgo es *High*. Aparece un diagrama SHAP demostrando que el alto peso recae sobre el valor CEA aportado.
- **Paso 3:** El doctor escanea la radiografía en el segmento "Visión de Imágenes (Deep Learning)". El sistema detecta "Pólipo Maligno". El doctor programa al paciente urgente.
