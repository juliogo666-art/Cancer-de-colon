# Modelos de IA del Proyecto ColonAI

## Resumen

El proyecto utiliza 3 modelos de IA complementarios:

| Modelo | Tipo | Framework | Entrada | Salida | Archivo |
|--------|------|-----------|---------|--------|---------|
| LightGBM Clínico | Machine Learning | scikit-learn + LightGBM | 11 factores de riesgo | Low/Medium/High | `lgbm_clinico.pkl` |
| CNN Colonoscopia | Deep Learning | TensorFlow/Keras | Imagen 150×150 | Pólipo/Sano | `modelo_pro_agresivo.keras` |
| ResNet18 Biopsia | Deep Learning | PyTorch | Imagen 224×224 | Benigno/Maligno | `biopsia_resnet18_best.pth` |

---

## 1. Modelo ML — Predicción de Riesgo Clínico

**Qué hace**: Recibe 11 factores de riesgo de un paciente y predice si su riesgo de cáncer de colon es Bajo, Medio o Alto.

**Algoritmo**: LightGBM (Gradient Boosting Decision Tree). Es un "bosque de árboles de decisión" donde cada árbol corrige los errores del anterior.

**Las 11 features de entrada (en orden)**:
1. `Smoking` — Nivel de tabaquismo (0-10)
2. `Alcohol_Use` — Consumo de alcohol (0-10)
3. `Obesity` — Nivel de obesidad (0-10)
4. `Family_History` — Antecedentes familiares (0=No, 1=Sí)
5. `Diet_Red_Meat` — Consumo de carne roja (0-10)
6. `Diet_Salted_Processed` — Consumo de alimentos procesados (0-10)
7. `Fruit_Veg_Intake` — Consumo de frutas/verduras (0-10)
8. `Physical_Activity` — Actividad física (0-10)
9. `BMI` — Índice de masa corporal (kg/m²)
10. `FOBT_Resultado_n` — Test de sangre oculta en heces (0=Negativo, 1=Positivo)
11. `CEA_Level_ng_mL` — Nivel del marcador tumoral CEA

**Salida**:
- `predict_proba()` → Array con 3 probabilidades: [P(Low), P(Medium), P(High)]
- `predict()` → Clase predicha: 0 (Low), 1 (Medium), 2 (High)

**Archivo de entrenamiento**: `src/models/ml/ml_v3.py`

---

## 2. Modelo CNN — Detección de Pólipos en Colonoscopia

**Qué hace**: Analiza una imagen de colonoscopia y detecta si hay pólipos (crecimientos anómalos que pueden volverse cancerosos).

**Arquitectura**: MobileNetV2 (transfer learning) + capas personalizadas.

**Preprocesamiento**:
1. Redimensionar imagen a 150×150 píxeles
2. Normalizar con `preprocess_input()` de MobileNetV2 (rango [-1, 1])

**Interpretación de la salida**:
- Valor < 0.5 → **Pólipo detectado** (confianza = 1 - valor)
- Valor ≥ 0.5 → **Tejido sano** (confianza = valor)

**Grad-CAM**: Se genera un mapa de calor que muestra qué zonas de la imagen está mirando el modelo para tomar su decisión.

**Dataset de entrenamiento**: Imágenes de colonoscopia de HuggingFace, organizadas en carpetas `polipo/` y `sano/`.

---

## 3. Modelo ResNet18 — Clasificación de Biopsias

**Qué hace**: Analiza una microfotografía de biopsia de colon y clasifica si el tejido es benigno (normal) o maligno (adenocarcinoma).

**Arquitectura**: ResNet18 (pretrained en ImageNet) con la última capa reemplazada por una neurona con salida sigmoidal.

**Preprocesamiento**:
1. Redimensionar a 224×224 píxeles
2. Convertir a tensor PyTorch
3. Normalizar con media y desviación estándar de ImageNet:
   - Media: [0.485, 0.456, 0.406]
   - Desv: [0.229, 0.224, 0.225]

**Interpretación de la salida**:
- Sigmoid ≥ 0.5 → **Benigno** (normal)
- Sigmoid < 0.5 → **Maligno** (adenocarcinoma)

**Grad-CAM**: Se extrae de `model.model.layer4[-1]` (última capa convolucional de ResNet18).

**Archivo de entrenamiento**: `src/networks/dl_biopsia/modelo_biopsia_v0.py`

---

## Métricas de Evaluación

Para evaluar los modelos usamos métricas de clasificación clínica:

| Métrica | Qué mide | Por qué importa en oncología |
|---------|----------|------------------------------|
| Accuracy | % de aciertos totales | Visión general del rendimiento |
| Precision | De las alertas, ¿cuántas fueron reales? | Evitar alarmas innecesarias |
| **Recall** | **De los enfermos reales, ¿cuántos detectó?** | **La más crítica: no perder pacientes** |
| F2-Score | Media entre Precision y Recall (prioriza Recall) | Balance orientado a detección |
| ROC-AUC | Capacidad de discriminación del modelo | Evaluar calidad global del ranking |
| Confusion Matrix | Desglose TP/TN/FP/FN por clase | Identificar dónde falla exactamente |
