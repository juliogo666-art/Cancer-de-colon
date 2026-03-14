# Implementación del Modelo de Detección de Pólipos

## Contexto y Objetivo

El objetivo es desarrollar un modelo en PyTorch capaz de procesar imágenes de colonoscopias para clasificar o detectar la presencia de pólipos. Utilizaremos el dataset de HuggingFace descargado en `src/data/raw/colonoscopic_processed`.

## Análisis del Dataset

Utilizaremos el dataset de HuggingFace: `sageofai/colonoscopy_data_for_vqa`.
Este dataset está pensado originalmente para responder preguntas (Visual Question Answering), pero resulta perfecto porque tiene las etiquetas.
Cada registro contiene:

1. La `image` (la foto de la colonoscopia).
2. Un `question` (ej. "Does the image display a polyp?").
3. Un `answer` (ej. "Yes." o "No.").

Con este dataset, no hace falta que descargar las imágenes. Usaremos la librería oficial `datasets` de HuggingFace dentro del propio código para que descargue el paquete completo y lo lea directamente en memoria, filtrando las respuestas para obtener nuestras etiquetas 1 para "Yes", 0 para "No".

## Arquitectura

### 1. Modelo Base (Transfer Learning)

Utilizaremos **ResNet18** preentrenado en ImageNet por su excelente balance entre velocidad computacional y precisión para extraer características visuales de las colonoscopias.

- [NEW] `src/models/modelo_busca_polipos.py`: Se definirá la clase `PolypDetector(nn.Module)` que heredará de `resnet18`. Reemplazaremos la última capa fully-connected para que tenga la salida deseada (ej. 2 clases: Pólipo / No Pólipo).

### 2. Pipeline de Datos (Dataset y Dataloader)

Necesitamos preparar las imágenes para que PyTorch las entienda.

- [MODIFY] `src/models/modelo_busca_polipos.py`:
  - Crearemos un `ColonoscopyDataset(Dataset)` que usará `load_dataset("sageofai/colonoscopy_data_for_vqa")`.
  - El dataset leerá el texto `answer` y lo convertirá en tensor binario `[1.0]` si es "Yes." y `[0.0]` si es "No.".
  - Implementaremos _Data Augmentation_ usando `torchvision.transforms`: `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter` y normalización estándar para mejorar la fiabilidad del modelo.

### 3. Loop de Entrenamiento y Evaluación

- **Loss Function (Función de pérdida)**: `BCEWithLogitsLoss` o `CrossEntropyLoss` dependiendo de si lo formulamos binario o multiclase. Si las clases están desbalanceadas, añadiremos pesos.
- **Optimizer**: `Adam` con `learning_rate` en torno a `1e-4` y un _Learning Rate Scheduler_ (`ReduceLROnPlateau`).
- Crearemos funciones separadas `train_epoch()` y `evaluate()`.

## Flujo de Trabajo

1. Instalar dependencias (`uv add torch torchvision datasets`).
2. Implementar el Dataset integrado con HuggingFace en `modelo_busca_polipos.py`.
3. Cargar ResNet18 y modificar su clasificador.
4. Escribir el ciclo de optimización (training loop).
5. Guardar el modelo en `src/models/polyp_resnet18.pth`.

## Verification Plan

### Automated / Coded Tests

Crearemos un script pequeño dentro del propio `modelo_busca_polipos.py` bajo el bloque `if __name__ == '__main__':` que:

1. Instancie el dataset y verifique que un lote (batch) de imágenes se carga correctamente (tensor de tamaño `[batch_size, 3, 224, 224]`).
2. Instancie el modelo ResNet18 y pase un tensor dummy (ej. `torch.randn(1, 3, 224, 224)`) para asegurar que la predicción funciona sin errores de dimensiones.

### Manual Verification

Ejecutar `python src/models/modelo_busca_polipos.py` y observar si se imprimen las dimensiones del batch y del tensor de salida correctamente sin fallos de compilación ni de PyTorch. Posteriormente se verificará con una pequeña cantidad de epochs que la loss disminuye.
