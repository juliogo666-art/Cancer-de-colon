# artifacts/

Directorio centralizado para todos los modelos de IA entrenados del proyecto.

## Estructura

```
artifacts/
├── weights/       # Modelos clásicos serializados (.pkl, .joblib, .keras)
│                  
│
├── exports/       # Modelos exportados a ONNX para inferencia sin PyTorch
│                  
│
├── checkpoints/   # Pesos de PyTorch (.pth) para reentrenamiento
│                  
│
└── mappings/      # Mapeos de IDs internos <-> IDs reales (.json, .pkl)
                   
```
