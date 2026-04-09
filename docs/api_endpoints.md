# API Endpoints — Cancer de Colon

## Arrancar la API

```bash
uvicorn src.api.main_api:app --reload --port 8000
```

La documentación interactiva Swagger se genera automáticamente en:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Health Check

### `GET /`

Verifica que el servidor está activo y qué modelos están cargados.

**Respuesta (200)**:
```json
{
  "status": "online",
  "models": {
    "ml_clinico": true,
    "cnn_colonoscopia": true,
    "resnet_biopsia": false
  }
}
```

---

## Predicción de Riesgo (ML)

### `POST /api/v1/predict/risk`

Predice el nivel de riesgo de cáncer de colon a partir de factores clínicos.

**Parámetros (query)**:

| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `smoking` | float | 0-10 | Nivel de tabaquismo |
| `alcohol_use` | float | 0-10 | Consumo de alcohol |
| `obesity` | float | 0-10 | Nivel de obesidad |
| `family_history` | int | 0-1 | Antecedentes familiares |
| `diet_red_meat` | float | 0-10 | Consumo de carne roja |
| `diet_salted_processed` | float | 0-10 | Consumo de procesados |
| `fruit_veg_intake` | float | 0-10 | Consumo de fruta/verdura |
| `physical_activity` | float | 0-10 | Actividad física |
| `bmi` | float | 10-60 | Índice de masa corporal |
| `fobt_resultado` | int | 0-1 | Test de sangre oculta (FOBT) |
| `cea_level` | float | ≥0 | Marcador tumoral CEA |

**Respuesta (200)**:
```json
{
  "risk_level": "High",
  "risk_score": 0.847,
  "probabilities": {
    "Low": 0.05,
    "Medium": 0.10,
    "High": 0.85
  },
  "features_used": {
    "Smoking": 8.0,
    "BMI": 32.5
  }
}
```

**Errores**:
- `503`: Modelo ML no disponible
- `422`: Parámetros inválidos (fuera de rango)

---

## Análisis de Imagen

### `POST /api/v1/analyze/colonoscopy`

Analiza una imagen de colonoscopia para detectar pólipos.

**Parámetros**: `file` (multipart/form-data) — Imagen JPG/PNG

**Respuesta (200)**:
```json
{
  "diagnosis": "POLIPO DETECTADO",
  "is_polyp": true,
  "confidence": 0.92,
  "raw_prediction": 0.08,
  "recommendation": "Se recomienda revisión inmediata por especialista.",
  "gradcam_base64": "iVBORw0KGgo..."
}
```

### `POST /api/v1/analyze/biopsy`

Analiza una imagen de biopsia para clasificar tejido benigno vs maligno.

**Parámetros**: `file` (multipart/form-data) — Imagen JPG/PNG

**Respuesta (200)**:
```json
{
  "diagnosis": "MALIGNO (ADENOCARCINOMA)",
  "is_benign": false,
  "confidence": 0.88,
  "raw_probability": 0.12,
  "recommendation": "Sospecha de malignidad. Se recomienda estudio histopatológico completo.",
  "gradcam_base64": "iVBORw0KGgo..."
}
```

---

## Gestión de Pacientes (CRUD)

### `GET /api/v1/patients`

Lista pacientes con paginación.

| Parámetro | Default | Rango | Descripción |
|-----------|---------|-------|-------------|
| `skip` | 0 | ≥0 | Registros a saltar |
| `limit` | 50 | 1-500 | Máximo de registros |

### `GET /api/v1/patients/{patient_id}`

Busca un paciente específico por su ID.

### `POST /api/v1/patients`

Crea un nuevo paciente (todos los campos por query params).

### `PUT /api/v1/patients/{patient_id}`

Actualiza campos de un paciente existente (solo los que se pasan).
