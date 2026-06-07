"""
main_api.py — Backend FastAPI del proyecto ColonAI.

Endpoints organizados en 3 grupos:
    1. /api/v1/predict/    → Predicción de riesgo clínico (ML) + explicabilidad SHAP
    2. /api/v1/analyze/    → Análisis de imagen (colonoscopia y biopsias)
    3. /api/v1/patients/   → CRUD de pacientes sobre CSV

Arrancar con:
    uvicorn src.api.main_api:app --reload --port 8000
"""

import io
import os
import base64
import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision import transforms

from src.api.dependencies import (
    CSV_MASTER_PATH,
    ML_FEATURE_NAMES,
    RISK_LEVEL_MAP,
    lifespan,
)
from src.tracking.prediction_logger import PredictionLogger
from src.utils.gradcam_utils import generate_gradcam_pytorch

###############################################################################
# Inicialización
###############################################################################

app = FastAPI(
    title="ColonAI API",
    description="API de predicción de riesgo de cáncer de colon y análisis de imagen.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — Permite que el frontend Streamlit se comunique con la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###############################################################################
# Utilidades internas
###############################################################################

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB — Generoso para imágenes médicas de alta resolución



def _image_to_base64(img_array: np.ndarray) -> str:
    """Convierte un array numpy (imagen RGB) a string base64 para JSON."""
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _read_upload_as_array(file_bytes: bytes) -> np.ndarray:
    """Lee bytes de un archivo subido y devuelve un array numpy RGB."""
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


# Instancia central del logger para las predicciones
logger = PredictionLogger()


###############################################################################
# Verificación de estado y Heartbeat
###############################################################################

last_heartbeat_time = 0.0

@app.get("/api/v1/heartbeat")
async def heartbeat_ping():
    """Recibe un ping desde el JS del cliente."""
    global last_heartbeat_time
    last_heartbeat_time = time.time()
    return {"status": "ok"}

@app.get("/api/v1/heartbeat_status")
async def heartbeat_status():
    """Retorna el último timestamp del cliente."""
    return last_heartbeat_time


@app.get("/", tags=["Health"])
async def health_check(request: Request):
    """Verifica el estado del servidor y los modelos cargados."""
    return {
        "status": "online",
        "models": {
            "ml_clinico": getattr(request.app.state, "modelo_ml", None) is not None,
            "cnn_colonoscopia": getattr(request.app.state, "modelo_cnn", None) is not None,
            "densenet_biopsia": getattr(request.app.state, "modelo_biopsia", None) is not None,
        },
    }

###############################################################################
# Predicción de riesgo clínico (ML)
###############################################################################


@app.post("/api/v1/predict/risk", tags=["Predicción ML"])
async def predict_risk(
    request: Request,
    patient_id: Optional[int] = Query(None, description="ID del paciente (opcional)"),
    smoking: float = Query(..., ge=0, le=10, description="Nivel de tabaquismo (0-10)"),
    alcohol_use: float = Query(
        ..., ge=0, le=10, description="Consumo de alcohol (0-10)"
    ),
    obesity: float = Query(..., ge=0, le=10, description="Nivel de obesidad (0-10)"),
    family_history: int = Query(
        ..., ge=0, le=1, description="Historial familiar (0=No, 1=Sí)"
    ),
    diet_red_meat: float = Query(
        ..., ge=0, le=10, description="Consumo de carne roja (0-10)"
    ),
    diet_salted_processed: float = Query(
        ..., ge=0, le=10, description="Consumo de sal/procesados (0-10)"
    ),
    fruit_veg_intake: float = Query(
        ..., ge=0, le=10, description="Consumo de fruta/verdura (0-10)"
    ),
    physical_activity: float = Query(
        ..., ge=0, le=10, description="Actividad física (0-10)"
    ),
    bmi: float = Query(..., ge=10, le=60, description="Índice de masa corporal"),
    fobt_resultado: int = Query(
        ..., ge=-1, le=1, description="FOBT (-1=Desc, 0=Neg, 1=Pos)"
    ),
    cea_level: float = Query(
        ..., ge=-1.0, description="Nivel CEA en ng/mL (-1.0=Desc)"
    ),
):
    """
    Predice el nivel de riesgo de cáncer de colon a partir de factores clínicos.

    Devuelve la probabilidad de cada clase (Low, Medium, High) y la clase predicha.
    """
    if fobt_resultado == -1 or cea_level == -1.0:
        # TRIAGE MODEL (No Analytics)
        modelo = getattr(request.app.state, "modelo_ml_triage", None)
        if modelo is None:
            raise HTTPException(
                status_code=503, detail="Modelo ML de triaje no disponible."
            )

        features = np.array(
            [
                [
                    smoking,
                    alcohol_use,
                    obesity,
                    family_history,
                    diet_red_meat,
                    diet_salted_processed,
                    fruit_veg_intake,
                    physical_activity,
                    bmi,
                ]
            ]
        )
        features_dict = dict(zip(ML_FEATURE_NAMES[:9], features[0].tolist()))

    else:
        # FULL CLINICAL MODEL
        modelo = getattr(request.app.state, "modelo_ml", None)
        if modelo is None:
            raise HTTPException(
                status_code=503, detail="Modelo ML clínico no disponible."
            )

        features = np.array(
            [
                [
                    smoking,
                    alcohol_use,
                    obesity,
                    family_history,
                    diet_red_meat,
                    diet_salted_processed,
                    fruit_veg_intake,
                    physical_activity,
                    bmi,
                    fobt_resultado,
                    cea_level,
                ]
            ]
        )
        features_dict = dict(zip(ML_FEATURE_NAMES, features[0].tolist()))

    try:
        probabilidades = modelo.predict_proba(features)[0]
        clase_predicha = int(np.argmax(probabilidades))

        # Calcular un score de riesgo ponderado (0.0 a 1.0)
        if len(probabilidades) == 3:
            riesgo_score = (
                0.005 + (probabilidades[1] * 0.45) + (probabilidades[2] * 1.0)
            )
        else:
            riesgo_score = float(probabilidades[1]) if len(probabilidades) == 2 else 0.0
        riesgo_score = min(riesgo_score, 1.0)

        # Registro de auditoría
        risk_lvl = RISK_LEVEL_MAP.get(clase_predicha, "Unknown")
        max_prob = float(np.max(probabilidades))

        # Crear recomendación (clave de traducción, el frontend la traduce)
        recomendacion = ""
        if fobt_resultado == -1 or cea_level == -1.0:
            if risk_lvl == "Low":
                recomendacion = "rec_low_risk"
            else:
                recomendacion = "rec_need_analytics"
        else:
            if risk_lvl in ["Medium", "High"]:
                if fobt_resultado == 0 and cea_level < 5.0:
                    recomendacion = "rec_low_risk"
                else:
                    recomendacion = "rec_high_risk"
            else:
                recomendacion = "rec_low_risk"

        logger.log_risk_prediction(
            patient_id=patient_id,
            risk_level=risk_lvl,
            risk_score=riesgo_score,
            confidence=max_prob,
            features=features_dict,
        )

        return {
            "risk_level": risk_lvl,
            "risk_score": round(riesgo_score, 4),
            "recommendation": recomendacion,
            "probabilities": {
                "Low": round(float(probabilidades[0]), 4),
                "Medium": round(float(probabilidades[1]), 4)
                if len(probabilidades) > 1
                else 0.0,
                "High": round(float(probabilidades[2]), 4)
                if len(probabilidades) > 2
                else 0.0,
            },
            "features_used": features_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


###############################################################################
# Análisis de imagen (DL)
###############################################################################

@app.post("/api/v1/analyze/colonoscopy", tags=["Análisis de Imagen"])
async def analyze_colonoscopy(
    request: Request,
    file: UploadFile = File(...),
    patient_id: Optional[int] = Query(None),
):
    modelo = request.app.state.modelo_cnn
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo CNN no cargado.")

    try:
        # 1. Cargar imagen
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="El archivo excede el límite de 20MB.",
            )
        img_array = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 2. Inferencia (Predicción)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img_pil).unsqueeze(0)

        # Usamos no_grad solo para la probabilidad rápida
        with torch.no_grad():
            output = modelo(input_tensor)
            prob = output.item()

        # Interpretación (0=Polipo, 1=Sano)
        es_polipo = prob < 0.5 
        confianza = (1.0 - prob) if es_polipo else prob

        # 3. Generar Grad-CAM (Requiere gradientes activos)
        heatmap_b64 = None
        try:
            # Intenta primero con esta capa que es la última convolucional real de MobileNetV2
            if hasattr(modelo, 'features'):
                target_layer = modelo.features[-1]
            else:
                # Fallback genérico: busca la última capa convolucional
                target_layers = [m for m in modelo.modules() if isinstance(m, torch.nn.Conv2d)]
                target_layer = target_layers[-1]

            gradcam_img = generate_gradcam_pytorch(modelo, img_pil, target_layer)
            
            _, buffer = cv2.imencode('.jpg', gradcam_img)
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[ERROR Grad-CAM]: {e}")
            heatmap_b64 = None

        diagnosis_key = "diag_polyp_detected" if es_polipo else "diag_healthy_tissue"

        return {
            "diagnosis": diagnosis_key,
            "is_polyp": es_polipo,
            "confidence": round(float(confianza), 4),
            "raw_prediction": round(float(prob), 4),
            "gradcam_base64": heatmap_b64,
            "recommendation": "rec_biopsy" if es_polipo else "rec_routine_followup"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el análisis: {str(e)}")


@app.post("/api/v1/analyze/biopsy", tags=["Análisis de Imagen"])
async def analyze_biopsy(
    request: Request,
    file: UploadFile = File(...),
    patient_id: Optional[int] = Query(None, description="ID del paciente (opcional)"),
):
    """
    Analiza una imagen de biopsia para clasificar tejido benigno vs maligno.

    Devuelve la probabilidad, el diagnóstico y el mapa de calor Grad-CAM
    codificado en base64.
    """
    modelo = request.app.state.modelo_biopsia
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo de biopsias no disponible.")

    try:
        # 1. Leer imagen
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="El archivo excede el límite de 5MB.")
            
        img_array = _read_upload_as_array(contents)
        img_pil = Image.fromarray(img_array)

        # 2. Preprocesar para PyTorch (224x224, normalización ImageNet)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = transform(img_pil).unsqueeze(0)

        # 3. Inferencia
        with torch.no_grad():
            output = modelo(input_tensor)
            prob = torch.sigmoid(output).item()

        es_benigno = prob >= 0.5
        confianza = prob if es_benigno else (1.0 - prob)

        # 4. Grad-CAM
        heatmap_b64 = None
        try:
            from src.utils.gradcam_utils import generate_gradcam

            target_layer = modelo.model.features.denseblock4
            heatmap_img, _ = generate_gradcam(modelo, img_pil, target_layer)
            heatmap_b64 = _image_to_base64(heatmap_img)
        except Exception:
            pass

        diagnosis_key = (
            "diag_benign" if es_benigno else "diag_malignant"
        )

        # Registro de auditoría
        logger.log_image_prediction(
            analysis_type="biopsy",
            diagnosis=diagnosis_key,
            confidence=confianza,
            patient_id=patient_id,
            image_name=file.filename,
        )

        return {
            "diagnosis": diagnosis_key,
            "is_benign": es_benigno,
            "confidence": round(confianza, 4),
            "raw_probability": round(prob, 4),
            "recommendation": (
                "rec_benign_tissue"
                if es_benigno
                else "rec_malignant_tissue"
            ),
            "gradcam_base64": heatmap_b64,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analizando biopsia: {str(e)}"
        )


###############################################################################
# Gestión de pacientes (CRUD sobre CSV)
###############################################################################


@app.get("/api/v1/patients", tags=["Pacientes"])
async def list_patients(
    skip: int = Query(0, ge=0, description="Registros a saltar"),
    limit: int = Query(50, ge=1, le=500, description="Máximo de registros"),
):
    """Lista los pacientes del dataset con paginación."""
    if not os.path.exists(CSV_MASTER_PATH):
        raise HTTPException(
            status_code=404, detail="Archivo de pacientes no encontrado."
        )

    try:
        df = pd.read_csv(CSV_MASTER_PATH)
        total = len(df)
        page = df.iloc[skip : skip + limit]
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "patients": page.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error leyendo pacientes: {str(e)}"
        )


@app.get("/api/v1/patients/{patient_id}", tags=["Pacientes"])
async def get_patient(patient_id: int):
    """Obtiene los datos de un paciente específico por su ID."""
    if not os.path.exists(CSV_MASTER_PATH):
        raise HTTPException(
            status_code=404, detail="Archivo de pacientes no encontrado."
        )

    try:
        df = pd.read_csv(CSV_MASTER_PATH)
        patient = df[df["Patient_ID"] == patient_id]

        if patient.empty:
            raise HTTPException(
                status_code=404, detail=f"Paciente {patient_id} no encontrado."
            )

        return patient.iloc[0].to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error buscando paciente: {str(e)}"
        )


@app.post("/api/v1/patients", tags=["Pacientes"])
async def create_patient(
    patient_id: int = Query(..., description="ID único del paciente"),
    smoking: int = Query(0, ge=0, le=10),
    alcohol_use: int = Query(0, ge=0, le=10),
    obesity: int = Query(0, ge=0, le=10),
    family_history: int = Query(0, ge=0, le=1),
    diet_red_meat: int = Query(0, ge=0, le=10),
    diet_salted_processed: int = Query(0, ge=0, le=10),
    fruit_veg_intake: int = Query(0, ge=0, le=10),
    physical_activity: int = Query(0, ge=0, le=10),
    bmi: float = Query(25.0, ge=10, le=60),
    overall_risk_score: float = Query(0.0, ge=0, le=1),
    risk_level: str = Query("Low", description="Low, Medium o High"),
):
    """Crea un nuevo registro de paciente en el CSV."""
    if not os.path.exists(CSV_MASTER_PATH):
        raise HTTPException(
            status_code=404, detail="Archivo de pacientes no encontrado."
        )

    try:
        df = pd.read_csv(CSV_MASTER_PATH)

        if (df["Patient_ID"] == patient_id).any():
            raise HTTPException(
                status_code=409, detail=f"El paciente {patient_id} ya existe."
            )

        risk_level_n = {"Low": 0, "Medium": 1, "High": 2}.get(risk_level, 0)

        new_row = {
            "Patient_ID": patient_id,
            "Smoking": smoking,
            "Alcohol_Use": alcohol_use,
            "Obesity": obesity,
            "Family_History": family_history,
            "Diet_Red_Meat": diet_red_meat,
            "Diet_Salted_Processed": diet_salted_processed,
            "Fruit_Veg_Intake": fruit_veg_intake,
            "Physical_Activity": physical_activity,
            "BMI": bmi,
            "Overall_Risk_Score": overall_risk_score,
            "Risk_Level": risk_level,
            "Risk_Level_n": risk_level_n,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_MASTER_PATH, index=False)

        return {
            "message": f"Paciente {patient_id} creado correctamente.",
            "patient": new_row,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando paciente: {str(e)}")


@app.put("/api/v1/patients/{patient_id}", tags=["Pacientes"])
async def update_patient(
    patient_id: int,
    smoking: Optional[int] = Query(None, ge=0, le=10),
    alcohol_use: Optional[int] = Query(None, ge=0, le=10),
    obesity: Optional[int] = Query(None, ge=0, le=10),
    family_history: Optional[int] = Query(None, ge=0, le=1),
    diet_red_meat: Optional[int] = Query(None, ge=0, le=10),
    diet_salted_processed: Optional[int] = Query(None, ge=0, le=10),
    fruit_veg_intake: Optional[int] = Query(None, ge=0, le=10),
    physical_activity: Optional[int] = Query(None, ge=0, le=10),
    bmi: Optional[float] = Query(None, ge=10, le=60),
    overall_risk_score: Optional[float] = Query(None, ge=0, le=1),
    risk_level: Optional[str] = Query(None, description="Low, Medium o High"),
):
    """Actualiza los datos de un paciente existente."""
    if not os.path.exists(CSV_MASTER_PATH):
        raise HTTPException(
            status_code=404, detail="Archivo de pacientes no encontrado."
        )

    try:
        df = pd.read_csv(CSV_MASTER_PATH)
        mask = df["Patient_ID"] == patient_id

        if not mask.any():
            raise HTTPException(
                status_code=404, detail=f"Paciente {patient_id} no encontrado."
            )

        idx = df[mask].index[0]

        # Solo actualizamos los campos que se proporcionan (no son None)
        updates = {
            "Smoking": smoking,
            "Alcohol_Use": alcohol_use,
            "Obesity": obesity,
            "Family_History": family_history,
            "Diet_Red_Meat": diet_red_meat,
            "Diet_Salted_Processed": diet_salted_processed,
            "Fruit_Veg_Intake": fruit_veg_intake,
            "Physical_Activity": physical_activity,
            "BMI": bmi,
            "Overall_Risk_Score": overall_risk_score,
            "Risk_Level": risk_level,
        }

        for col, val in updates.items():
            if val is not None and col in df.columns:
                df.at[idx, col] = val

        # Recalcular Risk_Level_n si se cambió Risk_Level
        if risk_level is not None and "Risk_Level_n" in df.columns:
            df.at[idx, "Risk_Level_n"] = {"Low": 0, "Medium": 1, "High": 2}.get(
                risk_level, 0
            )

        df.to_csv(CSV_MASTER_PATH, index=False)

        return {
            "message": f"Paciente {patient_id} actualizado correctamente.",
            "patient": df.loc[idx].to_dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error actualizando paciente: {str(e)}"
        )


###############################################################################
# Punto de entrada directo
###############################################################################

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main_api:app", host="0.0.0.0", port=8000, reload=True)
