"""
Pydantic para validación de datos de pacientes.

Define la estructura de las peticiones y respuestas HTTP relacionadas
con la gestión de pacientes en la API.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class PatientBase(BaseModel):
    """Campos comunes de un paciente."""

    smoking: int = Field(0, ge=0, le=10, description="Nivel de tabaquismo (0-10)")
    alcohol_use: int = Field(0, ge=0, le=10, description="Consumo de alcohol (0-10)")
    obesity: int = Field(0, ge=0, le=10, description="Nivel de obesidad (0-10)")
    family_history: int = Field(
        0, ge=0, le=1, description="Historial familiar (0=No, 1=Sí)"
    )
    diet_red_meat: int = Field(
        0, ge=0, le=10, description="Consumo de carne roja (0-10)"
    )
    diet_salted_processed: int = Field(
        0, ge=0, le=10, description="Consumo de sal/procesados (0-10)"
    )
    fruit_veg_intake: int = Field(
        0, ge=0, le=10, description="Consumo de fruta/verdura (0-10)"
    )
    physical_activity: int = Field(
        0, ge=0, le=10, description="Actividad física (0-10)"
    )
    bmi: float = Field(25.0, ge=10, le=60, description="Índice de masa corporal")
    overall_risk_score: float = Field(
        0.0, ge=0, le=1, description="Score global de riesgo (0-1)"
    )
    risk_level: str = Field("Low", description="Nivel de riesgo: Low, Medium o High")


class PatientCreate(PatientBase):
    """Esquema para crear un nuevo paciente. Requiere Patient_ID."""

    patient_id: int = Field(..., description="Identificador único del paciente")


class PatientUpdate(BaseModel):
    """
    Esquema para actualizar un paciente existente.
    Todos los campos son opcionales: solo se actualizan los proporcionados.
    """

    smoking: Optional[int] = Field(None, ge=0, le=10)
    alcohol_use: Optional[int] = Field(None, ge=0, le=10)
    obesity: Optional[int] = Field(None, ge=0, le=10)
    family_history: Optional[int] = Field(None, ge=0, le=1)
    diet_red_meat: Optional[int] = Field(None, ge=0, le=10)
    diet_salted_processed: Optional[int] = Field(None, ge=0, le=10)
    fruit_veg_intake: Optional[int] = Field(None, ge=0, le=10)
    physical_activity: Optional[int] = Field(None, ge=0, le=10)
    bmi: Optional[float] = Field(None, ge=10, le=60)
    overall_risk_score: Optional[float] = Field(None, ge=0, le=1)
    risk_level: Optional[str] = Field(None, description="Low, Medium o High")


class PatientResponse(PatientBase):
    """Esquema de respuesta con todos los datos del paciente."""

    patient_id: int
    risk_level_n: int = Field(
        0, description="Nivel de riesgo numérico: 0=Low, 1=Medium, 2=High"
    )

    model_config = ConfigDict(from_attributes=True)


class PatientListResponse(BaseModel):
    """Respuesta paginada para listado de pacientes."""

    total: int
    skip: int
    limit: int
    patients: list[dict]
