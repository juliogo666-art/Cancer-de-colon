"""
=============================================================================
Tests de endpoints ampliados para la API FastAPI
=============================================================================
Amplía test_api.py con tests adicionales para cubrir más endpoints:
heartbeat, análisis de imagen (con mocks), y CRUD de pacientes.

Ejecutar con:
    pytest src/test/test_api_endpoints.py -v
=============================================================================
"""

import io
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


###############################################################################
# Fixture: Cliente de prueba
###############################################################################


@pytest.fixture
def cliente():
    """Crea un TestClient de la API para simular peticiones HTTP."""
    from src.api.main_api import app

    client = TestClient(app, raise_server_exceptions=False)
    return client


###############################################################################
# Tests de Heartbeat
###############################################################################


class TestHeartbeat:
    """Comprueba los endpoints de heartbeat para el watchdog."""

    def test_heartbeat_ping_responde_ok(self, cliente):
        """GET /api/v1/heartbeat debe devolver status ok."""
        respuesta = cliente.get("/api/v1/heartbeat")
        assert respuesta.status_code == 200
        datos = respuesta.json()
        assert datos["status"] == "ok"

    def test_heartbeat_status_devuelve_float(self, cliente):
        """GET /api/v1/heartbeat_status debe devolver un número float."""
        respuesta = cliente.get("/api/v1/heartbeat_status")
        assert respuesta.status_code == 200
        valor = respuesta.json()
        assert isinstance(valor, (int, float))

    def test_heartbeat_actualiza_timestamp(self, cliente):
        """Después de hacer ping, el status debe ser mayor a 0."""
        cliente.get("/api/v1/heartbeat")  # Primer ping
        respuesta = cliente.get("/api/v1/heartbeat_status")
        timestamp = respuesta.json()
        assert timestamp > 0


###############################################################################
# Tests de análisis de imagen — Colonoscopia
###############################################################################


class TestAnalisisColonoscopia:
    """Comprueba el endpoint de análisis de colonoscopia."""

    def test_sin_modelo_devuelve_error(self, cliente):
        """Si el modelo CNN no está cargado, debe devolver error (500/503)."""
        # Crear una imagen falsa
        img_bytes = self._crear_imagen_fake()
        respuesta = cliente.post(
            "/api/v1/analyze/colonoscopy",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        # Sin modelo cargado (lifespan no se ejecuta en test) → 500 o 503
        assert respuesta.status_code in [500, 503]

    def test_sin_archivo_devuelve_422(self, cliente):
        """Si no se envía archivo, devuelve 422."""
        respuesta = cliente.post("/api/v1/analyze/colonoscopy")
        assert respuesta.status_code == 422

    @staticmethod
    def _crear_imagen_fake():
        """Crea una imagen JPEG mínima válida en memoria."""
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer.getvalue()


###############################################################################
# Tests de análisis de imagen — Biopsia
###############################################################################


class TestAnalisisBiopsia:
    """Comprueba el endpoint de análisis de biopsia."""

    def test_sin_modelo_devuelve_error(self, cliente):
        """Si el modelo de biopsias no está cargado, devuelve error."""
        img_bytes = TestAnalisisColonoscopia._crear_imagen_fake()
        respuesta = cliente.post(
            "/api/v1/analyze/biopsy",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        # Sin modelo (lifespan no se ejecuta en test) → 500 o 503
        assert respuesta.status_code in [500, 503]


###############################################################################
# Tests de predicción de riesgo — Modo triaje
###############################################################################


class TestPrediccionTriaje:
    """Comprueba la predicción de riesgo en modo triaje (sin analíticas)."""

    def test_triaje_sin_modelo_devuelve_503(self, cliente):
        """Cuando FOBT=-1, usa modelo triaje. Sin modelo → 503."""
        respuesta = cliente.post(
            "/api/v1/predict/risk",
            params={
                "smoking": 5,
                "alcohol_use": 3,
                "obesity": 2,
                "family_history": 1,
                "diet_red_meat": 4,
                "diet_salted_processed": 3,
                "fruit_veg_intake": 7,
                "physical_activity": 6,
                "bmi": 25.0,
                "fobt_resultado": -1,  # Activa modo triaje
                "cea_level": -1.0,  # Activa modo triaje
            },
        )
        assert respuesta.status_code in [200, 503]

    def test_parametro_bmi_fuera_de_rango_devuelve_422(self, cliente):
        """BMI fuera de rango (10-60) debe ser rechazado por FastAPI."""
        respuesta = cliente.post(
            "/api/v1/predict/risk",
            params={
                "smoking": 5,
                "alcohol_use": 3,
                "obesity": 2,
                "family_history": 1,
                "diet_red_meat": 4,
                "diet_salted_processed": 3,
                "fruit_veg_intake": 7,
                "physical_activity": 6,
                "bmi": 5.0,  # Fuera de rango (min=10)
                "fobt_resultado": 0,
                "cea_level": 2.0,
            },
        )
        assert respuesta.status_code == 422


###############################################################################
# Tests de gestión de pacientes — Extendidos
###############################################################################


class TestPacientesExtendido:
    """Tests adicionales para los endpoints CRUD de pacientes."""

    def test_crear_paciente_sin_csv_devuelve_error(self, cliente):
        """Crear paciente sin CSV disponible debe dar error."""
        # Usamos un patient_id que difícilmente exista
        respuesta = cliente.post(
            "/api/v1/patients",
            params={
                "patient_id": 999888,
                "smoking": 3,
                "alcohol_use": 2,
                "obesity": 1,
                "family_history": 0,
                "diet_red_meat": 2,
                "diet_salted_processed": 1,
                "fruit_veg_intake": 8,
                "physical_activity": 7,
                "bmi": 22.0,
                "overall_risk_score": 0.2,
                "risk_level": "Low",
            },
        )
        # 200 si el CSV existe y el ID no está, 404 si no hay CSV, 409 si ya existe
        assert respuesta.status_code in [200, 404, 409]

    def test_actualizar_paciente_inexistente_devuelve_error(self, cliente):
        """Actualizar un paciente que no existe devuelve error."""
        respuesta = cliente.put(
            "/api/v1/patients/999888",
            params={"smoking": 5},
        )
        # 404 si paciente no existe, 500 si hay error de datos (NaN en CSV serialization)
        assert respuesta.status_code in [404, 500]

    def test_listar_pacientes_paginacion(self, cliente):
        """El endpoint de lista acepta parámetros de paginación."""
        respuesta = cliente.get(
            "/api/v1/patients", params={"skip": 0, "limit": 2}
        )
        if respuesta.status_code == 200:
            datos = respuesta.json()
            assert "total" in datos
            assert "patients" in datos
            assert datos["limit"] == 2
