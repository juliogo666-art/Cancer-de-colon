"""
=============================================================================
Tests de integración para la API FastAPI
=============================================================================
Estos tests comprueban que los endpoints de la API responden correctamente
sin necesidad de tener los modelos reales cargados.

Usamos el TestClient de FastAPI para simular peticiones HTTP.

Ejecutar con:
    pytest src/test/test_api.py -v
=============================================================================
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


###############################################################################
# Crear un cliente de prueba de la API
###############################################################################


@pytest.fixture
def cliente_de_prueba():
    """
    Crea una instancia de TestClient para hacer peticiones a la API
    sin arrancar un servidor real.
    """
    from src.api.main_api import app

    client = TestClient(app)
    return client


###############################################################################
# Test de verificacion de que la API responde
###############################################################################


class TestHealthCheck:
    """Comprueba que el endpoint raíz (/) funciona."""

    def test_el_servidor_responde(self, cliente_de_prueba):
        """El endpoint / debe devolver status 200 y 'online'."""
        respuesta = cliente_de_prueba.get("/")
        assert respuesta.status_code == 200
        datos = respuesta.json()
        assert datos["status"] == "online"

    def test_devuelve_estado_de_modelos(self, cliente_de_prueba):
        """La respuesta debe incluir el estado de los 3 modelos."""
        respuesta = cliente_de_prueba.get("/")
        datos = respuesta.json()
        assert "models" in datos
        assert "ml_clinico" in datos["models"]
        assert "cnn_colonoscopia" in datos["models"]
        assert "densenet_biopsia" in datos["models"]


###############################################################################
# Test de prediccion de riesgo
###############################################################################


class TestPrediccionRiesgo:
    """Comprueba el endpoint de predicción ML."""

    def test_prediccion_sin_modelo_devuelve_503(self, cliente_de_prueba):
        """Si el modelo ML no está cargado, debe devolver error 503."""
        # El modelo probablemente no estará cargado en el entorno de test
        # porque no tiene el archivo .pkl disponible
        respuesta = cliente_de_prueba.post(
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
                "fobt_resultado": 0,
                "cea_level": 2.0,
            },
        )
        # Si el modelo no está cargado → 503, si lo está → 200
        assert respuesta.status_code in [200, 503]

    def test_parametros_invalidos_devuelve_422(self, cliente_de_prueba):
        """Si faltan parámetros requeridos, debe devolver error 422."""
        respuesta = cliente_de_prueba.post(
            "/api/v1/predict/risk",
            params={"smoking": 5},  # Faltan los demás parámetros
        )
        assert respuesta.status_code == 422


###############################################################################
# Test de gestión de pacientes
###############################################################################


class TestPacientes:
    """Comprueba los endpoints CRUD de pacientes."""

    def test_listar_pacientes(self, cliente_de_prueba):
        """El endpoint de listar pacientes debe responder correctamente."""
        respuesta = cliente_de_prueba.get(
            "/api/v1/patients", params={"skip": 0, "limit": 5}
        )
        # 200 si el CSV existe, 404 si no
        assert respuesta.status_code in [200, 404]

    def test_limite_maximo_de_paginacion(self, cliente_de_prueba):
        """El límite de paginación no puede superar 500."""
        respuesta = cliente_de_prueba.get("/api/v1/patients", params={"limit": 1000})
        assert respuesta.status_code == 422  # Pydantic rechaza limit > 500

    def test_buscar_paciente_inexistente(self, cliente_de_prueba):
        """Buscar un paciente con ID que no existe debe dar 404."""
        respuesta = cliente_de_prueba.get("/api/v1/patients/999999")
        # 404 porque el paciente no existe (o el CSV no existe)
        assert respuesta.status_code in [404]
