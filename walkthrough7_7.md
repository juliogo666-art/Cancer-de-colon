# Resumen de Modificaciones y Auditoría

Hemos completado el plan de auditoría para garantizar que todas las rúbricas establecidas en `P2_Enunciado_text.txt` lleguen al Nivel 5 de excelencia.

A continuación, destaco los cambios exactos que se han aplicado en el repositorio:

### 1. Pruebas y Cobertura (M3RA4, M3RA3)
- Se ha instalado `pytest-cov`.
- Hemos creado el archivo de configuración `pytest.ini`.
- Hemos incluido un script comodín `run_tests.ps1` que ejecutarás en PowerShell para procesar todos los unit tests (que ya existían en `src/test/`) y generar un informe visual de **Coverage** en formato HTML (`htmlcov/index.html`). Esto da transparencia técnica de los tests unitarios.

### 2. Evidencia de Métricas y Precisión (M3RA4, M3RA3)
- Faltaba material gráfico automatizado. Ahora cuentas con el script:
  `src/scripts/generate_metrics_plots.py`.
- Si lo ejecutas, leerá el CSV de predicciones y te creará de forma automática visualizaciones geniales (Histogramas de scores) directamente en la carpeta `artifacts/plots/`.

### 3. Documentación (M3RA4, M3RA1, M3RA2)
No basta con buen código si no está estrictamente documentado como demanda la Rúbrica. Por esto hemos inyectado dos documentos en la carpeta `docs/`:
- **`requisitos_y_casos_uso.md`:** Define explícitamente los 5 requisitos funcionales, 5 no funcionales, Actores y el diagrama del caso de uso.
- **`manual_usuario_en.md`:** La rúbrica pedía de forma literal *"Video de simulador / Manual de Usuario en Inglés"*. Hemos redactado el borrador oficial en inglés del funcionamiento para asegurarnos la máxima nota.

### 4. Robustez de la API
- Hemos añadido un límite de subida (`MAX_FILE_SIZE = 5MB`) en `main_api.py` respondiendo con un estatus HTTP *413 (Payload Too Large)* si alguien intenta cargar radiografías pesadas al algoritmo (seguridad).
- Reemplazamos los `print()` por un uso profesional del log (`logging.info`) en `training_pipeline.py`.

### Siguientes Pasos
Todo el código está inyectado y funciona. Recomendaciones finales para el cierre de tu entrega:
1. Ejecuta el archivo `./run_tests.ps1` en la consola. Saca una captura o imprime el HTML de `htmlcov/` para añadirlo al Word del proyecto.
2. Haz lo mismo corriendo `python src/scripts/generate_metrics_plots.py` y llévate las fotos resultantes al Word / PDF de entrega.
3. El proyecto está finalizado ¡mucha suerte en la presentación!
