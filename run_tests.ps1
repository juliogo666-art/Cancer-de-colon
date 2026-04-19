# Ejecutar todos los tests y generar informe de cobertura en HTML
Write-Host "Ejecutando tests de pytest y generando coverage..." -ForegroundColor Cyan

# Añade la variable de entorno de Python
$env:PYTHONPATH = ".;src;"

# Ejecuta uv run pytest (o pytest directo si estas en el entorno)
uv run pytest --cov=src --cov-report=html

if ($LASTEXITCODE -eq 0) {
    Write-Host "Tests exitosos! Puedes revisar el reporte en ./htmlcov/index.html" -ForegroundColor Green
    Start-Process "./htmlcov/index.html"
} else {
    Write-Host "Atencion: Ciertos tests han fallado." -ForegroundColor Yellow
}
