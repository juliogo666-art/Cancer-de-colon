"""
Orquestador del proyecto.
Desde aquí puedes lanzar cualquier parte del sistema usando comandos por terminal.

COMANDOS DISPONIBLES:
    python main.py start                      # Modo facil: Arranca API y Frontend a la vez
    python main.py api                        # Arranca solo la API FastAPI
    python main.py frontend                   # Arranca solo Streamlit (la app visual)
    python main.py eda                        # Lanzar la app de análisis exploratorio
    python main.py train-ml                   # Entrenar el modelo ML de riesgo
    python main.py test                       # Ejecutar todos los tests
    python main.py generate-data              # Generar datos sintéticos
=============================================================================
"""

import argparse
import os
import sys


def ejecutar_api():
    """
    Arranca el servidor de la API FastAPI en el puerto 8000.
    La API expone los modelos de IA como endpoints HTTP.
    Acceder a la documentación en: http://localhost:8000/docs
    """
    import uvicorn

    print("=" * 60)
    print("  Arrancando ColonAI API en http://localhost:8000")
    print("  Documentación en: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("src.api.main_api:app", host="0.0.0.0", port=8000, reload=True)


def ejecutar_frontend():
    """
    Arranca la aplicación Streamlit unificada (el frontend visual).
    Se abrirá automáticamente en el navegador.
    """
    os.system("streamlit run src/frontend/app.py")


def ejecutar_todo():
    """
    Arranca la API y el Frontend a la vez en una misma terminal.
    """
    import subprocess
    import time

    print(" Levantando el backend (API) en segundo plano...")
    # Usamos sys.executable para asegurarnos de usar el mismo intérprete de Python
    api_proc = subprocess.Popen([sys.executable, "main.py", "api"])

    print(
        " Esperando 5 segundos para que los modelos de la API aseguren su carga inicial..."
    )
    time.sleep(5)

    print(" Levantando el frontend Visual de Galeno...")
    frontend_proc = subprocess.Popen(["streamlit", "run", "src/frontend/app.py"])

    try:
        import urllib.request
        # Ventana de gracia: Streamlit necesita tiempo para cargar completamente
        # y renderizar el componente JS del heartbeat. Sin esta espera, el watchdog
        # detecta un heartbeat inicial y luego lo ve "muerto" antes de que el
        # frontend termine de arrancar.
        GRACIA_INICIO = 15  # segundos de gracia antes de activar el watchdog
        TIMEOUT_HEARTBEAT = 10  # segundos sin heartbeat para considerar navegador cerrado
        tiempo_arranque = time.time()
        
        while True:
            # Si alguno de los dos cierra de forma natural, salimos
            if api_proc.poll() is not None or frontend_proc.poll() is not None:
                break
            
            # No comprobar heartbeat hasta que pase el periodo de gracia
            if time.time() - tiempo_arranque < GRACIA_INICIO:
                time.sleep(1)
                continue
            
            # Watchdog - Comprobar si el navegador se ha cerrado (heartbeat via HTTP)
            try:
                req = urllib.request.Request("http://localhost:8000/api/v1/heartbeat_status")
                with urllib.request.urlopen(req, timeout=2.0) as response:
                    last_beat = float(response.read().decode("utf-8"))
                    
                # Si last_beat es mayor a 0 (alguien se conectó) y han pasado +10 seg sin ping, asumimos cerrado
                if last_beat > 0.0 and time.time() - last_beat > TIMEOUT_HEARTBEAT:
                    print("\n[Galeno] Navegador cerrado. Apagando el servidor internamente...")
                    break
            except Exception:
                pass # Si no contesta temporalmente, ignoramos
                
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n Deteniendo los servidores (API y Frontend)...")
        api_proc.terminate()
        frontend_proc.terminate()
        print("Adios!")
        # Forzar el cierre de la terminal de Windows usando el título de la ventana
        if os.name == "nt":
            os.system('taskkill /F /FI "WINDOWTITLE eq Lanzador Galeno*" /T > nul 2>&1')


def ejecutar_eda():
    """Lanza la aplicación de análisis exploratorio de datos (EDA)."""
    from src.scripts.eda import eda

    directorio_base = os.path.dirname(__file__)
    eda(directorio_base)


def ejecutar_entrenamiento_ml():
    """
    Entrena el modelo ML de riesgo clínico usando el TrainingPipeline.
    Carga el CSV, divide en train/test, entrena y guarda el modelo + métricas.
    """
    from lightgbm import LGBMClassifier
    from src.pipelines.training_pipeline import TrainingPipeline
    from src.config.settings import settings

    print("=" * 60)
    print("  Entrenando modelo ML con TrainingPipeline...")
    print("=" * 60)

    # Crear el pipeline con la ruta centralizada
    pipeline = TrainingPipeline(csv_path=settings.CSV_RISK_PATH)
    pipeline.load_and_prepare()

    # Configurar el modelo LightGBM con los hiperparámetros de ml_v3.py
    modelo = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=5,
        random_state=42,
        verbose=-1,  # Sin mensajes de entrenamiento de LightGBM
    )

    # Entrenar y evaluar
    resultados = pipeline.train_and_evaluate(modelo, "LightGBM")

    # Guardar el modelo
    pipeline.save_model(modelo, "lgbm_clinico")

    print("\n Modelo entrenado y guardado correctamente.")


def ejecutar_tests():
    """Ejecuta todos los tests del proyecto con pytest."""
    os.system("pytest src/test/ -v --tb=short")


def ejecutar_generacion_datos():
    """Genera los datos sintéticos de pacientes."""
    from src.scripts.sintetiza_historiales import generar_datos_sinteticos

    generar_datos_sinteticos()


def main():
    """
    Función principal que analiza qué comando has escrito en la terminal
    y ejecuta la función correspondiente.
    """
    parser = argparse.ArgumentParser(
        description="ColonAI — Sistema de apoyo al diagnóstico de cáncer de colon"
    )

    # Definimos los comandos disponibles
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    subparsers.add_parser("start", help="Levanta la API y el Frontend a la vez")
    subparsers.add_parser("api", help="Arrancar la API FastAPI (puerto 8000)")
    subparsers.add_parser("frontend", help="Arrancar el frontend Streamlit")
    subparsers.add_parser("eda", help="Lanzar el análisis exploratorio de datos")
    subparsers.add_parser("train-ml", help="Entrenar el modelo ML de riesgo clínico")
    subparsers.add_parser("test", help="Ejecutar todos los tests con pytest")
    subparsers.add_parser("generate-data", help="Generar datos sintéticos de pacientes")

    argumentos = parser.parse_args()

    # Mapa de comandos → funciones
    comandos_disponibles = {
        "start": ejecutar_todo,
        "api": ejecutar_api,
        "frontend": ejecutar_frontend,
        "eda": ejecutar_eda,
        "train-ml": ejecutar_entrenamiento_ml,
        "test": ejecutar_tests,
        "generate-data": ejecutar_generacion_datos,
    }

    if argumentos.command is None:
        # Si no se especifica comando, mostramos la ayuda
        if "streamlit" in sys.modules:
            ejecutar_eda()
        else:
            parser.print_help()
            print("\n Comandos mas usados:")
            print(
                "   python main.py start       -> Arrancar todo de una vez (Recomendado)"
            )
            print("   python main.py api         -> Arrancar solo la API")
            print("   python main.py frontend    -> Arrancar solo Streamlit")
            print("   python main.py test        -> Ejecutar tests")
    elif argumentos.command in comandos_disponibles:
        comandos_disponibles[argumentos.command]()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
