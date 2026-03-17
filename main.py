"""
main.py — Punto de entrada principal del proyecto de Cáncer de Colón.

Uso:
    streamlit run main.py                    # Lanzar la app EDA (por defecto)
    python main.py train-polyps              # Entrenar modelo de detección de pólipos
    python main.py train-rf                  # Entrenar Random Forest con datos sintéticos
    python main.py generate-data             # Generar datos sintéticos de pacientes
"""

import argparse
import os
import sys


def run_eda():
    """Lanza la aplicación Streamlit de EDA."""
    from src.scripts.eda import eda
    base_path = os.path.dirname(__file__)
    eda(base_path)


def run_train_polyps():
    """Entrena el modelo ResNet18 de detección de pólipos."""
    from src.models.modelo_busca_polipos_Clas import train_model
    train_model()


def run_train_rf():
    """Entrena el modelo Random Forest con datos sintéticos."""
    import pandas as pd
    from test.test_ml_v0 import entrenar_random_forest

    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        base_path, 'src', 'data', 'raw', 'historial_pacientes',
        'historiales_sinteticos', 'pacientes_simulador_colon.csv'
    )

    if os.path.exists(csv_path):
        print(f"Cargando datos desde: {csv_path}")
        df = pd.read_csv(csv_path)
        entrenar_random_forest(df)
    else:
        print(f"No se encontró: {csv_path}")
        print("Genera primero los datos con: python main.py generate-data")


def run_generate_data():
    """Genera los datos sintéticos de pacientes."""
    from src.scripts.sintetiza_historiales import generar_datos_sinteticos
    generar_datos_sinteticos()


def main():
    parser = argparse.ArgumentParser(
        description="Proyecto de análisis y predicción de Cáncer de Colón"
    )
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    subparsers.add_parser("eda", help="Lanzar la aplicación de EDA con Streamlit")
    subparsers.add_parser("train-polyps", help="Entrenar modelo de detección de pólipos (ResNet18)")
    subparsers.add_parser("train-rf", help="Entrenar Random Forest con datos de pacientes")
    subparsers.add_parser("generate-data", help="Generar datos sintéticos de pacientes")

    args = parser.parse_args()

    commands = {
        "eda": run_eda,
        "train-polyps": run_train_polyps,
        "train-rf": run_train_rf,
        "generate-data": run_generate_data,
    }

    if args.command is None:
        # Sin argumento: Si Streamlit está ejecutando, lanza EDA; si no, muestra help
        if "streamlit" in sys.modules:
            run_eda()
        else:
            parser.print_help()
    elif args.command in commands:
        commands[args.command]()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
