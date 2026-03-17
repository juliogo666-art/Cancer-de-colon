"""
testeos.py — Modelo de clasificación Random Forest para predicción de cáncer de colon.

Recibe un DataFrame con datos de pacientes y entrena un Random Forest con
GridSearchCV para predecir el diagnóstico.
"""

import os

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def entrenar_random_forest(df, target_col='Diagnostico', output_dir=None):
    """
    Entrena un modelo Random Forest con búsqueda de hiperparámetros para
    clasificación binaria de cáncer de colon.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos de pacientes. Debe contener la columna target.
    target_col : str
        Nombre de la columna objetivo (default: 'Diagnostico').
    output_dir : str, optional
        Directorio donde guardar el modelo entrenado.

    Returns
    -------
    dict
        Diccionario con el modelo, mejores parámetros, scores y reporte.
    """
    df_model = df.copy()

    # Eliminar columnas de ID si existen
    cols_to_drop = ['Paciente_ID', 'Patient_ID']
    for col in cols_to_drop:
        if col in df_model.columns:
            df_model = df_model.drop(columns=[col])

    # Codificar columnas categóricas que sigan siendo texto
    le = LabelEncoder()
    for col in df_model.select_dtypes(include=['object']).columns:
        if col != target_col:
            df_model[col] = le.fit_transform(df_model[col].astype(str))

    # Separar features y target
    X = df_model.drop(target_col, axis=1)
    y = df_model[target_col]

    # División entrenamiento/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Configuración del Random Forest y búsqueda de hiperparámetros
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    print("Ejecutando GridSearchCV (esto puede tardar unos minutos)...")
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    # Evaluación con Cross-Validation sobre el set de entrenamiento
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)

    print(f"\nMejores parámetros: {grid_search.best_params_}")
    print(f"Precisión media CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Predicción final sobre test
    y_pred = best_rf.predict(X_test)

    print("\nReporte de Clasificación:")
    report = classification_report(y_test, y_pred)
    print(report)

    print("Matriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Guardar el modelo
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = base_dir

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'random_forest_colon.joblib')
    joblib.dump(best_rf, model_path)
    print(f"\nModelo guardado en: {model_path}")

    return {
        'model': best_rf,
        'best_params': grid_search.best_params_,
        'cv_scores': cv_scores,
        'classification_report': report,
        'confusion_matrix': cm,
        'feature_names': list(X.columns),
    }


if __name__ == "__main__":
    # Ejemplo: cargar el CSV de pacientes sintéticos y entrenar
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        base_dir, '..', 'data', 'raw', 'historial_pacientes',
        'historiales_sinteticos', 'pacientes_simulador_colon.csv'
    )

    if os.path.exists(csv_path):
        print(f"Cargando datos desde: {csv_path}")
        df = pd.read_csv(csv_path)
        resultado = entrenar_random_forest(df)
    else:
        print(f"No se encontró el archivo: {csv_path}")
        print("Genera primero los datos con: python -m src.scripts.sintetiza_historiales")