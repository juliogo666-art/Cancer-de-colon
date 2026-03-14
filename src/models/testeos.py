import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

# Cargar el dataset correcto con ruta absoluta y dinámica
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # apunta a la raiz
csv_path = os.path.join(base_dir, 'src', 'data', 'raw', 'historial_pacientes', 'historiales_sinteticos', 'pacientes_simulador_colon.csv')
df = pd.read_csv(csv_path)

# Preprocesamiento rápido
df_model = df.drop(columns=['Paciente_ID'])
le = LabelEncoder()
df_model['Genero'] = le.fit_transform(df_model['Genero'])
df_model['FOBT_Resultado (Sangre en heces)'] = le.fit_transform(df_model['FOBT_Resultado (Sangre en heces)'])

X = df_model.drop('Diagnostico', axis=1)
y = df_model['Diagnostico']

# División entrenamiento/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuración del Random Forest y búsqueda de hiperparámetros
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# Evaluación con Cross-Validation sobre el set de entrenamiento
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Precisión media CV: {cv_scores.mean():.4f}")

# Predicción final
y_pred = best_rf.predict(X_test)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))