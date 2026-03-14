import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
import os
import matplotlib.pyplot as plt

# cargar datos
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # apunta a la raiz
csv_path = os.path.join(base_dir, 'src', 'data', 'raw', 'historial_pacientes', 'datos_combinados_completos.csv')
data = pd.read_csv(csv_path)

# crear variable objetivo
data["cancer"] = data["Cancer_Stage"] != "Stage 0"

# eliminar columnas que no sirven
data = data.drop(["Patient_ID", "Cancer_Stage"], axis=1)

# convertir variables categóricas a números
data = pd.get_dummies(data)

# separar variables
X = data.drop("cancer", axis=1)
y = data["cancer"]

# dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# crear modelo de regresión logística
model = LogisticRegression(max_iter=1000)

# entrenar
model.fit(X_train, y_train)

# predecir
pred = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

# matriz de confusión
cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.show()