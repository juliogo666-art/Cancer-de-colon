import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import matplotlib.pyplot as plt

# cargar datos
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # apunta a la raiz
csv_path = os.path.join(base_dir, 'src', 'data', 'raw', 'historial_pacientes', 'datos_combinados_completos.csv')
data = pd.read_csv(csv_path)
# separar variables

data["cancer"] = data["Cancer_Stage"] != "Stage 0"
data = data.drop(["Patient_ID", "Cancer_Stage", "Year"], axis=1)
data = pd.get_dummies(data)

X = data.drop("cancer", axis=1)
y = data["cancer"]

# dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# crear modelo
model = XGBClassifier()

# entrenar
model.fit(X_train, y_train)

# predecir
pred = model.predict(X_test)

# evaluar
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)


cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.show()