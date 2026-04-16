#####################################################################################################

# Modelo_biopsia_v0

# Modelo de Deep Learning para clasificación de imágenes de biopsias de colon
# Basado en ResNet18 preentrenada
# Clasificación binaria: Maligno o Benigno

#####################################################################################################

import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import ImageFile
from src.tracking.experiment_tracker import ExperimentTracker

# Para evitar errores si hay imágenes corruptas o truncadas en el dataset original
ImageFile.LOAD_TRUNCATED_IMAGES = True

#####################################################################################################
# Variables globales y Transformaciones
#####################################################################################################

# Calcula la ruta automáticamente al dataset de biopsias
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "raw",
    "colon_image_sets",
)

# Normalización estándar de ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transformaciones para Data Augmentation (Solo para Entrenamiento)
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Transformaciones sin aumento (Para Validación y Test)
transform_eval = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

#####################################################################################################


class TransformSubset(torch.utils.data.Dataset):
    """
    Wrapper para aplicar una transformación específica a un subconjunto de datos
    creado con random_split, de manera que el Data Augmentation solo se aplique en Train
    y NO durante validación ni test.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Obtenemos la ruta y la etiqueta desde el dataset original ImageFolder
        image_path, label = self.subset.dataset.samples[self.subset.indices[idx]]

        # Cargamos la imagen directamente usando el loader de ImageFolder (RGB)
        image = self.subset.dataset.loader(image_path)

        if self.transform:
            image = self.transform(image)

        # Para usar BCEWithLogitsLoss, necesitamos la etiqueta como float de dimensión [1]
        return image, torch.tensor([label], dtype=torch.float32)


#####################################################################################################


class BiopsyClassifier(nn.Module):
    """
    Clasificador Binario usando ResNet18 preentrenada, ajustada para diferenciar
    entre biopsias malignas y benignas.
    """

    def __init__(self):
        super(BiopsyClassifier, self).__init__()
        # Cargar ResNet18 con pesos preentrenados de ImageNet
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modificar la última capa conectada (fc) para tener solo 1 salida (Clasificación Binaria)
        # La salida nos dirá si es Maligno o Benigno
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)


#####################################################################################################


def evaluate_model(model, data_loader, criterion, device):
    """
    Evalúa el modelo sobre un DataLoader y calcula las métricas.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Usamos Sigmoide > 0.5 para predecir si es clase 1 o 0
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = running_loss / len(data_loader)
    accuracy = sum(p == label for p, label in zip(all_preds, all_labels)) / len(
        all_labels
    )
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


#####################################################################################################


def train_biopsy_model():
    print(f"Buscando imagenes en: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: No se encuentra la carpeta de datos en {DATA_DIR}.")
        return

    # Usamos ImageFolder sin transform inicial para leer todo el directorio.
    # ImageFolder mapea automáticamente las subcarpetas alfabéticamente a números (0, 1)
    # colon_aca (Adenocarcinoma) -> 0
    # colon_n (Benigno/Normal) -> 1
    base_dataset = datasets.ImageFolder(root=DATA_DIR)

    print(f"Clases detectadas automaticamente: {base_dataset.class_to_idx}")
    print(f"Total de imagenes: {len(base_dataset)}")

    # Divisiones (Split): 70% Entreno, 15% Validación, 15% Test
    total = len(base_dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(
        42
    )  # Semilla para resultados reproducibles
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Inyectamos el Data Augmentation correspondientemente
    train_dataset = TransformSubset(train_subset, transform_train)
    val_dataset = TransformSubset(val_subset, transform_eval)
    test_dataset = TransformSubset(test_subset, transform_eval)

    print(
        f"Tamanos -> {train_size} Entrenamiento | {val_size} Validacion | {test_size} Prueba (Test)"
    )

    # DataLoaders: Si pones num_workers > 0 puede dar problemas de multiprocesamiento en Windows, por eso lo dejo a 0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Forzamos CPU debido a incompatibilidad temporal de drivers CUDA con RTX 50xx en esta versión de PyTorch
    device = torch.device("cpu")
    print(f"Entrenando usando: {device} (Forzado por compatibilidad)")

    model = BiopsyClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10  # Veces que el modelo verá todos los datos del dataset
    best_val_loss = float("inf")
    
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": []
    }

    # Preparamos rutas relativas para el guardado
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    model_dir = os.path.join(project_root, "artifacts", "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    # Mejor modelo, es decir menor perdida con datos de validación,
    # pero no necesariamente el que tenga mayor precision o recall
    # evitamos overfitting
    best_model_path = os.path.join(model_dir, "biopsia_resnet18_best.pth")
    # Ultimo modelo entrenado, en la 10 epoca, por si fuera mejorable
    final_model_path = os.path.join(model_dir, "biopsia_resnet18_final.pth")

    print("\n-------------------- Empezando Entrenamiento --------------------")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print de estado a lo largo del epoch
            if i % 20 == 0:
                print(
                    f"Epoca [{epoch + 1}/{epochs}], Batch [{i}/{len(train_loader)}], Loss Local: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)

        # Validación finalizando el epoch
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        print(
            f"--> Resumen Epoca {epoch + 1}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Guardamos estadísticas para el historial
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_metrics['loss'])
        training_history["val_accuracy"].append(val_metrics['accuracy'])
        training_history["val_f1"].append(val_metrics['f1'])

        # Guardamos el modelo solo cuando se supere a si mismo (Mejor Pérdida / Loss de Validación)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  [!] Nuevo record: el modelo fue guardado en {best_model_path}")

    print(
        "\n-------------------- Fase de Test (Imagenes Desconocidas) --------------------"
    )
    # Cargamos la versión que sacó mejores notas en validación
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    print(f"Perdida (Loss) Test     : {test_metrics['loss']:.4f}")
    print(f"Exactitud (Accuracy)    : {test_metrics['accuracy']:.4f}")
    print(f"Precision (Precision)   : {test_metrics['precision']:.4f}")
    print(f"Sensibilidad (Recall)   : {test_metrics['recall']:.4f}")
    print(f"Puntuacion F1 (F1-score): {test_metrics['f1']:.4f}")

    # Guardo una copia de la versión de la última época por si nos interesase analizar sobreajuste.
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinalizado. Check Final guardado en: {final_model_path}")
    
    # Registro en el Tracker
    tracker = ExperimentTracker()
    tracker.log_experiment(
        model_name="BiopsyResNet18",
        metrics=test_metrics,
        hyperparameters={"epochs": epochs, "learning_rate": 1e-4, "batch_size": 32},
        dataset_path=DATA_DIR,
        model_path=best_model_path,
        train_size=train_size,
        test_size=test_size,
        training_history=training_history
    )


if __name__ == "__main__":
    train_biopsy_model()
