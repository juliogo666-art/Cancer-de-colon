#####################################################################################################

# Modelo_biopsia_v1_DenseNet

# Modelo de Deep Learning para clasificación de imágenes de biopsias de colon
# Basado en DenseNet121 preentrenada
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
    Clase auxiliar para aplicar transformaciones (data augmentation) 
    solamente a un subconjunto específico de datos (train, val o test).
    Esto soluciona el problema de PyTorch donde random_split hereda 
    la misma transformación para todos los splits.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = self.subset.dataset.loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)

#####################################################################################################

class DenseNetClassifier(nn.Module):
    """
    Clasificador Binario usando DenseNet121 preentrenada.
    """
    def __init__(self):
        super(DenseNetClassifier, self).__init__()
        # Cargar DenseNet121
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        
        # En DenseNet, la última capa se llama 'classifier' (en ResNet es 'fc')
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)

#####################################################################################################

def evaluate_model(model, data_loader, criterion, device):
    """
    Evalúa el modelo en un conjunto de datos dado (validación o test).
    Calcula la pérdida (loss) y varias métricas clínicas clave:
    accuracy, precision, recall y f1-score.
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

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = running_loss / len(data_loader)
    accuracy = sum(p == label for p, label in zip(all_preds, all_labels)) / len(all_labels)
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

def train_biopsy_model():
    print(f"Buscando imagenes en: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: No se encuentra la carpeta de datos.")
        return

    base_dataset = datasets.ImageFolder(root=DATA_DIR)
    print(f"Total de imagenes: {len(base_dataset)}")

    total = len(base_dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_dataset = TransformSubset(train_subset, transform_train)
    val_dataset = TransformSubset(val_subset, transform_eval)
    test_dataset = TransformSubset(test_subset, transform_eval)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Detectar si hay CUDA disponible por si acaso, aunque forzamos CPU si sigue fallando
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando usando: {device}")

    model = DenseNetClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10 
    best_val_loss = float("inf")
    
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": []
    }

    # Corregimos la ruta de guardado para que vaya a la carpeta centralizada de checkpoints
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    checkpoints_dir = os.path.join(project_root, "artifacts", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    best_model_path = os.path.join(checkpoints_dir, "biopsia_densenet121_best.pth")
    final_model_path = os.path.join(checkpoints_dir, "biopsia_densenet121_final.pth")

    print("\n-------------------- Empezando Entrenamiento DenseNet --------------------")
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
            if i % 20 == 0:
                print(f"Epoca [{epoch + 1}/{epochs}], Batch [{i}/{len(train_loader)}], Loss Local: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        print(
            f"--> Resumen Epoca {epoch + 1}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_metrics["loss"])
        training_history["val_accuracy"].append(val_metrics["accuracy"])
        training_history["val_f1"].append(val_metrics["f1"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  [!] Nuevo record: el modelo fue guardado en {best_model_path}")

    print("\n-------------------- Fase de Test DenseNet --------------------")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    print(f"Perdida (Loss) Test     : {test_metrics['loss']:.4f}")
    print(f"Exactitud (Accuracy)    : {test_metrics['accuracy']:.4f}")
    print(f"Precision (Precision)   : {test_metrics['precision']:.4f}")
    print(f"Sensibilidad (Recall)   : {test_metrics['recall']:.4f}")
    print(f"Puntuacion F1 (F1-score): {test_metrics['f1']:.4f}")

    torch.save(model.state_dict(), final_model_path)
    
    # Registro en el Tracker
    tracker = ExperimentTracker()
    tracker.log_experiment(
        model_name="BiopsyDenseNet121",
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
