#####################################################################################################

# Modelo de clasificación de pólipos Clasificador

# El dataset original cuenta con un aproximado de 11.000 imágenes, de las cuales 
# 3.912 contienen pólipos. Para garantizar el correcto aprendizaje de la red y 
# evitar el desbalanceo de clases, se toma de forma aleatoria una muestra de 
# 3.912 imágenes sin pólipos, obteniendo así un dataset balanceado de 7.824 imágenes.

# Compilamos una red neuronal profunda ResNet18 precargada con pesos de ImageNet.
# Ajustamos la capa final fc para resolver un problema de Clasificación Binaria
# (BCEWithLogitsLoss).
# Le inyectamos técnicas de Data Augmentation en tiempo real
# (RandomHorizontalFlip, RandomRotation, ColorJitter) para garantizar que el modelo
# no se memorizara las 7.824 imágenes sino que aprendiera las texturas de los pólipos
# desde diferentes ángulos y condiciones de luz.

#####################################################################################################

import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env local
load_dotenv()


#####################################################################################################
# Transforms separados para entrenamiento y evaluación
#####################################################################################################

# Normalización estándar de ImageNet (usada en train, val y test)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transform CON augmentation — solo para entrenamiento
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Transform SIN augmentation — para validación y test
transform_eval = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


#####################################################################################################


class ColonoscopyDataset(Dataset):
    """
    Dataset personalizado para cargar las imágenes de colonoscopia desde
    el dataset 'sageofai/colonoscopy_data_for_vqa' de HuggingFace.
    """

    def __init__(self, split="train", transform=None):
        # Cargamos el dataset y lo cacheamos para acceso aleatorio
        print(f"Descargando/Cargando dataset de HuggingFace (split={split})...")
        self.dataset = load_dataset(
            "sageofai/colonoscopy_data_for_vqa",
            split=split,
            verification_mode="no_checks",
            cache_dir="src/data/raw/huggingface_vqa_dataset",
        )

        # Filtramos positivas (pólipos) y negativas (sano)
        print("Filtrando y balanceando el dataset de polipos vs sano...")
        positives = []
        negatives = []

        for item in self.dataset:
            text = item["text"].lower()
            if "polyp" in text:
                positives.append({"image": item["image"], "label": 1.0})
            elif "finding" not in text and "instrument" not in text:
                negatives.append({"image": item["image"], "label": 0.0})

        # Fijamos semilla para que siempre coja las mismas aleatorias
        random.seed(42)

        # Balanceamos las clases
        limit_per_class = min(len(positives), len(negatives))

        self.data = random.sample(positives, limit_per_class) + random.sample(
            negatives, limit_per_class
        )
        random.shuffle(self.data)

        print(
            f"Dataset final balanceado. Encontrados: {limit_per_class} Polipos y "
            f"{limit_per_class} Sanos (Total: {len(self.data)})."
        )

        # Si no se provee transformación, usamos la de evaluación (sin augmentation)
        self.transform = transform if transform is not None else transform_eval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        label = item["label"]

        # Asegurarnos de que está en RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)

        # Devolvemos la imagen y la etiqueta como un tensor flotante (para BCEWithLogitsLoss)
        return image, torch.tensor([label], dtype=torch.float32)


#####################################################################################################


class TransformSubset(Dataset):
    """
    Wrapper para aplicar un transform diferente a un subset de datos.
    Necesario porque random_split devuelve Subsets que heredan el transform
    del dataset original, pero queremos augmentation solo en train.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        item = self.subset.dataset.data[self.subset.indices[idx]]
        image = item["image"]
        label = item["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)


#####################################################################################################


class PolypDetector(nn.Module):
    """
    Clasificador Binario usando ResNet18 preentrenada.
    """

    def __init__(self):
        super(PolypDetector, self).__init__()
        # Cargamos pesos preentrenados usando el método moderno de torchvision
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Congela las capas base --> si necesitamos que vaya mas rapido porque la RAM no da.
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Modificamos la última capa (fc = fully connected)
        # ResNet18 tiene out_features=1000 por ImageNet, es multicategorico (perros, barcos ...)
        # Nosotros queremos 1 (Pólipo o No Pólipo).
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)


#####################################################################################################


def evaluate_model(model, data_loader, criterion, device):
    """
    Evalúa el modelo sobre un DataLoader dado.

    Returns
    -------
    dict
        Diccionario con loss, accuracy, precision, recall y f1.
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

            # Convertir logits a predicciones binarias (umbral 0.5 via sigmoid)
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


def train_model():
    """Función de entrenamiento con validación y evaluación final en test."""
    print("Preparando Data Augmentation y Dataset...")

    # IMPORTANTE: Cargamos el dataset completo SIN augmentation,
    # luego aplicamos el transform correcto a cada split
    full_dataset = ColonoscopyDataset(split="train", transform=transform_eval)

    # Split: 70% train, 15% val, 15% test
    total = len(full_dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Aplicar augmentation SOLO al train, eval transform a val y test
    train_dataset = TransformSubset(train_subset, transform_train)
    val_dataset = TransformSubset(val_subset, transform_eval)
    test_dataset = TransformSubset(test_subset, transform_eval)

    print(f"Split: {train_size} train / {val_size} val / {test_size} test")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = PolypDetector().to(device)

    # Loss para clasificacion binaria (ya incluye Sigmoid interno)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    best_val_loss = float("inf")

    print("\n--- Iniciando Entrenamiento ---")
    for epoch in range(epochs):
        # ---- Fase de entrenamiento ----
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
            if i % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{i}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)

        # ---- Fase de validación ----
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        print(
            f"Fin Epoch {epoch + 1} — "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Guardar el mejor modelo según val loss (early stopping simple)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model_path_best = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polyp_resnet18_best.pth")
            torch.save(model.state_dict(), model_path_best)
            print(f"  → Mejor modelo guardado (Val Loss: {best_val_loss:.4f})")

    # ---- Evaluación final sobre Test ----
    print("\n--- Evaluación Final en Test ---")
    # Cargar el mejor modelo
    model_path_best = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polyp_resnet18_best.pth")
    model.load_state_dict(torch.load(model_path_best, weights_only=True))
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    print(f"Test Loss:      {test_metrics['loss']:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1']:.4f}")

    # Guardar modelo final
    model_path_final = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polyp_resnet18.pth")
    torch.save(model.state_dict(), model_path_final)
    print("\nModelo final guardado en la misma carpeta que el script (polyp_resnet18.pth)")


#####################################################################################################

if __name__ == "__main__":
    print("--- Test de Compilacion del Modelo ---")

    # 1. Probar el modelo
    dummy_input = torch.randn(1, 3, 224, 224)
    detector = PolypDetector()
    output = detector(dummy_input)
    print(f"Salida del modelo: {output.shape} (Esperado: [1, 1])")

    # Iniciar el entrenamiento
    # Tarda porque descargará el dataset de HuggingFace la primera vez
    train_model()
