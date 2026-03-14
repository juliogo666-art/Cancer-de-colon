#####################################################################################################

# Modelo de clasificación de pólipos Clasificador

# fuente con 680 imágenes válidas y balanceadas

# Compilamos una red neuronal profunda ResNet18 precargada con pesos de ImageNet.
# Ajustamos la capa final fc para resolver un problema de Clasificación Binaria
# (BCEWithLogitsLoss).
# Le inyectamos técnicas de Data Augmentation en tiempo real
# (RandomHorizontalFlip, RandomRotation, ColorJitter) para garantizar que el modelo
# no se memorizara las 680 imágenes sino que aprendiera las texturas de los pólipos
# desde diferentes ángulos y condiciones de luz.

#####################################################################################################

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env local
load_dotenv()

#####################################################################################################


class ColonoscopyDataset(Dataset):
    """
    Dataset personalizado para cargar las imágenes de colonoscopia desde
    el dataset 'sageofai/colonoscopy_data_for_vqa' de HuggingFace.
    """

    def __init__(self, split="train", transform=None):
        # Cargamos en modo streaming para no bloquear la RAM si el dataset es enorme
        # pero como necesitamos tamaño y acceso aleatorio para el DataLoader estándar,
        # lo descargamos y cacheamos.
        print(f"Descargando/Cargando dataset de HuggingFace (split={split})...")
        self.dataset = load_dataset(
            "sageofai/colonoscopy_data_for_vqa",
            split=split,
            verification_mode="no_checks",
            cache_dir="src/data/raw/huggingface_vqa_dataset",  # <- Esto les obligaría
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

        import random

        # Fijamos semilla para que siempre coja las mismas aleatorias
        random.seed(42)

        # Balanceamos las clases: cogemos el mínimo entre ambas para maximizar los datos sin desbalancearlo
        # Ahora cogerá todas las imágenes posibles (aprox.3900 de cada clase, 7800 total).
        limit_per_class = min(len(positives), len(negatives))

        self.data = random.sample(positives, limit_per_class) + random.sample(
            negatives, limit_per_class
        )
        random.shuffle(self.data)

        print(
            f"Dataset final balanceado. Encontrados: {limit_per_class} Polipos y {limit_per_class} Sanos (Total: {len(self.data)})."
        )

        self.transform = transform

        # Si no se provee transformación, usamos una básica
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        label = item["label"]

        # Asegurarnos de que está en RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Devolvemos la imagen y la etiqueta como un tensor flotante (para BCEWithLogitsLoss)
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
        # ResNet18 tiene out_features=1000 por ImageNet, es multicategorico (perros,barcos ...)
        # Nosotros queremos 1 (Pólipo o No Pólipo), se volveria loco con 1000 salidas.
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)


def train_model():
    """Ejemplo de función de entrenamiento"""
    print("Preparando Data Augmentation y Dataset...")
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2
            ),  # Simula cambios luz endoscopio
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # IMPORTANTE: Esto puede tardar la primera vez
    train_dataset = ColonoscopyDataset(split="train", transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = PolypDetector().to(device)

    # Loss para clasificacion binaria (ya incluye Sigmoid interno, por eso sale 1 neurona lineal)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    print("\n--- Iniciando Entrenamiento ---")
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
            if i % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        print(
            f"Fin Epoch {epoch + 1} - Loss Promedio: {running_loss / len(train_loader):.4f}"
        )

    print("Entrenamiento completado.")
    torch.save(model.state_dict(), "polyp_resnet18.pth")
    print("Modelo guardado como polyp_resnet18.pth")


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
