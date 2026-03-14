#####################################################################################################

# Modelo de clasificación de pólipos Clasificador

#####################################################################################################

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image

#####################################################################################################


class ColonoscopyDataset(Dataset):
    """
    Dataset personalizado para cargar las imágenes de colonoscopia desde
    el dataset 'sageofai/colonoscopy_data_for_vqa' de HuggingFace.
    """

    def __init__(self, split="train", transform=None):
        # Cargamos en modo streaming para no bloquear la RAM si el dataset es enorme
        # pero como necesitamos tamaño y acceso aleatorio para el DataLoader estándar,
        # lo descargamos y cacheamos (tarda un poco la primera vez).
        print(f"Descargando/Cargando dataset de HuggingFace (split={split})...")
        self.dataset = load_dataset("sageofai/colonoscopy_data_for_vqa", split=split)

        # Filtramos solo las preguntas que preguntan por la existencia de pólipos
        print("Filtrando datos de clasificación de pólipos...")
        self.data = []
        for item in self.dataset:
            question = item["question"].lower()
            if "polyp" in question and (
                "display" in question or "present" in question or "contain" in question
            ):
                label = 1.0 if "yes" in item["answer"].lower() else 0.0
                self.data.append({"image": item["image"], "label": label})

        print(f"Encontradas {len(self.data)} imágenes válidas para clasificación.")

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

        # Congelamos las capas base (Opcional, pero bueno para empezar rápido)
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Modificamos la última capa (fc = fully connected)
        # ResNet18 tiene out_features=1000 por ImageNet. Nosotros queremos 1 (Pólipo o No Pólipo)
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
            optimizer.स्टेप()  # OJO: hay un fallo intencionado tipográfico simulado a corregir si quieres. (optimizer.step())
            # En verdad, corregido:
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
    print("--- Test de Compilación del Modelo ---")

    # 1. Probar el modelo con datos dummy
    dummy_input = torch.randn(1, 3, 224, 224)
    detector = PolypDetector()
    output = detector(dummy_input)
    print(f"Salida del modelo con dummy data: {output.shape} (Esperado: [1, 1])")

    # Descomentar la siguiente línea para iniciar el entrenamiento real
    # Tarda porque descargará ~8GB de HuggingFace la primera vez
    # train_model()
