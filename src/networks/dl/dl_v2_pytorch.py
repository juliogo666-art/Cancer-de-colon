import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import random
import numpy as np

# --- 0. REPRODUCIBILIDAD ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. RUTAS ---
RUTA_DATASET = r'C:\Users\Ana-L\Desktop\cosas de Juan\Programacion\cancer de colon\prueba\dataset_colon_completo\dataset_colon_completo\datos_l'
SAVE_DIR = r'C:\Users\Ana-L\Desktop\cosas de Juan\Programacion\cancer de colon\prueba\modelos\dl'
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16 

# --- 2. DATA AUGMENTATION EXTREMO ---
# Esto genera "imágenes nuevas" infinitas en memoria durante el entrenamiento
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(90), # Rotaciones completas
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)), # Zoom y traslación
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1), # Cambios de luz/color
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # Desenfoque aleatorio
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carga de datos
full_dataset = datasets.ImageFolder(RUTA_DATASET)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data_raw, val_data_raw = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Aplicamos transformaciones por separado (importante: val no lleva aumento)
train_data_raw.dataset.transform = transform_train
val_data_raw.dataset.transform = transform_val

train_loader = DataLoader(train_data_raw, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data_raw, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. MODELO ---
class PolipoModel(nn.Module):
    def __init__(self):
        super(PolipoModel, self).__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Congelamos toda la base. Con 1200 fotos NO debemos hacer mucho fine-tuning.
        for param in self.base.parameters():
            param.requires_grad = False
            
        self.features = self.base.features 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(1280)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256), # Capa un poco más ancha
            nn.ReLU(),
            nn.Dropout(0.8), # Dropout extremo (80%) para evitar memorización
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.bn(x)
        x = self.classifier(x)
        return x

model = PolipoModel().to(device)

# --- 4. COMPILACIÓN ---
# Label Smoothing: ayuda a que el modelo no se "sobre-ajuste" a las etiquetas
def label_smoothed_bce(outputs, labels, smoothing=0.1):
    with torch.no_grad():
        labels = labels * (1 - smoothing) + 0.5 * smoothing
    return nn.BCELoss()(outputs, labels)

# Weight Decay (Regularización L2) añadida en Adam
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=5e-5, weight_decay=1e-3)

# --- 5. ENTRENAMIENTO ---
def train_model(epochs=40):
    best_val_loss = float('inf')
    patience = 8
    trigger = 0
    path_mejor_modelo = os.path.join(SAVE_DIR, 'mejor_modelo_anti_overfit.pth')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = label_smoothed_bce(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, tp, fn = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                val_loss += nn.BCELoss()(outputs, labels).item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        acc = correct / len(val_data_raw)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"Época {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f} | Recall: {recall:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), path_mejor_modelo)
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print("--- Early stopping: El modelo dejó de mejorar de forma real ---")
                break

train_model()

# --- 6. EXPORTAR ---
# ... (Mantenemos todo igual hasta la parte de exportación)

# --- 6. EXPORTAR (CORRECCIÓN DE ERROR) ---
print("\n📦 Exportando modelo final...")
try:
    path_pth = os.path.join(SAVE_DIR, 'mejor_modelo_anti_overfit.pth')
    model.load_state_dict(torch.load(path_pth, map_location=device))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Añadimos parámetros de compatibilidad para evitar el error de onnxscript
    torch.onnx.export(
        model, 
        dummy_input, 
        os.path.join(SAVE_DIR, "modelo_anti_overfit.onnx"), 
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # Esta línea ayuda si no tienes onnxscript instalado
        export_params=True 
    )
    print("✅ Proceso terminado y ONNX generado.")
except Exception as e:
    print(f"❌ Error al exportar: {e}")
    print("TIP: Intenta instalar: pip install onnx onnxscript") 