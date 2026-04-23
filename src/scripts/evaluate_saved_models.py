"""
Script para evaluar todos los modelos guardados (ML Tabular y DL Imágenes)
y generar un informe final centralizado.
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

import joblib
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split, Dataset
from datasets import load_dataset
import warnings

# Si keras/tensorflow está disponible
try:
    from keras.models import load_model
except ImportError:
    pass

from src.config.settings import settings
from src.metrics.accuracy import AccuracyMetric
from src.metrics.precision import PrecisionMetric
from src.metrics.recall import RecallMetric
from src.metrics.f_score import FBetaMetric
from src.metrics.confusion import ConfusionMatrixMetric
from src.metrics.roc_auc import ROCAUCMetric
from src.pipelines.evaluation_pipeline import ModelEvaluationPipeline

warnings.filterwarnings('ignore')

# --- REPRODUCIBILIDAD ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================================
# DEFINICIONES DE ARQUITECTURA DE MODELOS DL
# ====================================================================

class BiopsyResNet(nn.Module):
    def __init__(self):
        super(BiopsyResNet, self).__init__()
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
    def forward(self, x): return self.model(x)

class BiopsyDenseNet(nn.Module):
    def __init__(self):
        super(BiopsyDenseNet, self).__init__()
        self.model = models.densenet121(weights=None)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 1)
    def forward(self, x): return self.model(x)

class PolypDetector(nn.Module):
    def __init__(self):
        super(PolypDetector, self).__init__()
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
    def forward(self, x): return self.model(x)

class PolipoModel(nn.Module): # MobileNetV2 para colonoscopias (anti overfit)
    def __init__(self):
        super(PolipoModel, self).__init__()
        self.base = models.mobilenet_v2(weights=None)
        self.features = self.base.features 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(1280)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.bn(x)
        return self.classifier(x)

# Utils para datasets DL
class TransformSubset(Dataset):
    def __init__(self, subset, transform, is_hf=False):
        self.subset = subset
        self.transform = transform
        self.is_hf = is_hf

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        if self.is_hf:
            item = self.subset.dataset.data[self.subset.indices[idx]]
            image = item["image"]
            label = item["label"]
            if image.mode != "RGB": image = image.convert("RGB")
            image = self.transform(image)
            return image, torch.tensor([label], dtype=torch.float32)
        else:
            image_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            image = self.subset.dataset.loader(image_path)
            if self.transform: image = self.transform(image)
            return image, torch.tensor([label], dtype=torch.float32)

class ColonoscopyHFDataset(Dataset):
    def __init__(self):
        self.dataset = load_dataset(
            "sageofai/colonoscopy_data_for_vqa", split="train",
            verification_mode="no_checks", cache_dir="src/data/raw/huggingface_vqa_dataset"
        )
        positives, negatives = [], []
        for item in self.dataset:
            text = item["text"].lower()
            if "polyp" in text:
                positives.append({"image": item["image"], "label": 1.0})
            elif "finding" not in text and "instrument" not in text:
                negatives.append({"image": item["image"], "label": 0.0})
        
        random.seed(42)
        limit = min(len(positives), len(negatives))
        self.data = random.sample(positives, limit) + random.sample(negatives, limit)
        random.shuffle(self.data)

    def __len__(self): return len(self.data)

# ====================================================================
# EVALUADORES
# ====================================================================

def predict_dl_model(model, loader, is_keras=False):
    all_preds, all_probs, all_labels = [], [], []
    if not is_keras:
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                
                if outputs.shape[1] == 1: # Clasificacion binaria con BCEWithLogitsLoss / BCELoss
                    if not isinstance(model, PolipoModel): # PolipoModel ya devuelve Sigmoid internamente
                        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    else:
                        probs = outputs.cpu().numpy().flatten()
                else: # Softmax
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().flatten()
                
                preds = (probs >= 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy().flatten())
    else:
        for images, labels in loader:
            # images to numpy for keras
            imgs_np = images.permute(0, 2, 3, 1).numpy()
            probs = model.predict(imgs_np, verbose=0).flatten()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().flatten())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def evaluate_tabular_kaggle(eval_pipeline):
    from sklearn.model_selection import train_test_split
    
    file_path = os.path.join(settings.CARPETA_HISTORIALES, 'datos_finales_Kaggle.csv')
    if not os.path.exists(file_path):
        print(f"[!] No se encontró el dataset Kaggle: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    features = [
        'Age', 'Gender', 'Cancer_Stage', 'Tumor_Size_mm', 'Family_History', 
        'Smoking_History', 'Alcohol_Consumption', 'Obesity_BMI', 
        'Inflammatory_Bowel_Disease', 'FOBT_Resultado_n', 
        'CEA_Level_ng_mL..Marcador.Tumoral.', 'altura_cm', 'peso_kg'
    ]
    target = 'Survival_Prediction'
    
    X = df[features].values
    y = df[target].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    modelos_kaggle = {
        "best_rf_model": "artifacts/weights/best_rf_model.pkl",
        "xgb_sensible": "artifacts/weights/xgb_sensible.pkl",
        "lgbm_sensible": "artifacts/weights/lgbm_sensible.pkl"
    }
    
    for nombre, path in modelos_kaggle.items():
        if os.path.exists(path):
            model = joblib.load(path)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)
            # Para modelos sensibles el umbral era ajustado, pero evaluamos estandar para el pipeline
            eval_pipeline.evaluate_model(nombre, y_test, preds, proba)
            print(f"  [OK] {nombre} evaluado.")

def evaluate_tabular_risk(eval_pipeline):
    from sklearn.model_selection import train_test_split
    if not os.path.exists(settings.CSV_MASTER_PATH): return
    
    df = pd.read_csv(settings.CSV_MASTER_PATH)
    target = "Risk_Level_n"
    
    feat_clinico = settings.ML_FEATURE_NAMES
    feat_triage = [f for f in feat_clinico if f not in ["FOBT_Resultado_n", "CEA_Level_ng_mL"]]
    
    _, X_test_c, _, y_test_c = train_test_split(df[feat_clinico], df[target], test_size=0.2, random_state=42, stratify=df[target])
    _, X_test_t, _, y_test_t = train_test_split(df[feat_triage], df[target], test_size=0.2, random_state=42, stratify=df[target])
    
    modelos_risk = {
        "lgbm_clinico": ("artifacts/weights/lgbm_clinico.pkl", X_test_c, y_test_c),
        "rf_clinico": ("artifacts/weights/rf_clinico.pkl", X_test_t, y_test_t),
        "xgb_clinico": ("artifacts/weights/xgb_clinico.pkl", X_test_t, y_test_t),
        "lgbm_triage": ("artifacts/weights/lgbm_triage.pkl", X_test_t, y_test_t)
    }
    
    for nombre, (path, X_t, y_t) in modelos_risk.items():
        if os.path.exists(path):
            model = joblib.load(path)
            preds = model.predict(X_t)
            proba = model.predict_proba(X_t)
            eval_pipeline.evaluate_model(nombre, y_t, preds, proba)
            print(f"  [OK] {nombre} evaluado.")

def evaluate_biopsies(eval_pipeline):
    if not os.path.exists(settings.CARPETA_IMAGENES_COLON): return
    
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    base_dataset = datasets.ImageFolder(root=settings.CARPETA_IMAGENES_COLON)
    total = len(base_dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    _, _, test_subset = random_split(base_dataset, [train_size, val_size, test_size], generator=generator)
    test_dataset = TransformSubset(test_subset, transform_eval)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    modelos = {
        "biopsia_densenet121_best": ("artifacts/checkpoints/biopsia_densenet121_best.pth", BiopsyDenseNet()),
        "biopsia_resnet18_best": ("artifacts/checkpoints/biopsia_resnet18_best.pth", BiopsyResNet())
    }
    
    for nombre, (path, architecture) in modelos.items():
        if os.path.exists(path):
            model = architecture.to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            y_true, y_pred, y_prob = predict_dl_model(model, test_loader)
            eval_pipeline.evaluate_model(nombre, y_true, y_pred, y_prob)
            print(f"  [OK] {nombre} evaluado.")

def evaluate_polyps(eval_pipeline):
    try:
        full_dataset = ColonoscopyHFDataset()
    except Exception as e:
        print(f"[!] Error cargando dataset de HuggingFace: {e}")
        return
        
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    total = len(full_dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    _, _, test_subset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    
    test_dataset = TransformSubset(test_subset, transform_eval, is_hf=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    path = "artifacts/weights/polyp_resnet18_best.pth"
    if os.path.exists(path):
        model = PolypDetector().to(device)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        # Handle dict wrapper if exists
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        
        y_true, y_pred, y_prob = predict_dl_model(model, test_loader)
        eval_pipeline.evaluate_model("polyp_resnet18_best", y_true, y_pred, y_prob)
        print("  [OK] polyp_resnet18_best evaluado.")

def evaluate_colonoscopy(eval_pipeline):
    if not os.path.exists(settings.CARPETA_IMAGENES_COLONOSCOPIA): 
        print(f"[!] Carpeta no encontrada: {settings.CARPETA_IMAGENES_COLONOSCOPIA}")
        return
        
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(settings.CARPETA_IMAGENES_COLONOSCOPIA)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    _, val_data_raw = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    
    val_data_raw.dataset.transform = transform_val
    test_loader = DataLoader(val_data_raw, batch_size=16, shuffle=False)
    
    # Modelo PyTorch (MobileNetV2)
    path_pth = "artifacts/weights/mejor_modelo_anti_overfit.pth"
    if os.path.exists(path_pth):
        model = PolipoModel().to(device)
        model.load_state_dict(torch.load(path_pth, map_location=device, weights_only=True))
        y_true, y_pred, y_prob = predict_dl_model(model, test_loader)
        eval_pipeline.evaluate_model("mejor_modelo_anti_overfit", y_true, y_pred, y_prob)
        print("  [OK] mejor_modelo_anti_overfit evaluado.")
        
    # Modelo Keras
    path_keras = "artifacts/weights/modelo_pro_agresivo.keras"
    if os.path.exists(path_keras):
        try:
            k_model = load_model(path_keras)
            y_true, y_pred, y_prob = predict_dl_model(k_model, test_loader, is_keras=True)
            eval_pipeline.evaluate_model("modelo_pro_agresivo", y_true, y_pred, y_prob)
            print("  [OK] modelo_pro_agresivo evaluado.")
        except Exception as e:
            print(f"[!] Error cargando Keras modelo: {e}")

# ====================================================================
# MAIN
# ====================================================================

def main():
    print("Iniciando Pipeline de Evaluación Maestro (Tabulares + Imágenes)...")
    
    eval_pipeline = ModelEvaluationPipeline(metrics=[
        AccuracyMetric(),
        PrecisionMetric(),
        RecallMetric(),
        FBetaMetric(beta=1),
        ConfusionMatrixMetric(),
        ROCAUCMetric()
    ])

    print("\n--- 1. Evaluando Modelos Tabulares (Datos Kaggle) ---")
    evaluate_tabular_kaggle(eval_pipeline)
    
    print("\n--- 2. Evaluando Modelos Tabulares (Datos Risk) ---")
    evaluate_tabular_risk(eval_pipeline)
    
    print("\n--- 3. Evaluando Modelos de Biopsias (PyTorch DL) ---")
    evaluate_biopsies(eval_pipeline)
    
    print("\n--- 4. Evaluando Modelo de Pólipos (HuggingFace + PyTorch DL) ---")
    evaluate_polyps(eval_pipeline)
    
    print("\n--- 5. Evaluando Modelos de Colonoscopia (PyTorch + Keras DL) ---")
    evaluate_colonoscopy(eval_pipeline)
    
    print("\n--- RESULTADOS FINALES ---")
    eval_pipeline.print_report()
    
    df_summary = eval_pipeline.get_summary_dataframe()
    artifacts_dir = os.path.join("artifacts", "reports")
    os.makedirs(artifacts_dir, exist_ok=True)
    report_path = os.path.join(artifacts_dir, "model_evaluation_report_full.csv")
    
    df_summary.to_csv(report_path, index=False)
    print(f"\n[OK] Reporte completo guardado en: {report_path}")

if __name__ == "__main__":
    main()
