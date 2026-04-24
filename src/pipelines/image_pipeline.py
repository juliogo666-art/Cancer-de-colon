"""
=============================================================================
Pipeline de análisis de imagen médica
=============================================================================
Encapsula el flujo completo de inferencia de imagen para:
    1. Colonoscopia (TensorFlow/Keras - MobileNetV2): Detección de pólipos
    2. Biopsias (PyTorch - ResNet18): Clasificación benigno/maligno

Incluye preprocesamiento, inferencia y generación de Grad-CAM, todo
en un pipeline limpio y desacoplado del frontend Streamlit.

Uso:
    from src.pipelines.image_pipeline import ImageAnalysisPipeline

    pipeline = ImageAnalysisPipeline()
    resultado = pipeline.analyze_colonoscopy(img_array)
    resultado = pipeline.analyze_biopsy(img_array)
=============================================================================
"""

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


@dataclass
class ImageAnalysisResult:
    """Resultado estandarizado de cualquier análisis de imagen."""

    diagnosis: str
    is_positive: bool  # True = hallazgo detectado (pólipo / maligno)
    confidence: float  # 0.0 a 1.0
    raw_prediction: float  # Valor crudo de la predicción
    recommendation: str
    heatmap: Optional[np.ndarray] = None  # Imagen Grad-CAM (RGB, uint8)


class ImageAnalysisPipeline:
    """
    Pipeline unificado de análisis de imagen médica.

    Carga los modelos bajo demanda y cachea las instancias para reutilización.
    """

    def __init__(self):
        self._modelo_cnn = None
        self._modelo_biopsia = None

    ######################################################################
    # Carga de modelos
    ######################################################################

    def _get_cnn_model(self):
        """Carga el modelo CNN de colonoscopia."""
        if self._modelo_cnn is not None:
            return self._modelo_cnn

        from src.api.dependencies import load_cnn_model

        self._modelo_cnn = load_cnn_model()
        return self._modelo_cnn

    def _get_biopsy_model(self):
        """Carga el modelo de biopsias."""
        if self._modelo_biopsia is not None:
            return self._modelo_biopsia

        from src.api.dependencies import load_biopsy_model

        self._modelo_biopsia = load_biopsy_model()
        return self._modelo_biopsia

    ######################################################################
    # Análisis de colonoscopia
    ######################################################################

    def analyze_colonoscopy(
        self,
        img_array: np.ndarray,
        generate_heatmap: bool = True,
    ) -> ImageAnalysisResult:
        """
        Analiza una imagen de colonoscopia para detectar pólipos.

        Parameters
        ----------
        img_array : np.ndarray
            Imagen en formato RGB (H, W, 3), valores 0-255.
        generate_heatmap : bool
            Si True, genera el mapa de calor Grad-CAM.

        Returns
        -------
        ImageAnalysisResult
        """
        # TensorFlow imports comentados — El modelo CNN ahora es PyTorch (MobileNetV2).
        # Si hubiera que volver a TensorFlow, descomentar estas líneas:
        # import tensorflow as tf
        # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        modelo = self._get_cnn_model()
        if modelo is None:
            return ImageAnalysisResult(
                diagnosis="ERROR",
                is_positive=False,
                confidence=0.0,
                raw_prediction=0.0,
                recommendation="Modelo CNN no disponible.",
            )

        # 1. Preprocesamiento (PyTorch MobileNetV2)
        img_pil = Image.fromarray(img_array)
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(img_pil).unsqueeze(0)

        # 2. Predicción
        with torch.no_grad():
            output = modelo(input_tensor)
            pred_val = output.item()
        es_polipo = pred_val < 0.5
        confianza = (1.0 - pred_val) if es_polipo else pred_val

        # 3. Grad-CAM (opcional)
        heatmap = None
        if generate_heatmap:
            try:
                from src.utils.gradcam_utils import generate_gradcam_colon

                heatmap, _ = generate_gradcam_colon(modelo, img_array)
            except Exception:
                pass

        return ImageAnalysisResult(
            diagnosis="PÓLIPO DETECTADO" if es_polipo else "TEJIDO SANO",
            is_positive=es_polipo,
            confidence=round(confianza, 4),
            raw_prediction=round(pred_val, 4),
            recommendation=(
                "Se recomienda revisión inmediata por especialista."
                if es_polipo
                else "No se observan anomalías evidentes."
            ),
            heatmap=heatmap,
        )

    ######################################################################
    # Análisis de biopsias
    ######################################################################

    def analyze_biopsy(
        self,
        img_array: np.ndarray,
        generate_heatmap: bool = True,
    ) -> ImageAnalysisResult:
        """
        Analiza una imagen de biopsia para clasificar tejido benigno vs maligno.

        Parameters
        ----------
        img_array : np.ndarray
            Imagen en formato RGB (H, W, 3), valores 0-255.
        generate_heatmap : bool
            Si True, genera el mapa de calor Grad-CAM.

        Returns
        -------
        ImageAnalysisResult
        """
        modelo = self._get_biopsy_model()
        if modelo is None:
            return ImageAnalysisResult(
                diagnosis="ERROR",
                is_positive=False,
                confidence=0.0,
                raw_prediction=0.0,
                recommendation="Modelo de biopsias no disponible.",
            )

        # 1. Convertir a PIL
        img_pil = Image.fromarray(img_array)
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        # 2. Preprocesamiento PyTorch
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = transform(img_pil).unsqueeze(0)

        # 3. Inferencia
        with torch.no_grad():
            output = modelo(input_tensor)
            prob = torch.sigmoid(output).item()

        es_benigno = prob >= 0.5
        confianza = prob if es_benigno else (1.0 - prob)

        # 4. Grad-CAM (opcional)
        heatmap = None
        if generate_heatmap:
            try:
                from src.utils.gradcam_utils import generate_gradcam

                target_layer = modelo.model.features.denseblock4
                heatmap, _ = generate_gradcam(modelo, img_pil, target_layer)
            except Exception:
                pass

        return ImageAnalysisResult(
            diagnosis="BENIGNO (NORMAL)" if es_benigno else "MALIGNO (ADENOCARCINOMA)",
            is_positive=not es_benigno,  # Positivo = hallazgo maligno
            confidence=round(confianza, 4),
            raw_prediction=round(prob, 4),
            recommendation=(
                "Tejido dentro de los parámetros normales."
                if es_benigno
                else "Sospecha de malignidad. Se recomienda estudio histopatológico completo."
            ),
            heatmap=heatmap,
        )
