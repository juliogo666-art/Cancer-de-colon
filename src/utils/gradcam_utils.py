#########################################################################################

# GRAD-CAM

# Mapa de calor para visualizar las zonas de interés de la imagen
# Si la muestra es sana: El modelo se ilumina en rojo en las estructuras glandulares
# Si la muestra es cancerosa: Como el modelo "busca cordura" y solo ve caos
# (células rotas, núcleos gigantes), el mapa de calor se apaga o se dispersa
#########################################################################################

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import tensorflow as tf

import shap
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def generate_gradcam(model, img_pil, target_layer):
    """
    Genera un mapa de calor Grad-CAM sobre la imagen PIL usando el modelo PyTorch.

    Args:
        model: El modelo entrenado (ej. ResNet18)
        img_pil: Imagen original cargada con PIL
        target_layer: La capa convolucional final de la que extraer las activaciones (ej. model.model.layer4[-1])
    """
    model.eval()

    # 1. Preprocesamiento (idéntico al de inferencia)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = transform(img_pil).unsqueeze(0)  # [1, 3, 224, 224]

    # --- HOOKS PARA EXTRAER GRADIENTES Y ACTIVACIONES ---
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        # Guardamos el gradiente de la salida de la capa
        gradients.append(grad_output[0])

    def forward_hook(module, args, output):
        # Guardamos la activación de la capa
        activations.append(output)

    # Enganchamos los hooks a la capa objetivo
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # 2. Forward pass
    output = model(input_tensor)

    # Obtenemos la probabilidad con Sigmoide
    prob = torch.sigmoid(output).item()

    # 3. Backward pass
    # Como es clasificación binaria (1 salida), hacemos backward sobre el valor de salida directo
    model.zero_grad()
    output.backward()

    # 4. Procesamiento de Grad-CAM
    # Cogemos el primer elemento del batch
    grads = gradients[0][0]  # [Canales, H_activacion, W_activacion]
    acts = activations[0][0]  # [Canales, H_activacion, W_activacion]

    # Global Average Pooling de los gradientes para obtener los pesos de importancia de canales
    weights = torch.mean(grads, dim=(1, 2), keepdim=True)  # [Canales, 1, 1]

    # Suma ponderada de las activaciones
    cam = torch.sum(weights * acts, dim=0)  # [H_activacion, W_activacion]

    # Aplicar ReLU (solo nos interesan las características que aportan POSITIVAMENTE a la decisión)
    cam = F.relu(cam)

    # Normalizar entre 0 y 1
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.detach().cpu().numpy()

    # --- SUPERPOSICIÓN VISUAL (OPENCV) ---
    # Convertir PIL a array OpenCV (BGR) para montar el mapa de calor
    img_cv = np.array(img_pil.resize((224, 224)))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Redimensionar el mapa de calor de activación al tamaño de la imagen (224x224)
    cam_resized = cv2.resize(cam, (224, 224))

    # Aplicar mapa de color JET (Azul -> Verde -> Rojo)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    # Superponer: Imagen Original + Heatmap (con un peso Alpha de transparencia)
    # img_cv es uint8, heatmap es uint8
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    # Convertir de nuevo a RGB para que Streamlit o PIL lo lean bien
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Quitar los hooks para no dejar basura en memoria
    h1.remove()
    h2.remove()

    return superimposed_img_rgb, prob

def generate_gradcam_colon(model, img_array):
    """
    Versión blindada contra errores de 'output_shape' en Keras 3.
    """
    # 1. Preprocesamiento
    img_res = cv2.resize(img_array, (150, 150))
    img_batch = np.expand_dims(img_res, axis=0).astype(np.float32)
    img_pre = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)

    # 2. Identificar la última capa convolucional (4D) de forma segura
    last_conv_layer_name = None
    # Recorremos a la inversa buscando la última capa con 4 dimensiones (Batch, H, W, Channels)
    for layer in reversed(model.layers):
        try:
            # Usamos layer.output.shape que es más fiable que layer.output_shape
            shape = layer.output.shape
            if len(shape) == 4:
                last_conv_layer_name = layer.name
                break
        except:
            continue

    if not last_conv_layer_name:
        # Si falla la detección automática, intentamos nombres comunes de MobileNetV2
        for fallback in ["out_relu", "post_relu", "Conv_1"]:
            try:
                model.get_layer(fallback)
                last_conv_layer_name = fallback
                break
            except:
                continue

    if not last_conv_layer_name:
        return img_res, 0.0

    # 3. Construir el modelo de gradientes
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_pre)
        pred_val = preds[0][0]
        
        # Clase 0 es Pólipo (según tu lógica < 0.5)
        # Si es pólipo, queremos ver qué activó esa probabilidad baja
        if pred_val < 0.5:
            target_score = 1.0 - preds[:, 0]
        else:
            target_score = preds[:, 0]

    # 4. Cálculo de Gradientes
    grads = tape.gradient(target_score, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Generar Mapa de Calor
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU y Normalización
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # 6. Post-procesado visual
    heatmap_resized = cv2.resize(heatmap, (150, 150))
    # Limpiamos ruido: solo mostramos el 30% superior de intensidad
    heatmap_resized[heatmap_resized < 0.3] = 0 
    
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Superposición
    img_cv = img_res.copy()
    if img_cv.max() <= 1.0: img_cv = (img_cv * 255).astype(np.uint8)
    
    superimposed = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    return superimposed_rgb, float(pred_val)
    
def generar_explicacion_shap(modelo, features_array, target_class):
    try:
        nombres_columnas = [
            'Smoking', 'Alcohol_Use', 'Obesity', 'Family_History', 
            'Diet_Red_Meat', 'Diet_Salted_Processed', 'Fruit_Veg_Intake', 
            'Physical_Activity', 'BMI', 'FOBT_Resultado_n', 'CEA_Level_ng_mL'
        ]
        X_df = pd.DataFrame(features_array, columns=nombres_columnas)

        # Si nos han pasado una lista de modelos (ensamble), calculamos SHAP
        # para cada modelo y promediamos los valores. Esto evita errores
        # al pasar directamente una lista a TreeExplainer.
        def _extract_raw_shap(shap_values_obj, target_cls):
            if isinstance(shap_values_obj, list):
                try:
                    return np.array(shap_values_obj[target_cls][0])
                except Exception:
                    # Fallback conservador
                    return np.array(shap_values_obj[0][0])
            else:
                sv = np.array(shap_values_obj)
                if sv.ndim == 3:
                    return sv[0, :, target_cls]
                elif sv.ndim == 2:
                    return sv[0]
                else:
                    return sv.flatten()

        if isinstance(modelo, (list, tuple)):
            accum = None
            n_ok = 0
            for m in modelo:
                try:
                    expl = shap.TreeExplainer(m)
                    sv = expl.shap_values(X_df)
                    raw = _extract_raw_shap(sv, target_class)
                    if accum is None:
                        accum = np.array(raw, dtype=float)
                    else:
                        accum += np.array(raw, dtype=float)
                    n_ok += 1
                except Exception as e:
                    print(f"Error SHAP por modelo del ensemble: {e}")
                    continue

            if n_ok == 0:
                raise RuntimeError("Ningún modelo del ensemble pudo generar SHAP")

            raw_shap = accum / n_ok
        else:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(X_df)
            raw_shap = _extract_raw_shap(shap_values, target_class)

        # 2. Calcular porcentaje de influencia
        # Usamos el valor absoluto para entender cuánto "pesa" cada variable
        abs_shap = np.abs(raw_shap)
        total_impact = np.sum(abs_shap)
        
        influencia = []
        for i in range(len(nombres_columnas)):
            porcentaje = (abs_shap[i] / total_impact * 100) if total_impact > 0 else 0
            influencia.append({
                "Variable": nombres_columnas[i],
                "Impacto": f"{porcentaje:.1f}%",
                "Sentido": "Sube riesgo" if raw_shap[i] > 0 else "Baja riesgo"
            })
        
        df_importancia = pd.DataFrame(influencia).sort_values(by="Impacto", ascending=False)

        # 3. Crear Gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.gcf().set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        
        shap.bar_plot(raw_shap, feature_names=nombres_columnas, max_display=11, show=False)
        plt.title("Factores que definen tu perfil de riesgo", color="white")
        
        return fig, df_importancia
        
    except Exception as e:
        print(f"Error SHAP: {e}")
        return None, None