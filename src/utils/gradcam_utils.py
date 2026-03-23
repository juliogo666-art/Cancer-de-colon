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
