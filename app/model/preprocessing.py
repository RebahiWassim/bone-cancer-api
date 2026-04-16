import io
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import torch
import torch.nn.functional as F
from typing import Optional

def get_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

def preprocess(image_bytes: bytes, img_size: int):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    vis_arr = np.array(img.resize((img_size, img_size))).copy()
    transform = get_transform(img_size)
    return transform(img).unsqueeze(0), vis_arr

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.backbone.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        tensor = tensor.to(next(self.model.parameters()).device)
        tensor.requires_grad_(True)
        
        output = self.model(tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam).cpu().numpy()
        
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

def heatmap_to_bbox(cam: np.ndarray, threshold: float = 0.4, img_size: int = 224) -> Optional[tuple]:
    binary = (cam >= threshold).astype(np.uint8) * 255
    resized = cv2.resize(binary, (img_size, img_size))
    contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h)

def draw_bbox_on_image(vis_arr: np.ndarray, bbox: tuple) -> np.ndarray:
    img_pil = Image.fromarray(vis_arr.astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    x0, y0, x1, y1 = bbox
    for offset in range(3):
        draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                      outline=(255, 0, 0))
    return np.array(img_pil)