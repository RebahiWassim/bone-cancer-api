import io
import base64
import numpy as np
import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.model.model import load_model
from app.model.preprocessing import preprocess, GradCAM, heatmap_to_bbox, draw_bbox_on_image

router = APIRouter()
_model = None

PREDICTION_TYPE = "1D"

HIGH_RISK_CLASS_INDEX = 0


def init_model():
    global _model
    _model = load_model(settings.MODEL_PATH, settings.DEVICE)
    return _model


def encode_image_base64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_uncertainty_score(probs: np.ndarray) -> float:
    """
    Compute an uncertainty score using the entropy of the probability distribution.
    Normalized to [0, 1] by dividing by log(num_classes).
    A score close to 1.0 means maximum uncertainty; close to 0.0 means very confident.
    """
    num_classes = len(probs)
    # Clip to avoid log(0)
    clipped = np.clip(probs, 1e-9, 1.0)
    entropy = -np.sum(clipped * np.log(clipped))
    max_entropy = np.log(num_classes)
    return round(float(entropy / max_entropy), 4)


ALLOWED_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/jpg"}


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    try:
        contents = await file.read()
        tensor, vis_arr = preprocess(contents, settings.IMG_SIZE)

        tensor = tensor.to(settings.DEVICE)
        _model.eval()
        with torch.no_grad():
            probs = torch.softmax(_model(tensor), dim=1)[0].cpu().numpy()

        class_index = int(np.argmax(probs))
        pred_class = settings.CLASS_NAMES[class_index]
        confidence = round(float(probs[class_index]), 4)
        uncertainty_score = compute_uncertainty_score(probs)
        is_high_risk = class_index == HIGH_RISK_CLASS_INDEX

        original_image_b64 = encode_image_base64(vis_arr)

        if pred_class == "Cancer":
            try:
                cam = GradCAM(_model).generate(tensor.clone(), class_idx=0)
                bbox = heatmap_to_bbox(cam, threshold=0.4, img_size=settings.IMG_SIZE)
                if bbox is not None:
                    annotated = draw_bbox_on_image(vis_arr.copy(), bbox)
                    original_image_b64 = encode_image_base64(annotated)
            except Exception:
                pass 

        result = {
            "status": "success",
            "type": PREDICTION_TYPE,
            "prediction": pred_class,
            "class_index": class_index,
            "confidence": confidence,
            "diagnostics": {
                "uncertainty_score": uncertainty_score,
                "is_high_risk": is_high_risk,
                "all_probabilities": {
                    cls: round(float(p), 4)
                    for cls, p in zip(settings.CLASS_NAMES, probs)
                },
            },
            "original_image": original_image_b64,
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/mc")
async def predict_mc(file: UploadFile = File(...), n_passes: int = 10):
    
    pass