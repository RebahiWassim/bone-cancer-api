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

def init_model():
    global _model
    _model = load_model(settings.MODEL_PATH, settings.DEVICE)
    return _model

def encode_image_base64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/jpg"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported type")
    
    try:
        contents = await file.read()
        tensor, vis_arr = preprocess(contents, settings.IMG_SIZE)
        
        # Prédiction
        tensor = tensor.to(settings.DEVICE)
        _model.eval()
        with torch.no_grad():
            probs = torch.softmax(_model(tensor), dim=1)[0].cpu().numpy()
        
        idx = int(np.argmax(probs))
        pred_class = settings.CLASS_NAMES[idx]
        
        # Visualisation
        image_b64 = encode_image_base64(vis_arr)
        bbox_dict = None
        
        if pred_class == "Cancer":
            try:
                cam = GradCAM(_model).generate(tensor.clone(), class_idx=0)
                bbox = heatmap_to_bbox(cam, threshold=0.4, img_size=settings.IMG_SIZE)
                if bbox is not None:
                    annotated = draw_bbox_on_image(vis_arr.copy(), bbox)
                    image_b64 = encode_image_base64(annotated)
                    bbox_dict = {"x_min": bbox[0], "y_min": bbox[1],
                               "x_max": bbox[2], "y_max": bbox[3]}
            except Exception as e:
                pass
        
        result = {
            "classe": pred_class,
            "confiance": round(float(probs[idx]), 4),
            "probabilites": {
                cls: round(float(p), 4)
                for cls, p in zip(settings.CLASS_NAMES, probs)
            },
            "image_annotee": image_b64,
            "bounding_box": bbox_dict,
            "filename": file.filename
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/mc")
async def predict_mc(file: UploadFile = File(...), n_passes: int = 10):
    # Implémentation MC Dropout...
    pass