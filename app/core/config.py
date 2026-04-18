import os
import torch

class Settings:
    # Chemins
    MODEL_PATH = os.getenv("MODEL_PATH", "model/bone_cancer_model.pth")

    extra = "ignore"
    
    # Paramètres du modèle
    IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
    MC_PASSES = int(os.getenv("MC_PASSES", 10))
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Classes
    CLASS_NAMES = ["Cancer", "Normal"]
    
    # API
    API_TITLE = "Bone Cancer Detector"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "EfficientNet-B0 classifier for bone cancer detection"

settings = Settings()