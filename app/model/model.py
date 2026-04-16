import torch
import torch.nn as nn
from torchvision import models
import logging

logger = logging.getLogger(__name__)

class BoneCancerModel(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

def load_model(path: str, device: torch.device):
    m = BoneCancerModel(num_classes=2)
    checkpoint = torch.load(path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()
    logger.info(f"Model loaded on {device}")
    return m