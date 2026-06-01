import torch
import torch.nn as nn
from torchvision import models


def build_swin_t(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.swin_t(weights=weights)

    # Swin-T's classifier head is at model.head, a single Linear layer.
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model