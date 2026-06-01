import torch
import torch.nn as nn
from torchvision import models


def build_vit(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)

    # ViT's classifier head is a single Linear layer at model.heads.head
    # Replace it to match our number of classes.
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model