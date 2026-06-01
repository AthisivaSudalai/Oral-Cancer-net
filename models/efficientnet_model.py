import torch
import torch.nn as nn
from torchvision import models


def build_efficientnet_b2(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b2(weights=weights)

    # EfficientNet's classifier is a Sequential(Dropout, Linear).
    # We replace the Linear layer to match our number of classes.
    # The dropout is kept — EfficientNet was designed with it.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model