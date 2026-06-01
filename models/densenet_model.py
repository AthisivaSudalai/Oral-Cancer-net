import torch
import torch.nn as nn
from torchvision import models


def build_densenet121(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)

    # DenseNet's classifier is a single Linear layer.
    # Replace it to match our number of classes.
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model