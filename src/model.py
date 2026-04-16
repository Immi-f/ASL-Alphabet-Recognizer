"""Transfer learning models for ASL classification."""

import torch.nn as nn
from torchvision import models

AVAILABLE_MODELS = ("mobilenet_v3_small", "resnet50")


def build_model(num_classes: int = 29, pretrained: bool = True,
                arch: str = "mobilenet_v3_small") -> nn.Module:
    """Return a backbone with a replaced classifier head."""
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes),
        )
        return model

    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unknown arch '{arch}'. Choose from {AVAILABLE_MODELS}.")


def _head_param_prefix(model: nn.Module) -> str:
    # ResNet uses `fc`, MobileNetV3 uses `classifier`.
    return "fc" if hasattr(model, "fc") and isinstance(model.fc, nn.Sequential) else "classifier"


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    head = _head_param_prefix(model)
    for name, param in model.named_parameters():
        if not name.startswith(head):
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
