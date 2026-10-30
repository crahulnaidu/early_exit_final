"""VGG16-BN backbone with a CIFAR-style classifier (agreed in Step 0, used by every script).

features (torchvision VGG16-BN layout, 44 modules) -> AdaptiveAvgPool2d(1) -> flatten (512)
-> Dropout(0.5) -> Linear(512, 512) -> ReLU -> Linear(512, 10)
Weight init follows torchvision's VGG (kaiming conv, BN 1/0, Linear N(0, 0.01)), as the old 01 used.
"""
import torch
import torch.nn as nn
from torchvision.models.vgg import cfgs, make_layers

VGG16_CFG = cfgs["D"]
INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 10


class VGG16BNCifar(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.features = make_layers(VGG16_CFG, batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_backbone():
    return VGG16BNCifar()


def _layer_specs():
    """Analytic per-layer table for `features`: (type, C_in, C_out, H_in, W_in, H_out, W_out)."""
    specs = []
    c, h, w = INPUT_SHAPE
    for v in VGG16_CFG:
        if v == "M":
            specs.append(("maxpool", c, c, h, w, h // 2, w // 2))
            h, w = h // 2, w // 2
        else:
            specs.append(("conv", c, v, h, w, h, w))
            specs.append(("bn", v, v, h, w, h, w))
            specs.append(("relu", v, v, h, w, h, w))
            c = v
    return specs


LAYER_SPECS = _layer_specs()
# Output shape (C, H, W) after features[idx]; no forward pass needed.
FEATURE_SHAPES = {i: (s[2], s[5], s[6]) for i, s in enumerate(LAYER_SPECS)}
