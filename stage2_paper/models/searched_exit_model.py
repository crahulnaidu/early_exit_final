import torch
import torch.nn as nn
from models.backbone import build_backbone
from models.head_builder import make_head
from utils.ckpt import load_checkpoint, read_checkpoint_meta
from utils.freeze import BACKBONE_PARTS, assert_backbone_frozen, freeze_backbone
from utils.paths import current_path


class SearchedMultiExitVGG(nn.Module):
    """
    Instantiates ONLY the chosen exit heads at their selected positions on the frozen VGG-16 backbone.
    """
    def __init__(self, features, avgpool, classifier, positions, archs):
        super().__init__()
        self.features = features
        self.avgpool = avgpool
        self.classifier = classifier
        self.positions = list(positions)
        self.archs = list(archs)
        assert all(a < b for a, b in zip(self.positions, self.positions[1:])), \
            f"positions must be strictly increasing: {self.positions}"

        # Freeze backbone, avgpool, and classifier (requires_grad=False, eval mode)
        freeze_backbone(self)

        # Instantiate chosen exit heads with trainable parameters
        self.exit_heads = nn.ModuleList()
        for pos, arch in zip(self.positions, self.archs):
            self.exit_heads.append(make_head(pos, arch))   # menu name ("A1".."A5") or NAS cfg dict

    def train(self, mode=True):
        """Heads follow `mode`; the frozen backbone always stays in eval mode (BN statistics fixed)."""
        super().train(mode)
        for part in BACKBONE_PARTS:
            getattr(self, part).eval()
        return self

    def forward(self, x):
        logits_list = []
        curr_x = x

        # Run feature extractor under no_grad to save memory and keep backbone frozen
        head_idx = 0
        for idx, layer in enumerate(self.features):
            with torch.no_grad():
                curr_x = layer(curr_x)

            if head_idx < len(self.positions) and idx == self.positions[head_idx]:
                # Forward feature map through the trainable exit head (GRADIENTS ENABLED)
                head_logits = self.exit_heads[head_idx](curr_x)
                logits_list.append(head_logits)
                head_idx += 1

        # Final classifier logits under no_grad
        with torch.no_grad():
            pooled_x = self.avgpool(curr_x)
            flat_x = torch.flatten(pooled_x, 1)
            final_logits = self.classifier(flat_x)

        # Append final teacher logits as the last entry (K + 1)
        logits_list.append(final_logits)
        return logits_list


class MultiProbeVGG(nn.Module):
    """Several INDEPENDENT heads on the frozen backbone, positions may repeat (proxy check: train 5 configs in one
    pass). Because the backbone is frozen and each head has its own loss, this equals training them separately."""
    def __init__(self, features, avgpool, classifier, positions, archs):
        super().__init__()
        self.features, self.avgpool, self.classifier = features, avgpool, classifier
        self.positions, self.archs = list(positions), list(archs)
        freeze_backbone(self)
        self.exit_heads = nn.ModuleList([make_head(p, a) for p, a in zip(self.positions, self.archs)])

    def train(self, mode=True):
        super().train(mode)
        for part in BACKBONE_PARTS:
            getattr(self, part).eval()
        return self

    def forward(self, x):
        out = [None] * len(self.positions)
        for idx, layer in enumerate(self.features):
            with torch.no_grad():
                x = layer(x)
            for j, p in enumerate(self.positions):
                if p == idx:
                    out[j] = self.exit_heads[j](x)
        with torch.no_grad():
            final = self.classifier(torch.flatten(self.avgpool(x), 1))
        return out + [final]


def build_searched_model(base_ckpt, positions, archs):
    """New multi-exit model: frozen backbone loaded (strict) from base_ckpt, freshly initialised heads."""
    backbone = build_backbone()
    load_checkpoint(base_ckpt, backbone)
    model = SearchedMultiExitVGG(backbone.features, backbone.avgpool, backbone.classifier, positions, archs)
    assert_backbone_frozen(model)
    return model


def load_searched_model(base_ckpt, ee_ckpt):
    """Trained multi-exit model. Positions/archs come from the ee checkpoint's metadata; the backbone copy
    stored in the ee checkpoint must be bit-identical to base_ckpt (frozen backbone guarantee)."""
    ee_ckpt = current_path(ee_ckpt)
    meta = read_checkpoint_meta(ee_ckpt)
    model = build_searched_model(base_ckpt, meta["positions"], meta["archs"])
    base_state = {k: v.clone() for k, v in model.state_dict().items()
                  if k.split(".")[0] in BACKBONE_PARTS}
    load_checkpoint(ee_ckpt, model)
    for k, v in base_state.items():
        assert torch.equal(model.state_dict()[k], v), f"backbone tensor {k} in {ee_ckpt} differs from {base_ckpt}"
    assert_backbone_frozen(model)
    return model, meta
