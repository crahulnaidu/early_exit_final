import torch
import torch.nn as nn
from models.backbone import FEATURE_SHAPES
from models.exit_heads import HEAD_BUILDERS
from utils.freeze import BACKBONE_PARTS, freeze_backbone

# Section 2.1(a) Candidate Positions
CANDIDATE_POSITIONS = [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 6, 13, 23, 33]
CANDIDATE_POSITIONS = sorted(list(set(CANDIDATE_POSITIONS)))


def get_channel_count_at_layer(layer_idx):
    """Channels after features[layer_idx], from the analytic shape table (no forward pass)."""
    return FEATURE_SHAPES[layer_idx][0]


class SupernetMultiExitVGG(nn.Module):
    """
    Supernet containing ALL 85 candidate heads (17 positions x 5 archs).
    The backbone is kept completely frozen.
    """
    def __init__(self, features, avgpool, classifier, candidate_positions=None):
        super().__init__()
        self.features = features
        self.avgpool = avgpool
        self.classifier = classifier
        self.candidate_positions = candidate_positions or CANDIDATE_POSITIONS

        # Freeze backbone, avgpool, and classifier (requires_grad=False, eval mode)
        freeze_backbone(self)

        # Instantiate 85 candidate heads
        self.heads = nn.ModuleDict()
        self.head_keys = []

        for pos in self.candidate_positions:
            in_c = get_channel_count_at_layer(pos)
            for arch_name, builder_cls in HEAD_BUILDERS.items():
                key = f"pos_{pos}_arch_{arch_name}"
                self.heads[key] = builder_cls(in_channels=in_c, num_classes=10)
                self.head_keys.append(key)

    def train(self, mode=True):
        """Heads follow `mode`; the frozen backbone always stays in eval mode (BN statistics fixed)."""
        super().train(mode)
        for part in BACKBONE_PARTS:
            getattr(self, part).eval()
        return self

    def forward(self, x):
        outputs = {}
        curr_x = x

        with torch.no_grad():
            for idx, layer in enumerate(self.features):
                curr_x = layer(curr_x)
                if idx in self.candidate_positions:
                    # Pass feature map to all 5 heads at this position
                    outputs[f"pos_{idx}"] = curr_x

            # Final classifier logits: apply avgpool -> flatten -> classifier
            pooled_x = self.avgpool(curr_x)
            flat_x = torch.flatten(pooled_x, 1)
            outputs["final"] = self.classifier(flat_x)

        # Compute forward pass for active heads outside no_grad context
        logits_dict = {}
        for key, feat in outputs.items():
            if key == "final":
                logits_dict[key] = feat
            else:
                pos = int(key.split("_")[1])
                for arch_name in HEAD_BUILDERS.keys():
                    head_key = f"pos_{pos}_arch_{arch_name}"
                    logits_dict[head_key] = self.heads[head_key](feat)

        return logits_dict
