import torch
import torch.nn as nn
from models.exit_heads import HEAD_BUILDERS

# Section 2.1(a) Candidate Positions
CANDIDATE_POSITIONS = [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 6, 13, 23, 33]
CANDIDATE_POSITIONS = sorted(list(set(CANDIDATE_POSITIONS)))

def get_channel_count_at_layer(features, layer_idx):
    dummy = torch.zeros(1, 3, 32, 32)
    x = dummy
    for idx, layer in enumerate(features):
        x = layer(x)
        if idx == layer_idx:
            return x.shape[1]
    return x.shape[1]


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

        # Freeze backbone, avgpool, and classifier
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.features.eval()
        self.avgpool.eval()
        self.classifier.eval()

        # Instantiate 85 candidate heads
        self.heads = nn.ModuleDict()
        self.head_keys = []

        for pos in self.candidate_positions:
            in_c = get_channel_count_at_layer(self.features, pos)
            for arch_name, builder_cls in HEAD_BUILDERS.items():
                key = f"pos_{pos}_arch_{arch_name}"
                self.heads[key] = builder_cls(in_channels=in_c, num_classes=10)
                self.head_keys.append(key)

    def forward(self, x):
        # Backbone is frozen in eval mode
        self.features.eval()
        self.avgpool.eval()
        self.classifier.eval()

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
