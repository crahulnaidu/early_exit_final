import torch
import torch.nn as nn
from models.exit_heads import HEAD_BUILDERS
from models.multi_exit import get_channel_count_at_layer

class SearchedMultiExitVGG(nn.Module):
    """
    Instantiates ONLY the chosen exit heads at their selected positions on the frozen VGG-16 backbone.
    """
    def __init__(self, features, avgpool, classifier, positions, archs):
        super().__init__()
        self.features = features
        self.avgpool = avgpool
        self.classifier = classifier
        self.positions = positions
        self.archs = archs

        # Freeze backbone, avgpool, and classifier parameters
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.features.eval()
        self.avgpool.eval()
        self.classifier.eval()

        # Instantiate chosen exit heads with trainable parameters
        self.exit_heads = nn.ModuleList()
        for pos, arch_name in zip(positions, archs):
            in_c = get_channel_count_at_layer(self.features, pos)
            builder_cls = HEAD_BUILDERS[arch_name]
            self.exit_heads.append(builder_cls(in_channels=in_c, num_classes=10))

    def forward(self, x):
        self.features.eval()
        self.avgpool.eval()
        self.classifier.eval()

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
