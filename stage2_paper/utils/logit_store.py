"""Reader for the stored clean-logit files of 40_campaign_store.py (data files, not model checkpoints)."""
import torch


def load_clean_logits(path):
    """{"logits": [N, K+1, C], "labels", "positions", "archs", "rule", "ee_ckpt", "int8"} on the CPU."""
    return torch.load(path, map_location="cpu")
