import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbone import build_backbone  # noqa: E402
from models.searched_exit_model import SearchedMultiExitVGG  # noqa: E402
from utils.seed import set_seed  # noqa: E402


@pytest.fixture
def ee_model():
    """Random-weight multi-exit model with the validation configuration (16/A4, 23/A3), CPU, eval mode."""
    set_seed(0)
    b = build_backbone()
    m = SearchedMultiExitVGG(b.features, b.avgpool, b.classifier, [16, 23], ["A4", "A3"]).eval()
    for h in m.exit_heads:          # give heads non-trivial BN statistics
        for mod in h.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.1, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
    return m
