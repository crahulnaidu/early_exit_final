import pytest
import torch.optim as optim

from utils.freeze import (assert_backbone_frozen, assert_optimizer_heads_only, backbone_parameters,
                          head_parameters)


def test_backbone_frozen_and_stays_eval_in_train_mode(ee_model):
    ee_model.train()
    assert_backbone_frozen(ee_model)
    assert all(h.training for h in ee_model.exit_heads)
    assert all(not p.requires_grad for p in backbone_parameters(ee_model))


def test_optimizer_heads_only(ee_model):
    opt = optim.Adam(head_parameters(ee_model), lr=1e-3)
    assert_optimizer_heads_only(opt, ee_model)
    bad = optim.Adam(head_parameters(ee_model) + backbone_parameters(ee_model)[:1], lr=1e-3)
    with pytest.raises(AssertionError):
        assert_optimizer_heads_only(bad, ee_model)


def test_backbone_unchanged_after_head_training_step(ee_model):
    import torch
    import torch.nn.functional as F
    before = {k: v.clone() for k, v in ee_model.state_dict().items() if k.split(".")[0] in ("features", "classifier")}
    ee_model.train()
    opt = optim.SGD(head_parameters(ee_model), lr=0.1)
    x, y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])
    loss = sum(F.cross_entropy(z, y) for z in ee_model(x)[:-1])
    loss.backward()
    opt.step()
    after = ee_model.state_dict()
    for k, v in before.items():          # includes BN running stats: they must not move either
        assert torch.equal(v, after[k]), k
