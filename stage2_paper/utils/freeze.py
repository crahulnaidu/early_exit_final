"""Frozen-backbone helpers: the backbone (features, avgpool, classifier) is never trained after Step 1b."""
BACKBONE_PARTS = ("features", "avgpool", "classifier")


def backbone_parameters(model):
    return [p for part in BACKBONE_PARTS for p in getattr(model, part).parameters()]


def head_parameters(model):
    backbone_ids = {id(p) for p in backbone_parameters(model)}
    return [p for p in model.parameters() if id(p) not in backbone_ids]


def freeze_backbone(model):
    for part in BACKBONE_PARTS:
        module = getattr(model, part)
        for p in module.parameters():
            p.requires_grad = False
        module.eval()


def assert_backbone_frozen(model):
    for part in BACKBONE_PARTS:
        module = getattr(model, part)
        assert all(not p.requires_grad for p in module.parameters()), f"{part} has trainable parameters"
        assert all(not m.training for m in module.modules()), f"{part} is not in eval mode"


def assert_optimizer_heads_only(optimizer, model):
    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    head_ids = {id(p) for p in head_parameters(model)}
    assert opt_ids == head_ids, (
        f"optimizer holds {len(opt_ids)} tensors, heads have {len(head_ids)}; "
        f"{len(opt_ids - head_ids)} non-head tensors in the optimizer"
    )
    assert all(p.requires_grad for p in head_parameters(model)), "some head parameters are frozen"
