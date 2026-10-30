import pytest
import torch
import torch.nn as nn

from config import WORK_DIR
from utils.ckpt import load_checkpoint, read_checkpoint_meta, save_checkpoint
from utils.paths import safe_path


def small():
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "does_not_exist.pth", small())
    with pytest.raises(FileNotFoundError):
        read_checkpoint_meta(tmp_path / "does_not_exist.pth")


def test_roundtrip_with_meta(tmp_path):
    a, b = small(), small()
    save_checkpoint(tmp_path / "a.pth", a, {"seed": 7})
    meta = load_checkpoint(tmp_path / "a.pth", b)
    assert meta["seed"] == 7
    for x, y in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(x, y)


def test_strict_missing_key_raises(tmp_path):
    state = small().state_dict()
    del state["2.bias"]
    torch.save(state, tmp_path / "missing.pth")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(tmp_path / "missing.pth", small())


def test_strict_unexpected_key_raises(tmp_path):
    state = small().state_dict()
    state["extra.weight"] = torch.zeros(1)
    torch.save(state, tmp_path / "extra.pth")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_checkpoint(tmp_path / "extra.pth", small())


def test_strict_shape_mismatch_raises(tmp_path):
    other = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))
    torch.save(other.state_dict(), tmp_path / "shape.pth")
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(tmp_path / "shape.pth", small())


def test_safe_path_refuses_outside_work_folder():
    with pytest.raises(AssertionError):
        safe_path("/home1/jtt/early_exit_new/checkpoints/x.pth")
    with pytest.raises(AssertionError):
        safe_path(str(WORK_DIR) + "/../early_exit_new/x.pth")
    assert str(safe_path(WORK_DIR / "tmp" / "ok.txt")).startswith(str(WORK_DIR))


def test_no_exists_fallbacks_in_scripts():
    """No script may silently continue when a checkpoint is missing (old `if Path(...).exists()` pattern)."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for path in (root / "scripts").glob("*.py"):
        src = path.read_text()
        assert "strict=False" not in src, f"{path.name} uses strict=False"
        assert not re.search(r"if Path\(.*\)\.exists\(\):", src), f"{path.name} has an exists()-fallback"
        assert "torch.load(" not in src or path.name in ("04_search.py",), f"{path.name} bypasses load_checkpoint"


def test_writes_under_source_project_are_refused(tmp_path):
    """early_exit_new_3 may read the source project but never write into it."""
    from config import SRC_DIR
    with pytest.raises(AssertionError):
        safe_path(SRC_DIR / "outputs" / "x.json")
    with pytest.raises(AssertionError):
        safe_path(SRC_DIR / "checkpoints" / "new.pth")
    with pytest.raises(AssertionError):
        save_checkpoint(SRC_DIR / "checkpoints" / "x.pth", small(), {})


def test_ckpt_path_reads_source_checkpoint_but_cannot_overwrite_it():
    from config import BACKBONE_CKPT, SRC_DIR
    from utils.runmeta import ckpt_path
    p = ckpt_path("vgg16_bn_cifar10_base.pth", False)
    assert p == BACKBONE_CKPT and str(p).startswith(str(SRC_DIR))
    with pytest.raises(AssertionError):
        save_checkpoint(p, small(), {})
    assert str(ckpt_path("brand_new.pth", False)).startswith(str(WORK_DIR))


def test_safe_path_scratch_mirror_only():
    """Large data lives in /scratch/jtt/early_exit_final/<stage>/ (linked back); only this stage's mirror is writable."""
    from config import SRC_DIR
    from utils.paths import SCRATCH_ROOT
    assert SCRATCH_ROOT == "/scratch/jtt/early_exit_final/stage2_paper/"
    assert str(safe_path(SCRATCH_ROOT + "tmp/ok.txt")).startswith(SCRATCH_ROOT)
    with pytest.raises(AssertionError):
        safe_path("/scratch/jtt/early_exit_final/stage1_base_nas/cache/x.npy")
    with pytest.raises(AssertionError):
        safe_path("/scratch/jtt/x.npy")
    with pytest.raises(AssertionError):
        safe_path(SRC_DIR / "cache" / "x.npy")


def test_current_path_maps_premerge_folders():
    from utils.paths import current_path
    assert str(current_path("/home1/jtt/early_exit_new_2/checkpoints/ee_nas.pth")) == \
        "/home1/jtt/early_exit_final/stage1_base_nas/checkpoints/ee_nas.pth"
    assert str(current_path("/home1/jtt/early_exit_new_3/checkpoints/a.pth")) == \
        "/home1/jtt/early_exit_final/stage2_paper/checkpoints/a.pth"
    assert str(current_path("/home1/jtt/early_exit_final/x.pth")) == "/home1/jtt/early_exit_final/x.pth"
    assert current_path("/home1/jtt/early_exit_new_2/checkpoints/ee_nas.pth").exists()
