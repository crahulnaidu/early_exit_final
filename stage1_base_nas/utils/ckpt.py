from pathlib import Path

import torch

from utils.paths import safe_path


def _read(path, map_location="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Run the script that produces it first; refusing to continue with randomly initialised weights."
        )
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("meta", {})
    return obj, {}


def load_checkpoint(path, model, map_location="cpu"):
    """Load a checkpoint into `model` with strict=True. Returns its metadata dict ({} for plain state dicts).

    Raises FileNotFoundError if the file is missing - there is no fallback to random weights.
    Accepts both {"state_dict": ..., "meta": ...} (written by save_checkpoint) and a plain state dict.
    """
    state, meta = _read(path, map_location)
    model.load_state_dict(state, strict=True)
    return meta


def read_checkpoint_meta(path):
    """Metadata of a checkpoint without loading it into a model (FileNotFoundError if missing)."""
    return _read(path)[1]


def save_checkpoint(path, model, meta):
    """Save {"state_dict", "meta"}; the path must be inside the work folder."""
    p = safe_path(path)
    torch.save({"state_dict": model.state_dict(), "meta": dict(meta)}, p)
    return p
