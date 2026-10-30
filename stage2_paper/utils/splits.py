"""Fixed 45k/5k train/val split of the CIFAR-10 training set, stored in outputs/splits.json.

The indices are the same as in the source project (np.random.RandomState(seed).shuffle of range(50000),
first 45k = train, last 5k = val), so they match the checkpoints trained there.
Create the file once with scripts/00_make_splits.py; every loader then reads and verifies it.
"""
import hashlib
import json

import numpy as np

from config import OUTPUT_DIR
from utils.paths import safe_path

SPLITS_PATH = OUTPUT_DIR / "splits.json"
SPLIT_SEED = 42  # fixed: the split never depends on the run seed
N_TRAIN_FULL = 50000
N_TRAIN = 45000
N_VAL = 5000


def compute_split(seed):
    indices = list(range(N_TRAIN_FULL))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    return indices[:N_TRAIN], indices[N_TRAIN:]


def split_hash(train_idx, val_idx):
    payload = json.dumps({"train": list(map(int, train_idx)), "val": list(map(int, val_idx))})
    return hashlib.sha256(payload.encode()).hexdigest()


def create_splits(seed=SPLIT_SEED, path=SPLITS_PATH):
    """Write splits.json (or verify an existing one). Never overwrites a file with different content."""
    train_idx, val_idx = compute_split(seed)
    data = {
        "seed": seed,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "sha256": split_hash(train_idx, val_idx),
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
    }
    if path.exists():
        existing = load_splits(seed, path)
        assert existing["sha256"] == data["sha256"], f"{path} exists with different content"
        return existing
    with open(safe_path(path), "w") as f:
        json.dump(data, f)
    return data


def load_splits(seed=SPLIT_SEED, path=SPLITS_PATH):
    """Read splits.json and assert it is internally consistent and equals the seed-derived split."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run scripts/00_make_splits.py first.")
    with open(path) as f:
        data = json.load(f)
    train_idx, val_idx = data["train_idx"], data["val_idx"]
    assert data["seed"] == seed, f"splits.json was made with seed {data['seed']}, requested {seed}"
    assert len(train_idx) == N_TRAIN and len(val_idx) == N_VAL
    assert not set(train_idx) & set(val_idx), "train/val overlap"
    assert sorted(train_idx + val_idx) == list(range(N_TRAIN_FULL)), "split does not cover the training set"
    assert split_hash(train_idx, val_idx) == data["sha256"], "splits.json content does not match its hash"
    ref_train, ref_val = compute_split(seed)
    assert train_idx == ref_train and val_idx == ref_val, "splits.json differs from the source project's split"
    return data


# ---- Step 4/5: fixed sub-splits (indices are saved and verified like splits.json) ----------------------------
CACHE_SUBSET_PATH = OUTPUT_DIR / "cache_train_subset.json"
VAL_CAL_SEL_PATH = OUTPUT_DIR / "val_cal_sel.json"
CACHE_SUBSET_SIZE = 20000
SUBSPLIT_SEED = 42


def _write_or_verify(path, data, keys):
    payload = json.dumps({k: data[k] for k in keys})
    data["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    if path.exists():
        with open(path) as f:
            old = json.load(f)
        assert old["sha256"] == data["sha256"], f"{path} exists with different content"
        return old
    with open(safe_path(path), "w") as f:
        json.dump(data, f)
    return data


def create_subsplits():
    """20k cache subset of the 45k TRAIN indices, and VAL_CAL / VAL_SEL halves of the 5k val (positions 0..4999
    in val order). Fixed SUBSPLIT_SEED; written once, verified afterwards."""
    sp = load_splits()
    rng = np.random.RandomState(SUBSPLIT_SEED)
    subset = sorted(int(i) for i in rng.choice(sp["train_idx"], CACHE_SUBSET_SIZE, replace=False))
    cache = _write_or_verify(CACHE_SUBSET_PATH, {"seed": SUBSPLIT_SEED, "train_idx": subset}, ["seed", "train_idx"])
    perm = np.random.RandomState(SUBSPLIT_SEED + 1).permutation(N_VAL)
    cal, sel = sorted(int(i) for i in perm[: N_VAL // 2]), sorted(int(i) for i in perm[N_VAL // 2:])
    vs = _write_or_verify(VAL_CAL_SEL_PATH, {"seed": SUBSPLIT_SEED + 1, "cal": cal, "sel": sel}, ["seed", "cal", "sel"])
    return cache, vs


def load_subsplits():
    for p in (CACHE_SUBSET_PATH, VAL_CAL_SEL_PATH):
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found. Run scripts/11_cache_features.py first.")
    with open(CACHE_SUBSET_PATH) as f:
        cache = json.load(f)
    with open(VAL_CAL_SEL_PATH) as f:
        vs = json.load(f)
    train = set(load_splits()["train_idx"])
    assert len(cache["train_idx"]) == CACHE_SUBSET_SIZE and set(cache["train_idx"]) <= train, "cache subset not in train"
    assert len(vs["cal"]) + len(vs["sel"]) == N_VAL and not set(vs["cal"]) & set(vs["sel"])
    assert sorted(vs["cal"] + vs["sel"]) == list(range(N_VAL))
    return cache, vs
