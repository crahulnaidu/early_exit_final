"""Central paths for early_exit_new_2.

The CIFAR-10 data (/home1/jtt/early_exit_final/data) is READ-ONLY: it is loaded by path,
nothing is ever written there. (Originally this project read data from /home1/jtt/early_exit_new.)
All outputs go under WORK_DIR (enforced by utils.paths.safe_path).
"""
from pathlib import Path

WORK_DIR = Path("/home1/jtt/early_exit_final/stage1_base_nas")
SRC_DIR = Path("/home1/jtt/early_exit_final")  # read-only (shared data of early_exit_final)

# Read-only inputs
SRC_CKPT_DIR = SRC_DIR / "checkpoints"  # unused (the old early_exit_new checkpoints are not needed)
DATA_DIR = SRC_DIR / "data"  # CIFAR-10 (cifar-10-batches-py already extracted, sha-verified)

# Writable outputs (work folder only)
CKPT_DIR = WORK_DIR / "checkpoints"
OUTPUT_DIR = WORK_DIR / "outputs"
RESULTS_DIR = WORK_DIR / "results"
REPORTS_DIR = WORK_DIR / "reports"
LOG_DIR = WORK_DIR / "logs"
CACHE_DIR = WORK_DIR / "cache"  # frozen-backbone feature cache (gitignored)

PYTHON = "/home1/jtt/haripriya/env/bin/python"
