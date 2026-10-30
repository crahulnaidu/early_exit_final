"""Central paths for early_exit_new_3 (NAS variant study).

SRC_DIR (/home1/jtt/early_exit_final/stage1_base_nas) is READ-ONLY: its checkpoints, feature cache and results are loaded by path,
nothing is ever written there (utils.paths.safe_path only allows writes under WORK_DIR).
The CIFAR-10 data comes read-only from early_exit_final/data (DATA_DIR).
"""
from pathlib import Path

WORK_DIR = Path("/home1/jtt/early_exit_final/stage2_paper")
SRC_DIR = Path("/home1/jtt/early_exit_final/stage1_base_nas")          # read-only
DATA_DIR = Path("/home1/jtt/early_exit_final/data")     # read-only, CIFAR-10 (cifar-10-batches-py, sha-verified copy)

# Read-only inputs from the source project (load only)
SRC_CKPT_DIR = SRC_DIR / "checkpoints"
SRC_OUTPUT_DIR = SRC_DIR / "outputs"
SRC_RESULTS_DIR = SRC_DIR / "results"
BACKBONE_CKPT = SRC_CKPT_DIR / "vgg16_bn_cifar10_base.pth"
EE_OLD16_23_CKPT = SRC_CKPT_DIR / "ee_old16_23.pth"
EE_NAS_CKPT = SRC_CKPT_DIR / "ee_nas.pth"
CACHE_DIR = SRC_DIR / "cache"                           # feature cache memmaps, opened with mmap_mode="r"

# Writable outputs (work folder only)
CKPT_DIR = WORK_DIR / "checkpoints"
OUTPUT_DIR = WORK_DIR / "outputs"
RESULTS_DIR = WORK_DIR / "results"
REPORTS_DIR = WORK_DIR / "reports"
LOG_DIR = WORK_DIR / "logs"
PROXY_MEMO_DIR = WORK_DIR / "cache" / "proxy_logits"    # copy of the source memo; new entries only here

PYTHON = "/home1/jtt/early_exit_final/.venv/bin/python"  # shared venv of early_exit_final
