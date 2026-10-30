"""Shared CLI flags, output locations and run metadata for every script."""
import datetime
import subprocess
import sys

from config import CKPT_DIR, OUTPUT_DIR, RESULTS_DIR, WORK_DIR
from utils.paths import safe_path
from utils.splits import load_splits


def add_common_args(parser):
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 2 batches per split, 2 epochs/runs; outputs go to */quick/")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tag", default="", help="sub-folder for outputs/results of one model (e.g. nas)")
    return parser


def _sub(root, quick):
    return root / "quick" if quick else root


def ckpt_path(name, quick):
    return safe_path(_sub(CKPT_DIR, quick) / name)


def output_path(name, quick, tag=""):
    return safe_path(_sub(OUTPUT_DIR, quick) / tag / name)


def results_path(name, quick, tag=""):
    return safe_path(_sub(RESULTS_DIR, quick) / tag / name)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(WORK_DIR), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_meta(args, **extra):
    """Metadata written into every checkpoint / JSON / CSV: seed, split hash, git commit, command."""
    meta = {
        "seed": args.seed,
        "quick": bool(args.quick),
        "split_sha256": load_splits()["sha256"],
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    meta.update(extra)
    return meta
