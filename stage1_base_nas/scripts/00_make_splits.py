"""Create (or verify) outputs/splits.json: the fixed 45k/5k train/val split used by every script."""
import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.splits import SPLIT_SEED, SPLITS_PATH, create_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    data = create_splits(args.seed)
    print(f"[splits] {SPLITS_PATH} | seed={data['seed']} | train={data['n_train']} | val={data['n_val']} | "
          f"sha256={data['sha256']}", flush=True)


if __name__ == "__main__":
    main()
