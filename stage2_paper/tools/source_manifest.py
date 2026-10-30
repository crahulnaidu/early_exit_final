"""Read-only manifest of the source project (/home1/jtt/early_exit_final/stage1_base_nas).

  python tools/source_manifest.py create   -> reports/SOURCE_MANIFEST.txt (in the WORK folder)
  python tools/source_manifest.py verify   -> exit 1 if any recorded file changed / disappeared

Full sha256 for every small file; for files > 256 MB (the feature-cache memmaps) sha256 of the first + last 64 MB
plus the exact size. The source is only ever opened for reading.
"""
import hashlib
import sys
from pathlib import Path

SRC = Path("/home1/jtt/early_exit_final/stage1_base_nas")
MANIFEST = Path("/home1/jtt/early_exit_final/stage2_paper/reports/SOURCE_MANIFEST.txt")
PARTIAL_ABOVE = 256 * 2 ** 20
CHUNK = 64 * 2 ** 20

PATTERNS = [
    "checkpoints/**/*.pth",
    "cache/meta.json", "cache/quick/meta.json", "cache/quick/*.npy", "cache/proxy_logits/*.pt",  # cache/*.npy deleted (P16)
    "outputs/splits.json", "outputs/cache_train_subset.json", "outputs/val_cal_sel.json",
    "outputs/backbone_results.json",
    "outputs/nas/*.json", "outputs/nas/*.csv", "results/nas/*.md",
]


def digest(path):
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= PARTIAL_ABOVE:
            for block in iter(lambda: f.read(2 ** 20), b""):
                h.update(block)
            return "full", size, h.hexdigest()
        h.update(f.read(CHUNK))
        f.seek(size - CHUNK)
        h.update(f.read(CHUNK))
        return "first+last64MB", size, h.hexdigest()


def files():
    out = set()
    for pat in PATTERNS:
        out.update(p for p in SRC.glob(pat) if p.is_file())
    return sorted(out)


def create():
    lines = [f"# source manifest of {SRC} (read-only)", "# mode\tsize_bytes\tsha256\trelative_path"]
    for p in files():
        mode, size, h = digest(p)
        lines.append(f"{mode}\t{size}\t{h}\t{p.relative_to(SRC)}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"[manifest] {len(lines) - 2} files -> {MANIFEST}")


def verify():
    bad = 0
    n = 0
    for line in MANIFEST.read_text().splitlines():
        if line.startswith("#"):
            continue
        mode, size, h, rel = line.split("\t")
        p = SRC / rel
        n += 1
        if not p.is_file():
            print(f"[MISSING] {rel}")
            bad += 1
            continue
        m2, s2, h2 = digest(p)
        if (m2, str(s2), h2) != (mode, size, h):
            print(f"[CHANGED] {rel}")
            bad += 1
    print(f"[manifest verify] {n} files, {bad} changed/missing -> {'OK' if bad == 0 else 'FAILED'}")
    return bad == 0


if __name__ == "__main__":
    if sys.argv[1:] == ["create"]:
        create()
    elif sys.argv[1:] == ["verify"]:
        sys.exit(0 if verify() else 1)
    else:
        sys.exit("usage: source_manifest.py create|verify")
