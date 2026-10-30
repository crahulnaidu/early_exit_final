import os
from pathlib import Path

WORK_ROOT = "/home1/jtt/early_exit_final/stage2_paper/"
SCRATCH_ROOT = "/scratch/jtt/early_exit_final/stage2_paper/"  # large data moved to scratch, linked back into WORK_ROOT


def safe_path(p):
    """Return p as a Path after asserting it resolves inside the work folder (or its scratch mirror).

    Use for EVERY write (checkpoints, CSV, JSON, figures, logs). Creates the parent directory.
    """
    resolved = os.path.realpath(os.path.abspath(str(p)))
    assert (resolved + "/").startswith((WORK_ROOT, SCRATCH_ROOT)), (
        f"Refusing to write outside {WORK_ROOT} (or {SCRATCH_ROOT}): {p} -> {resolved}"
    )
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    return Path(resolved)


# Folders that were merged into early_exit_final on 2026-10-24 (DECISIONS P13). Older result files (e.g. the
# ee_ckpt stored in campaign clean_*.pt) still name them; map such paths to their new home for reading.
LEGACY_ROOTS = {
    "/home1/jtt/early_exit_new_2/": "/home1/jtt/early_exit_final/stage1_base_nas/",
    "/home1/jtt/early_exit_new_3/": "/home1/jtt/early_exit_final/stage2_paper/",
}


def current_path(p):
    """Map a path recorded under a pre-merge folder to its location in early_exit_final (else unchanged)."""
    s = str(p)
    for old, new in LEGACY_ROOTS.items():
        if s.startswith(old):
            return Path(new + s[len(old):])
    return Path(s)
