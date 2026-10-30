import os
from pathlib import Path

WORK_ROOT = "/home1/jtt/early_exit_final/stage1_base_nas/"
SCRATCH_ROOT = "/scratch/jtt/early_exit_final/stage1_base_nas/"  # large data moved to scratch, linked back into WORK_ROOT


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
