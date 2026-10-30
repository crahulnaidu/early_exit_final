# Sourced by every job script: environment + header (nvidia-smi, git commit, seed) + run() helper.
WORK_DIR=/home1/jtt/early_exit_final/stage2_paper
cd "$WORK_DIR"
mkdir -p "$WORK_DIR/tmp" && export TMPDIR="$WORK_DIR/tmp"
source /home1/jtt/early_exit_final/.venv/bin/activate   # shared venv of early_exit_final
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
SEED=42

echo "===== JOB INFO ====="
echo "Job      : ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-none})"
echo "Node     : $(hostname)"
echo "Start    : $(date)"
echo "Git      : $(git rev-parse HEAD)"
echo "Dirty    : $(git status --porcelain --untracked-files=no | wc -l) modified tracked files"
echo "Seed     : $SEED"
echo "Python   : $(which python)"
echo "TMPDIR   : $TMPDIR"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
echo "===================="

run() {
    echo ">>> $(date +%T) $*"
    "$@"
    echo "<<< $(date +%T) exit 0: $*"
}

HOME_QUOTA_GB=50   # per-user quota of jtt on /home1 (not visible to df; see DECISIONS P13/P15)
quota_free_gb() { echo $(( HOME_QUOTA_GB - $(du -s --block-size=1G /home1/jtt 2>/dev/null | cut -f1) )); }
need_disk() {   # abort the job (chain stops) if the jtt quota on /home1 has less than $1 GB left
    local free=$(quota_free_gb)
    echo "disk: ${free} GB left in the jtt quota on /home1 (need >= $1)"
    [ "$free" -ge "$1" ] || { echo "[FAIL] disk: only ${free} GB free"; exit 1; }
}
