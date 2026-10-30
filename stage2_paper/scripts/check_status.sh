#!/bin/bash
# Status of a job chain: state (squeue/sacct), last 20 log lines, and which outputs exist.
# Usage: bash scripts/check_status.sh [jobs table]   (default: the newest logs/slurm/jobs*.tsv)
#   logs/slurm/jobs.tsv            overnight run 1 (Steps 1-3)
#   logs/slurm/jobs_overnight2.tsv overnight run 2 (Steps 3B-6)
WORK_DIR=/home1/jtt/early_exit_final/stage2_paper
cd "$WORK_DIR" || exit 1
JOBS=${1:-$(ls -t logs/slurm/jobs*.tsv 2>/dev/null | head -1)}
[ -f "$JOBS" ] || { echo "no jobs table found (nothing submitted yet)"; exit 1; }
echo "jobs table: $JOBS"

while IFS=$'\t' read -r label jobid script dep limit; do
    [ "$label" = "label" ] && continue
    state=$(squeue -h -j "$jobid" -o "%T %M %R" 2>/dev/null)
    [ -z "$state" ] && state=$(sacct -n -X -j "$jobid" -o State%22,Elapsed,ExitCode 2>/dev/null | head -1 | xargs)
    echo "=================================================================="
    echo "$label  job $jobid  [$script]  dep=$dep  limit=$limit"
    echo "state: ${state:-unknown}"
    log=$(ls logs/slurm/*-"$jobid".out 2>/dev/null | head -1)
    if [ -n "$log" ]; then
        echo "--- last 20 lines of $log (NFS cleanup noise filtered) ---"
        grep -v -e '\.nfs' -e 'rmtree' -e 'unlink' -e 'onerror' -e '_rmtree_safe_fd' -e '_run_finalizers' \
             -e 'multiprocessing/util.py' -e '^Traceback (most recent call last):$' "$log" | tail -20
    else
        echo "(no log yet)"
    fi
done < "$JOBS"

echo "=================================================================="
echo "outputs (NAS variant study):"
for f in results/phase0/0d_bit_summary.csv outputs/nas_variants/probe_faults.txt outputs/nas_variants/probe_validation.json \
         outputs/nas_variants/proxy_fidelity.json results/nas_variants/T1_search_variants.csv \
         outputs/nas_variants/picks.json results/nas_variants/variants_tables.md reports/FAILURE.md; do
    if [ -e "$f" ]; then echo "  [x] $f"; else echo "  [ ] $f"; fi
done
for f in outputs/nas_variants/trials_[A-E]*_s*.csv; do
    [ -f "$f" ] && echo "  [x] $f: $(( $(wc -l < "$f") - 1 ))/250 trials"
done
echo "disk: $(df -h /home1 | tail -1 | awk '{print $4" free of "$2" ("$5" used)"}') | work folder $(du -sh . 2>/dev/null | cut -f1)"
