#!/bin/bash
# Status of a job chain: state (squeue/sacct), last 20 log lines, and which outputs exist.
# Usage: bash scripts/check_status.sh [jobs table]   (default: the newest logs/slurm/jobs*.tsv)
#   logs/slurm/jobs.tsv            overnight run 1 (Steps 1-3)
#   logs/slurm/jobs_overnight2.tsv overnight run 2 (Steps 3B-6)
WORK_DIR=/home1/jtt/early_exit_final/stage1_base_nas
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
echo "outputs:"
for f in outputs/splits.json checkpoints/vgg16_bn_cifar10_base.pth outputs/backbone_results.json \
         checkpoints/ee_old16_23.pth outputs/clean_eval.csv results/fault_summary.csv results/analysis_status.json \
         cache/meta.json outputs/cache_train_subset.json outputs/val_cal_sel.json outputs/nas/proxy_check.json \
         outputs/nas/trials_tpe.csv outputs/nas/trials_random.csv outputs/nas/trials_handplaced.csv \
         outputs/nas/nas_summary.json outputs/nas/chosen_config.json checkpoints/ee_nas.pth \
         outputs/nas/clean_eval.csv results/nas/campaign_margin_band_status.json \
         results/nas/campaign_maxlogit_band_status.json results/nas/campaign_energy_status.json \
         results/nas/fault_summary.csv results/nas/analysis_status.json results/nas/compare_models.md; do
    if [ -e "$f" ]; then echo "  [x] $f"; else echo "  [ ] $f"; fi
done
for d in "" nas/; do
    for rule in margin_band maxlogit_band energy; do
        for loc in G1 G2 G3 HEADS; do
            f=results/${d}faults_${rule}_${loc}.csv
            [ -f "$f" ] && echo "  [x] $f: $(( $(wc -l < "$f") - 1 ))/800 runs"
        done
    done
done
for f in outputs/nas/trials_tpe.csv outputs/nas/trials_random.csv; do
    [ -f "$f" ] && echo "  NAS progress $f: $(( $(wc -l < "$f") - 1 ))/250 trials"
done
echo "disk: $(df -h /home1 | tail -1 | awk '{print $4" free of "$2" ("$5" used)"}') | work folder $(du -sh . 2>/dev/null | cut -f1)"
