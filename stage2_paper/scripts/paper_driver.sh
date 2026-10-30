#!/bin/bash
# Drives the paper chain K1..K8 from tmux (session "paper_driver", login node, CPU only; jobs run on dgx1 via sbatch).
# Submits one job at a time (account MaxJobs = 1), waits for it, then submits the next.
#   COMPLETED                         -> next job
#   failed with disk-full in its log  -> wait until the jtt quota has >= 30 GB left, resubmit (max 3 retries; campaigns resume)
#   any other failure / failed check  -> write reports/FAILURE.md and STOP (no workaround)
# K8 runs even if K7 failed (afterany semantics of the original chain). Job table: logs/slurm/jobs_paper2.tsv
WORK_DIR=/home1/jtt/early_exit_final/stage2_paper
cd "$WORK_DIR" || exit 1
TABLE=logs/slurm/jobs_paper2.tsv
LOG=logs/paper_driver.log
JOBS="K1:p_k1_campaigns K2:p_k2_val_mendnet K3:p_k3_training K5:p_k5_campaigns2 K6:p_k6_int8 K7:p_k7_latency K8:p_k8_eval_tables"
[ -f "$TABLE" ] || printf "label\tjobid\tscript\tdep\tlimit\n" > "$TABLE"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
free_gb() { echo $(( 50 - $(du -s --block-size=1G /home1/jtt 2>/dev/null | cut -f1) )); }   # jtt quota (50 GB) minus usage; df cannot see it

for spec in $JOBS; do
    L=${spec%%:*}; F=${spec##*:}
    if grep -q "^$L	.*	COMPLETED$" "$LOG.done" 2>/dev/null; then say "$L already completed, skipping"; continue; fi
    tries=0
    while true; do
        id=$(sbatch --parsable "scripts/slurm/$F.sbatch")
        lim=$(grep -- "--time=" "scripts/slurm/$F.sbatch" | cut -d= -f2)
        printf "%s\t%s\t%s\t%s\t%s\n" "$L" "$id" "$F.sbatch" "driver" "$lim" >> "$TABLE"
        say "$L submitted as $id (try $((tries + 1)))"
        while squeue -h -j "$id" 2>/dev/null | grep -q .; do sleep 120; done
        sleep 30
        state=$(sacct -n -X -j "$id" -o State%20 | head -1 | xargs)
        logf=$(ls logs/slurm/*-"$id".out 2>/dev/null | head -1)
        say "$L ($id) finished: $state"
        if [ "$state" = "COMPLETED" ]; then
            printf "%s\t%s\tCOMPLETED\n" "$L" "$id" >> "$LOG.done"
            break
        fi
        if [ -n "$logf" ] && grep -q -E "No space left|Disk quota exceeded|requested and [0-9]+ written|ENOSPC" "$logf" && [ "$tries" -lt 3 ]; then
            tries=$((tries + 1))
            say "$L hit disk-full; waiting for >= 30 GB left in the jtt quota (now $(free_gb) GB), then resubmitting"
            while [ "$(free_gb)" -lt 30 ]; do sleep 300; done
            find results/campaigns -name "*.part" -delete
            continue
        fi
        if [ "$L" = "K7" ]; then
            say "K7 failed ($state); continuing to K8 (afterany)"
            { echo "## $(date '+%F %T') job K7 ($id) state=$state (K8 still run)"; echo "log: $WORK_DIR/$logf"; echo; } >> reports/FAILURE.md
            break
        fi
        { echo "## $(date '+%F %T') job $L ($id) state=$state - chain STOPPED by paper_driver"
          echo "script: $F.sbatch ; log: $WORK_DIR/$logf"; echo; } >> reports/FAILURE.md
        say "$L failed ($state); chain stopped. See reports/FAILURE.md"
        exit 1
    done
done
say "ALL JOBS FINISHED"
