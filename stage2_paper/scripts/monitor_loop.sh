#!/bin/bash
# Monitor for the NAS variant chain (runs in tmux session "nasvar_monitor" on the login node; CPU only).
# Every 15 min: append job status + disk usage to logs/monitor.log. If a job FAILED / TIMEOUT / CANCELLED or its log
# shows a failed check, write the job and its log path to reports/FAILURE.md (once per job). No resubmission.
WORK_DIR=/home1/jtt/early_exit_final/stage2_paper
JOBS=${1:-$WORK_DIR/logs/slurm/jobs_variants.tsv}
cd "$WORK_DIR" || exit 1
declare -A REPORTED
while true; do
    {
        echo "##### $(date '+%F %T')"
        bash scripts/check_status.sh "$JOBS"
        df -h /home1
    } >> logs/monitor.log 2>&1
    while IFS=$'\t' read -r label jobid script dep limit; do
        [ "$label" = "label" ] && continue
        [ -n "${REPORTED[$jobid]}" ] && continue
        state=$(sacct -n -X -j "$jobid" -o State%20 2>/dev/null | head -1 | xargs)
        log=$(ls logs/slurm/*-"$jobid".out 2>/dev/null | head -1)
        failed_check=""
        [ -n "$log" ] && grep -q -e "\] FAILED" -e "^\[FAIL\]" "$log" && failed_check="yes"
        case "$state" in
            FAILED*|TIMEOUT*|CANCELLED*|OUT_OF_ME*|NODE_FAIL*) bad=1 ;;
            *) bad=0 ;;
        esac
        if [ "$bad" = 1 ] || [ -n "$failed_check" ]; then
            {
                echo "## $(date '+%F %T') job $label ($jobid) state=$state failed_check=${failed_check:-no}"
                echo "script: $script ; log: $WORK_DIR/${log:-<no log>}"
                echo
            } >> reports/FAILURE.md
            REPORTED[$jobid]=1
        fi
    done < "$JOBS"
    sleep 900
done
