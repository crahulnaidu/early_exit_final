# Paper experiments run (everything except HLS)

**Work folder:** `/home1/jtt/early_exit_new_3`. **Source:** `early_exit_new_2` (read-only; manifest verified at the end of every job). **Code commit:** `399c3b9`. **Submitted:** 2026-10-23 15:38.

**Check progress:**
- Status: `bash scripts/check_status.sh logs/slurm/jobs_paper.tsv`
- Monitor: `tmux attach -t paper_monitor` (log in `logs/monitor.log`)
- Failures: `reports/FAILURE.md`

## Pre-checks (15:20–15:38)

- **Queue and GPUs:** queue empty; 4 of 6 full A100s free.
- **Disk:** `/home1` has 59 GB free.
- **Tests:** **680 passed** (657 before + rules2 19 + offline 4).
- **Manifest:** source manifest OK.
- **`--quick` runs:** every new script (40 for each fault model and split, 41, 42, 43, 44 select, 05 with the new configs) and every new check mode passed.

## Chain (MaxJobs = 1; `--gres=gpu:a100:1`, longq)

| Job | ID | What | Limit |
|---|---|---|---|
| K1 | 47789 | Logit-storing exponent campaigns, test, 100 runs: NAS (16/22/23) and 16/23. Then the source-reproduction check (P5). | 5 h |
| K2 | 47790 | Val exponent runs (10) for MENDNet tuning, plus the MENDNet sign-flip fault model (FR 10/30/50 %, FP1–FP4, 10 runs). Both models. | 3 h |
| K3 | 47791 | Full training of 5 top candidates, then `44 select` (VAL_SEL), then P3/P4 heads, then the final configuration with seeds 1 and 2. | 4 h |
| K5 | 47792 | Campaigns: final model (if it is not the NAS model), P3, P4 (100 runs + val 10); seeds 1/2 (20 runs). | 9 h |
| K6 | 47793 | INT8 weight-fault campaign on the final model (test 100 + val 10). | 3 h |
| K7 | 47794 | Batch-1 latency, GPU and CPU, with real early stopping. | 1.5 h |
| K8 | 47795 | `41` offline evaluation of all rules (≈ 70 methods incl. α sweep, MENDNet clean / faulty-val-tuned, ablations), then `42` tables T2–T14 + figures. | 6 h (`afterany`) |

- **Expected GPU time:** about 9–12 h; the expected finish is the early morning of 2026-10-24.
- **Each job ends with:** its check and a source-manifest verify. A failed check stops the chain (not worked around).
- **First output (K1):** clean val **92.64 %** and test **92.27 %**, identical to the source.

## What comes out (for the paper)

| Output | Content |
|---|---|
| `results/paper/paper_tables.md` (+ CSVs, `figures/`) | T2 method comparison vs MENDNet / SDN / energy / msp. T3 per location. T4 MENDNet fault model. T5 INT8. T6 α sweep (ROC). T7 per bit. T8 band/slack ablation. T9 models. T10 seeds. T11 latency. T12 hardware cost. T13 NAS variants. T14 top-5 full training. R reproduction. |
| `results/offline/<tag>/` | Per-run rows for every method, summaries, per-bit / per-layer tables, calibrations. |
| `outputs/final/final_choice.json` | Resolution of the "unstable pick". |

**Decisions:** DECISIONS.md P1–P11.

**Not done here:** HLS synthesis (deferred, as agreed). T12 is the theoretical cost until then.
