# OVERNIGHT REPORT 3: NAS variant study on the 20 GB MIG slice

Work folder `/home1/jtt/early_exit_new_3` · source `/home1/jtt/early_exit_new_2` (read-only) · written 2026-10-22 ~15:35 IST · job code commit `b5542a8`

**Tonight:** no full training, no test-set evaluation, no full fault campaign. Phase 3 needs your approval.

**Check progress:**
- Status: `bash scripts/check_status.sh logs/slurm/jobs_variants.tsv`
- Monitor log: `tail -60 logs/monitor.log`
- Monitor session: `tmux attach -t nasvar_monitor` (detach with Ctrl-b d)
- Failures, if any: `reports/FAILURE.md`

---

## 1. Pre-checks

| Check | Result |
|---|---|
| P1 jobs 47755–47762 | Not in the queue (cancelled earlier today) |
| P2 disk | `/home1` **61 GB free** (> 20 GB); work folder 283 MB |
| P3 tests | **654 passed** before the changes, **657** after Step 1 |
| P3 source manifest | **1,534 files, 0 changed/missing** |

## 2. Step 1: E2 fix (commit `e1fac01`)

**E2 is now 3 objectives:** maximise accuracy, minimise expected FLOPs, minimise **exposure** (= faultable head weights × head_fault_sdc / 1e6).
- head_fault_sdc is no longer a separate E2 objective.
- E1 is unchanged: (accuracy, FLOPs, params, head_fault_sdc).
- **Code:** `objective_spec()` in `scripts/32_nas_variants.py`; tests in `tests/test_variants.py`.
- **Summary:** every variant gets the **common** 3-objective HV (accuracy, FLOPs, params) on the shared reference point. E1 and E2 also get an **own** HV:
  - E1: 4 objectives, SDC reference 20 %.
  - E2: (accuracy, FLOPs, exposure), exposure reference = params bound × 20 = 186.2.
- DECISIONS N7 and N8 were rewritten.

## 3. Step 2: GPU = 20 GB MIG slice

- **Availability (`sinfo` / `scontrol`):** `gres gpu:a100:6, gpu:a100_3g.20gb:2, gpu:a100_1g.5gb:7`. Allocated: all 6 `a100` plus 1 of the 2 `a100_3g.20gb`, so **one slice was free**.
- **Smoke job 47770** (`--gres=gpu:a100_3g.20gb:1`, 30 min limit; ran 47 s; all 17 positions of the **full feature cache preloaded**, 3 trials each):

| Run | GPU | Peak `max_memory_allocated` | Per-trial time |
|---|---|---|---|
| NSGA-II (J1 code) | A100-SXM4-40GB MIG 3g.20gb (21.1 GB) | **15.57 GB** | 3.95 s with 1 new head; ≈ 0 s memoised |
| E1 probe + head bank (J2 code) | same | **16.41 GB** | 9.4 s average (8.0 + 6.4, 5.1 + 6.9, 0.0 + 1.8 s) |

- **Cache mode: GPU.** Peak memory is under 18 GB, so the pinned-CPU fallback was **not** needed and arithmetic is unchanged (FP32, TF32 off).
- **Time limits:** measured per-trial time × trials × 1.5, with a minimum of 2× the old limit. The minimum dominated every job (E: 9.4 s × 1,500 × 1.5 ≈ 5.9 h < 12 h).
- **J3 memory:** `34_proxy_fidelity` now frees the feature cache before the 60-epoch training (N17).

## 4. Step 3: job chain (strict `afterok`, MaxJobs = 1; all `--gres=gpu:a100_3g.20gb:1 --partition=longq --qos=longq`)

| Label | Job ID | What | Limit | Dependency | Status at writing |
|---|---|---|---|---|---|
| J0 | **47771** | Phase 0d bit-26 vs bit-30 replay; check; manifest | 1:20 | – | **COMPLETED** (3 min 57 s), all checks passed |
| J1 | **47772** | A2 NSGA-II, seeds 3/13/23 | 4:00 | afterok J0 | RUNNING since 15:26 |
| J2 | **47773** | E probe timing → E1 × 3 → E2 × 3 → probe validation | 12:00 | afterok J1 | pending |
| J3 | **47774** | F proxy fidelity (20 configs, 4/8/16 vs 60 epochs) + pick stability | 5:00 | afterok J2 | pending |
| J4 | **47775** | C1/C2/C3 exactly 1/2/3 exits × 3 seeds | 7:00 | afterok J3 | pending |
| J5 | **47776** | B1 menu / B2 positions / B3 heads × 3 seeds | 6:00 | afterok J4 | pending |
| J6 | **47777** | D1 margin_band / D2 maxlogit_band × seed 1 | 3:00 | afterok J5 | pending |
| J7 | **47778** | Summary: T1, common + own HVs, picks P1–P4 (VAL_SEL), pick table, `picks.json`, figures | 3:00 | afterany J6 | pending |

**Budget and finish time:**
- **Expected GPU use:** about 10–11 h, under the 24 h budget, so **no reductions** to D, B or C.
- **Rough split:** J1 ≈ 1.2 h, J2 ≈ 3.2 h, J3 ≈ 1.4 h, J4 ≈ 2.5 h, J5 ≈ 1.2 h, J6 ≈ 0.5 h, J7 ≈ 0.3 h.
- **Expected finish: about 02:00–03:00 IST on 2026-10-23.** `squeue --start` shows N/A for dependent jobs (Slurm doesn't estimate them); the slice is held by our running job, so later jobs should start right after each predecessor.
- **Each job ends** with its check script and a source-manifest verify.

## 5. Phase 0d result (from J0)

All 100 k=1 G1 runs per model were replayed, with **0 mismatches** against the campaign over 200 runs.

| Bit | NAS mean SDC % | NAS mean flag % | Changed images that escaped as SDC | Max final max-logit ratio |
|---|---|---|---|---|
| 26 | **0.552** | 0.153 | **78 %** (756 of 966) | 2.1× |
| 30 | 0.135 | **25.07** | **0.5 %** (148 of 27,611) | 6.8e34× |

**Why bit 26 escapes:**
- A bit-26 flip that sets the bit ("grow"; 965 of 966 changed images came from such flips) multiplies the weight by 2⁸ = 256.
- That shifts the features enough to change predictions, but the max logit moves at most about 2.1×, so the logits **stay inside the calibrated [L, U] band**. Median max logit of the SDC images: 5.2.
- A bit-30 flip multiplies the weight by up to 2¹²⁸, so the logits explode (around 4e34) and land far above U, where they're flagged.
- The 16/23 model behaves the same way (bit 26: 80 % escape; bit 30: 0.5 %).

**Implication:** the band catches *magnitude* blow-ups but not *moderate* perturbations that keep logits in range. Catching those needs a different signal, such as cross-exit consistency.

Tables: `results/phase0/0d_bit_summary.csv`, `0d_bit_images.csv`, `0d_bit_runs.csv`. Figure: `results/phase0/figures/0d_bit26_vs_bit30.png`.

## 6. Monitor (tmux, login node, CPU only)

- **Session:** `nasvar_monitor`, running `scripts/monitor_loop.sh`, every 15 min.
- **Log:** appends `check_status.sh` output and `df -h /home1` to `logs/monitor.log`.
- **On failure:** if a job ends FAILED / TIMEOUT / CANCELLED or its log shows a failed check, it writes the job and log path to `reports/FAILURE.md` (once per job). It does nothing else (no resubmission).

## 7. Decisions added (DECISIONS.md)

- **N7/N8 (revised):** E2 has 3 objectives; common HV plus own HV for E.
- **N15:** MIG slice; smoke-test memory and timings; limits; no reductions.
- **N16:** J0 = Phase 0d only (0a–0c were done earlier, CPU only).
- **N17:** proxy fidelity frees the cache before full training.
- **N18:** monitor behaviour; the cancelled job table was archived.
- **N19:** Optuna's `constraints_func` deprecation warning (still works in 5.0.0; kept for comparability).

## 8. Next steps (need your instruction)

1. In the morning, review `results/nas_variants/variants_tables.md` (T1 + picks) and the figures in `results/nas_variants/figures/`.
2. Approve (or change) picks P1–P4 for **Phase 3** (full training + clean test + fault campaign per model; about 1.5–2 h each on an A100, longer on the slice).
