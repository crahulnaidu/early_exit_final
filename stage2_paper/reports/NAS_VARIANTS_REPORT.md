# NAS VARIANT STUDY: report (Phase S, Phase 0, Phase 1 submitted)

Work folder `/home1/jtt/early_exit_new_3` · source `/home1/jtt/early_exit_new_2` (read-only) · written 2026-10-22 ~12:40 IST · code commit `b80fc32`

**Status:**
- Phase S and Phase 0a–0c are done.
- Phase 0d, Phase 1 and the Phase 2 pick table run in the submitted job chain (§4).
- **Phase 3 (full evaluation of the picks) waits for your approval**, as instructed.

**Check progress:** `bash scripts/check_status.sh logs/slurm/jobs_variants.tsv`

---

## 1. Phase S: setup

| Step | Result |
|---|---|
| S1 manifest | `reports/SOURCE_MANIFEST.txt`: **1,534 source files**. 1,518 fully hashed (8 checkpoints, cache index files, 1,449 proxy-memo entries, splits, `outputs/nas/*.json|csv`, `results/nas/*.md`); 16 cache memmaps over 256 MB hashed on first + last 64 MB, with exact size |
| S2 clone | `git clone early_exit_new_2 early_exit_new_3` (commit `c5748d0`), tag **`v1-from-new2`**. Push URL set to `DISABLED-read-only-source` |
| S3 config | `WORK_DIR = early_exit_new_3`, `SRC_DIR = early_exit_new_2`; `BACKBONE_CKPT` / `EE_OLD16_23_CKPT` / `EE_NAS_CKPT` and `CACHE_DIR` point into the source (load only). `safe_path` allows writes only under `early_exit_new_3/` (new tests: writes under the source raise; `ckpt_path` returns source checkpoints for loading only, and saving to them raises) |
| S4 copies | splits, `cache_train_subset.json`, `val_cal_sel.json`, `backbone_results.json` (plus the quick one), and the proxy memo (1,406 + 43 files, 276 MB). **All sha256-identical** to the source; new memo entries go only to the work folder |
| S5 environment | Source `.venv` activated read-only (no new packages needed); TMPDIR = `early_exit_new_3/tmp`; logs in `early_exit_new_3/logs/slurm/` |
| S6 tests | **654 passed** (652 + 2 new write-protection tests). The `--quick` NAS run read the source's quick cache and checkpoint, wrote only to `early_exit_new_3/outputs/quick/nas`, and `find` showed **0 source files** newer than the start marker |
| S7 manifest | Verified after Phase S and before submission: **1,534 files, 0 changed/missing**. Every job re-verifies it at the end |

## 2. Phase 0: results so far

### 0a: Hypervolume over 3 sampler seeds (source runs, same reference point)

| Sampler | Seeds | Final HV mean ± std | Per seed |
|---|---|---|---|
| TPE | 1, 11, 21 | **7.038 ± 0.453** | 6.517, 7.270, 7.329 |
| Random | 2, 12, 22 | 6.230 ± 0.281 | 6.073, 6.063, 6.555 |

### 0b: NAS model vs 16/23 model (margin_band; G1–G3 paired Wilcoxon, HEADS Mann-Whitney; Holm)

Pooled over k; 800 runs per location. Means are percent of clean-correct test images.

| Location | SDC 16/23 | SDC NAS | Holm p | Flag 16/23 | Flag NAS | Holm p |
|---|---|---|---|---|---|---|
| G1 | 0.380 | **0.420** | 3.0e-14 | 22.31 | **20.70** | 8.5e-5 |
| G2 | 0.076 | **0.068** | 3.0e-19 | 4.50 | **2.96** | 1.8e-17 |
| G3 | 0.0039 | **0.0032** | 6.1e-6 | 1.48 | **1.36** | 1.2e-48 |
| HEADS | **0.020** | 0.083 | 1.2e-12 | 0.63 | 0.30 | 0.17 (n.s.) |

**What this shows:**
- **G2 and G3:** the NAS model has significantly **lower** SDC and fewer flags (fewer lost images).
- **G1:** slightly **higher** SDC (+0.04 points) but fewer flags.
- **HEADS:** higher SDC per head fault, though the NAS heads hold 2.7× fewer weights.
- **Per k (Holm):** SDC differences are significant in 5/8 k values for G1, 7/8 for G2, 0/8 for G3 and 3/8 for HEADS. NAS is lower in 1/8 (G1), 7/8 (G2), 8/8 (G3) and 0/8 (HEADS).
- **Robustness:** the same conclusions hold on identical-fault pairs only (all p ≤ 3e-5).

### 0c: HEADS faults per head and per bit (`results/phase0/0c_*`)

**Per head:**
- **NAS model:** almost all head SDC comes from **exit 1** (position 16, the head most images leave through):
  - E1: 0.044 % SDC per fault (k=1, n=34), 17.6 % of runs with any SDC.
  - E2: 0 (n=18).
  - E3: 0.0014 % (n=48).
- **16/23 model:** its large A3 head (E2) gets 98 % of head faults but causes only 0.0017 % SDC per fault. Its small A4 head (E1) was hit only twice.
- **So:** head SDC depends on *which exit carries the traffic*, not only on head size. This is the motivation for variant E.

**Per bit (k = 1):**
- **Bit 30 dominates.** Mean SDC 0.163 % for NAS vs 0.028 % for 16/23; any-SDC in 56 % vs 67 % of runs.
- **Bits 24, 25 and 27** cause rare SDC (NAS only).
- **Bits 23, 26, 28 and 29** caused none in the head runs.

### 0d: Bit-26 analysis

This runs in J0 (GPU replay of all 100 k=1 G1 runs per model, with an exact reproduction check against the campaign). Outputs: `results/phase0/0d_*` and `results/phase0/figures/0d_bit26_vs_bit30.png`.

## 3. Phase 1 / Phase 2: what runs and how it is compared

| Variant | Sampler / setting | Seeds | Source of the runs |
|---|---|---|---|
| A1 TPE (full space) | TPESampler(multivariate) | 1, 11, 21 | **reused** from the source (read-only CSVs) |
| A2 NSGA-II | NSGAIISampler(pop 50) | 3, 13, 23 | new (J1) |
| A3 Random | RandomSampler | 2, 12, 22 | **reused** from the source |
| B1 menu heads A1–A5 | TPE | 1, 11, 21 | new (J5) |
| B2 positions only (default head) | TPE | 1, 11, 21 | new (J5) |
| B3 heads only (positions 16, 23) | TPE | 1, 11, 21 | new (J5) |
| C1 / C2 / C3 exactly 1 / 2 / 3 exits | TPE | 1, 11, 21 | new (J4) |
| D1 margin_band / D2 maxlogit_band fixed | TPE | 1 | new (J6) |
| E1 4 objectives (+head_fault_sdc) | TPE | 1, 11, 21 | new (J2) |
| E2 4 objectives (params → exposure) | TPE | 1, 11, 21 | new (J2) |
| F proxy fidelity (4/8/16 vs 60 epochs, 20 configs) + pick stability | – | – | new (J3) |
| E probe validation (10 Pareto configs, probe vs 200-fault campaign) | – | – | new (J2) |

- **Comparison basis:** every hypervolume uses the same 3 objectives and the same reference point as the source's `14_pick_nas.py`. The summary reproduces the source's TPE value (7.038 ± 0.453). E also gets a separate 4-objective HV (SDC reference 20 %).
- **Fault probe** (`faults/head_probe.py`): about 0.5 s per trial on the quick cache. The real cost is measured on the first 5 trials in J2, which uses 40 faults or falls back to 20 if a trial takes more than 20 s, and records the choice.
- **Pick rules (Phase 2, fixed in `35_variants_summary.py` before any run):**
  - **P1:** lowest FLOPs on the best A variant's front (union of seeds).
  - **P2:** lowest head params on the same front.
  - **P3:** E1, lowest head_fault_sdc with saved ≥ 39.81 % (the 16/23 model's saving under the same scoring).
  - **P4:** C2, lowest FLOPs.
- **Pick table:** shows all four on VAL_SEL with recomputed probe values, together with the 16/23 model, the hand-placed design and the current NAS model, and flags a pick identical to the current NAS model. Picks are written to `outputs/nas_variants/picks.json` for Phase 3.

## 4. Submitted jobs (all `--partition=longq --qos=longq --gres=gpu:a100:1`)

| Label | Job ID | What | Limit | Dependency |
|---|---|---|---|---|
| J0 | **47755** | Phase 0 stats + 0d bit replay; check; manifest verify | 0:40 | – |
| J1 | **47756** | A2 NSGA-II × 3 seeds | 2:00 | afterok J0 |
| J2 | **47757** | E probe timing → E1 × 3, E2 × 3 → probe validation | 6:00 | afterok J1 |
| J3 | **47758** | F proxy fidelity + pick stability | 2:30 | afterok J2 |
| J4 | **47759** | C1/C2/C3 × 3 seeds | 3:30 | afterok J3 |
| J5 | **47760** | B1/B2/B3 × 3 seeds | 3:00 | afterok J4 |
| J6 | **47761** | D1/D2 × 1 seed | 1:30 | afterok J5 |
| J7 | **47762** | Summary, T1, picks P1–P4, figures | 1:30 | afterany J6 |

**Estimated GPU time** (from the source's measured A100 speeds: 13–16 min per 250-trial search with mostly memoised heads; bank retraining about 2–10 s per head):

| Job | Estimate |
|---|---|
| J0 | 0.1 h |
| J1 | 0.8 h |
| J2 | about 3 h |
| J3 | about 1 h |
| J4 | about 1.8 h |
| J5 | about 1.2 h |
| J6 | 0.5 h |
| J7 | 0.3 h |
| **Total** | **≈ 9 h**, under the 12 GPU-h budget, so nothing was dropped |

**Queue warning:** at submission all six full A100s were busy (jobs with 12–48 h limits plus a queued chain of eight 20-hour A100 jobs), and `--test-only` estimated the earliest start as 2026-10-30. That is a worst case from time limits; the real start depends on when jobs end. **I did not switch** to the free MIG slice (`a100_3g.20gb`, 20 GB) because you asked for full A100s, and the 15.5 GB feature cache would only just fit at about half the speed. If you want it, it only needs `--gres=gpu:a100_3g.20gb:1` in the job files.

## 5. Decisions

N1–N14 in `DECISIONS.md` (new section at the end). Most relevant:
- **N2:** A1/A3 reuse the source runs.
- **N4:** exact exit counts via combination categoricals.
- **N6:** the E probe uses in-memory bank heads, while accuracy/FLOPs/params use memo logits.
- **N7:** E2 objectives taken literally.
- **N11:** picks use the union of seeds.
- **N13:** strict `afterok` chain.

## 6. Source-manifest check

`tools/source_manifest.py verify` after Phase S and at the end of this session: **1,534 files, 0 changed/missing**. Each job repeats it, and a mismatch fails that job.

## 7. Open items

1. **Phase 3 (full evaluation of P1–P4):** waits for your approval after the pick table exists. Expect about 1.5–2 h per model; P1 is skipped if it equals the current NAS model.
2. **GPU:** decide whether to allow the MIG slice if the A100 queue stays blocked.
