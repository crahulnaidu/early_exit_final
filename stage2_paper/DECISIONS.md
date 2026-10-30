# DECISIONS

Design choices made while working autonomously (overnight run, 2026-10-15). In each case the most conservative option was chosen. Each entry says where it lives in the code.

## Training / infrastructure

**D1. TF32 disabled.**
`utils/seed.py` sets `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`.
The fault model is FP32, so all GPU arithmetic (training, calibration, campaign) runs in true FP32 on the A100. The cost is speed.

**D2. Fixed split seed.**
The 45k/5k split always uses `SPLIT_SEED = 42` (`utils/splits.py`), independent of the run `--seed`. The run seed only drives init, shuffling, augmentation and workers.
- The split is identical to the source project's: same `RandomState(42)` shuffle, asserted on every load.
- sha256: `33870a3317ad3c8954b768b3e6a3038a2630d212b3231af5a121dffddb1b723a`.

**D3. Checkpoint format and loading.**
- New checkpoints are saved as `{"state_dict", "meta"}`, where meta holds the seed, split hash, git commit, command and epoch.
- `load_checkpoint` also accepts plain state dicts and always uses `strict=True`.
- Multi-exit checkpoints hold the full model. `load_searched_model` asserts that the embedded backbone is bit-identical to the base checkpoint, which guarantees the backbone stayed frozen.

**D4. Frozen backbone in train mode.**
`SupernetMultiExitVGG` and `SearchedMultiExitVGG` override `train()`, so `features`, `avgpool` and `classifier` always stay in eval mode (BN statistics fixed). `assert_backbone_frozen` runs every epoch.

**D5. The validation model is a separate file.**
The old configuration (16/A4, 23/A3) retrained on the new backbone is saved as `checkpoints/ee_old16_23.pth`. `ee_searched.pth` is reserved for the NAS result. Scripts 06–08 read positions and heads from the checkpoint's metadata, not from `chosen_config.json`.

**D6. `--quick` outputs are isolated.**
Quick runs write to `checkpoints/quick/`, `outputs/quick/` and `results/quick/`, so they never overwrite real artifacts. In quick mode, `scripts/checks/run_check.py` skips the accuracy thresholds (the models are trained on 2 batches). The structural checks still run.

**D7. 04_search unchanged.**
Its internal function was renamed from `search_nas_pareto` to `search_random`, because it is random sampling. Behaviour is unchanged otherwise.
- **Quick mode only:** if no configuration is feasible, the most accurate one is used, with a printed warning.
- **Full mode:** it raises an error.
- Step 5 replaces it.

**D8. FLOPs conventions.**
- 1 MAC = 2 FLOPs.
- BatchNorm = 2·C·H·W.
- ReLU = C·H·W.
- **Pooling = input elements**, matching fvcore. The old code counted max-pool output elements.
- Bias adds are not counted; dropout = 0.
- Result: F_base = 627.88 MFLOPs (conv 626.69 M + BN/ReLU/pool + classifier 0.535 M).

## Exit rules and calibration (Step 2)

**D9. What "maxlogit_band (no margin term)" means.**
Read literally ("exit if L_k ≤ maxlogit ≤ U_k"), it would send almost every sample out at exit 1, because there is no confidence threshold. Implemented instead:
- **Exit:** if finite AND maxlogit ≥ t_k AND L_k ≤ maxlogit ≤ U_k.
- t_k is calibrated with the same accuracy-aware rule as the margin.
- So the lower bound is effectively max(L_k, t_k).

**D10. Energy rule (ablation).**
- **Score:** −E(z), with the same accuracy-aware t_k (equivalently, exit if E ≤ −t_k).
- **Band:** [0.1th, 99.9th] percentile of the clean val energy E. The 0.1th-percentile lower bound is the "two-sided" part.
- **Final:** an [L_fin, U_fin] band on E, set from alpha.

**D11. How t_k is chosen.**
Exits are calibrated sequentially, each on the val samples that did not leave earlier, with the band applied.
- **Candidates:** the 0, 0.5, …, 100th percentiles of the score, plus +inf (exit disabled).
- **Selection:** the smallest candidate for which the accuracy criterion (head accuracy ≥ final accuracy on the same samples − 0.5 points) holds **for that candidate and every larger one**. This is more conservative than the first candidate that happens to pass once.
- **Small sets:** with fewer than 25 exiting samples, the criterion counts as satisfied.

**D12. Final band [L_fin, U_fin].**
alpha is split equally between the two tails (alpha/2 of all val samples per tail), among the samples that reach the final classifier, so the val FAR is alpha.
- If fewer samples reach the final than alpha·N (this happened only in `--quick`), all of them are flagged and `alpha_reachable = false` is recorded.
- Default alpha is 1 %; 0.5, 1, 2 and 5 % are all saved.

**D13. float32 thresholds.**
All thresholds are rounded to the nearest float32, so the batched comparisons (float32 tensors) and the slow reference loop agree bit for bit.

## Fault campaign (Step 3)

**D14. Same faults for every rule.**
fault seed = 1,000,000·loc_id + 1000·k + run, with loc_id G1=1, G2=2, G3=3, HEADS=4. It doesn't depend on the rule, so both rules draw the same candidate faults.
- The clean-correct subset is defined per rule (golden = the fault-free system *with that rule*).
- The non-finite check runs over that subset, so in rare cases a rejection can differ between the rules.

**D15. Sequential injection with rejection.**
In each run, faults are flipped one at a time.
- **Rejected:** if the new weight is non-finite, or any logit at any exit on the subset becomes non-finite. The flip is undone, logged with its reason, and a new fault is drawn.
- **One flip per weight:** applies to accepted faults. A weight whose flip was rejected may be drawn again, with another bit.
- **Safety cap:** more than 1000 attempts in one run raises an error, which stops the chain.

**D16. Partial recompute.**
For a fault in G_g (g ≥ 2), the campaign recomputes only features[p_{g−1}+1 …] from the cached clean activation. For HEADS, only the heads are recomputed. pytest checks this is bit-identical to a full forward pass, so it is exact.

**D17. Outcome definitions.**
On the clean-correct subset the golden prediction equals the label, so `correct_rate == masked_rate`. Both are reported. Correct + SDC + flagged = 100 % is asserted on every run.

**D18. Undo check.**
After every run, every faultable tensor is compared bit-for-bit (int32 view) with a snapshot. A sha256 of all faultable weights is also logged at the start and end of the job, and the two must match.

## Jobs

**D19. Summary only for finished campaigns.**
J6 runs after J4 (`afterany`, as specified) while J5 is still running. 09 therefore summarises only campaigns whose `campaign_<rule>_status.json` exists, and partial ones are skipped with a message.
- An extra short job, **J6b** (`afterany:J5`), reruns 09 so the energy results are included too.
- It takes a few minutes, is not long, and adds no new computation.

**D20. Logs.**
stdout and stderr are merged into `logs/slurm/<jobname>-<jobid>.out`.

**D21. NFS warnings.**
`OSError: [Errno 16] Device or resource busy: '.nfs…'` at exit comes from DataLoader temp-dir cleanup with TMPDIR on NFS (required: TMPDIR is in the work folder). It is harmless (exit code 0); `check_status.sh` filters it.

**D22. Time limits.**
Measured RTX 3050 throughput × 1.5, rounded up. The A100 should be 2× faster or more (FP32, no TF32), so these are upper bounds.

---

# Overnight run 2: Steps 3B, 4, 5, 6 (2026-10-15)

## Faults

**D23. Three backbone groups for any number of exits.**
G1 = features[0..p_first], G2 = features[p_first+1..p_last] (all middle segments merged), G3 = features[p_last+1..43] + classifier, plus HEADS (`faults/inject.py`).
- With 1 exit, G2 is empty and is skipped: the campaign, the checks and the summary only iterate over existing locations.
- Location ids stay fixed (G1=1, G2=2, G3=3, HEADS=4), so fault seeds don't depend on K. For the 2-exit validation model the groups and seeds are identical to last night's.

**D38. Step 3B pairing.**
`10_analyze_faults.py` re-applies the fault set logged by the rule-A campaign (margin_band for the 16/23 model) and computes the logits once on the union of both rules' clean-correct subsets. Both rules are then scored on the intersection, so the comparison is paired by construction.
- **Validation:** the replay must reproduce rule A's recorded SDC and flag counts exactly, and rule B's counts on runs where both campaigns logged identical faults. Otherwise J0 fails.
- **Wilcoxon signed-rank:** `zero_method="wilcox"` (zero differences dropped; the number of non-zero pairs is reported). p = 1 when every difference is zero.
- **Definitions:**
  - coverage = 100 − flag rate
  - selective accuracy = correct / accepted
  - detection ratio = Σ flagged / Σ (flagged + SDC), pooled over runs
  - exposure-weighted SDC = Σ (location's share of faultable weights × mean SDC)

## Outputs

**D24. `--tag`.**
NAS-model outputs go to `outputs/nas/` and `results/nas/`; its checkpoint is `checkpoints/ee_nas.pth`. Last night's files (no tag) are never overwritten.

**D35. 05 refactor.**
The head-training loop moved unchanged to `utils/train_heads.py`. 05 now accepts `--config <json>` and stores the rule in the checkpoint metadata.

## Feature cache and proxy (Step 4)

**D25. Cache contents.**
- **Train:** 20k train indices from `RandomState(42).choice(train_idx, 20000, replace=False)`, sorted.
- **Val:** full 5k val in val order. VAL_CAL / VAL_SEL = `RandomState(43).permutation(5000)` halves (indices are positions within val), saved and verified in `outputs/cache_train_subset.json` and `outputs/val_cal_sel.json`.
- **Format:** eval transforms (no augmentation); fp16 features (an fp16-overflow assertion runs while caching); fp16 images (so positions not cached can be computed on the fly); fp32 final logits (KD teacher); labels.
- **Positions:** all 17 if more than 40 GB is free (checked inside the job), else positions 2 and 5 are skipped.
- **Check:** the cached val logits must reproduce the backbone's val accuracy (±0.05).

**D26. Proxy training.**
- **Recipe:** one head at a time, 8 epochs, batch 128, Adam 1e-3, wd 1e-4, cosine (eta_min 1e-5), 05 KD loss (T=3, alpha=0.3) with the cached final logits as teacher, no augmentation.
- **Seeding:** the seed is derived from the head's config key, so a head's result doesn't depend on trial order.
- **Memo:** only VAL logits are kept, in `cache/proxy_logits/<hash>.pt` (written atomically); head weights are never saved. The memo is shared by TPE and random.
- GPU training is not bit-deterministic (e.g. adaptive-avg-pool backward uses atomics). The first result computed for a head is reused by both samplers, so they score identical heads identically.

**D27. Proxy check.**
- **Configs:** 5 fixed configs (positions 9, 16, 16, 23, 33; two share position 16 to test within-position ranking).
- **Proxy score:** the head's argmax val accuracy on the full 5k val.
- **Full score:** the 05 recipe, 60 epochs, augmented 45k train. All 5 heads are trained in one pass of the frozen backbone (`MultiProbeVGG`), which is equivalent to separate runs because the heads are independent. The score is each head's best-epoch val accuracy.
- **Gate:** Spearman ρ < 0.6 → exit 1, and NAS does not run.

**D28. Head blocks** (`models/head_builder.py`).
- **Standard conv:** Conv3×3 (no bias) + BN + ReLU.
- **Depthwise-separable conv:** DW3×3 + PW1×1 (no bias) + BN + ReLU.
- **Channels:** stride 1; every conv outputs C × reduction channels.
- **FC:** input = channels × pooling²; hidden FC uses ReLU.
- Adaptive pooling larger than the map (e.g. 4×4 on 2×2) is allowed: it is wasteful, and the FLOPs/params objectives penalise it.
- The hand-placed head is `DEFAULT_HEAD` (no conv, 4×4 pool, hidden 256), the configurable twin of the old A3 (identical FLOPs and params).

## NAS (Step 5)

**D29. Duplicate positions.**
Positions are sampled per exit and then sorted. A repeated position keeps the head of the lower exit index, so the effective number of exits can be smaller than `n_exits`. The effective config is logged.

**D30. Accuracy objective.**
- **Final band off.** The objective uses VAL_SEL accuracy with the alpha final band turned off (exit-only accuracy).
  - The final band is calibrated to flag the same alpha·N val images for every architecture, so it is a detection operating point, not an architecture property.
  - Including it would subtract a near-constant amount and, with the 1-point margin, could make the constraint unsatisfiable for reasons unrelated to the exits.
  - The deployed accuracy (with the band) and the flag rate are logged for every trial.
- **FLOPs** are the same with and without the band (flagged images cost the same as the final exit).
- **Constraint:** measured like-for-like against the backbone's accuracy **on VAL_SEL** (not the full-val 92.64 %).

**D31. Hypervolume.**
- **Objectives in minimisation form:** (100 − acc, FLOPs / F_base, head params / 1e6), feasible trials only.
- **Reference point (fixed in code before running):** (100 − (backbone VAL_SEL acc − 1), 1.1, 1.1 × 3 × the largest single-head params in the space / 1e6).
- **Implementation:** exact HV in `utils/hypervolume.py`, tested against Optuna's internal implementation and a Monte-Carlo estimate.

**D32. What gets picked.**
The pick is taken from the **TPE** Pareto front (the proposed method); the random sampler's pick is reported for comparison. If no TPE Pareto point is within 1.0 point of the backbone (VAL_SEL), J4 fails and the chain stops.

**D34. Rules after the search.**
- **J5:** uses the rule chosen by NAS (`margin_band` or `maxlogit_band`).
- **J6:** the energy ablation on the same faults.
- **J7:** runs the paired analysis on the NAS model only if both campaigns finished.

## Follow-up after overnight 2 (2026-10-15 afternoon)

**D39. Checks compare integer counts, not float percentages.**
- **The bug:** J4's `nas_model` check failed falsely. 92.26999879 % (float32 mean) vs 92.27 % (exact) differed by 1.2e-6 against a 1e-6 tolerance, although both were 9,227 / 10,000 correct.
- **The fix:**
  - `run_check.py` now compares integer correct counts. `to_count(pct, n)` recovers the count and asserts the percentage lies within 1e-3 samples of an integer, so a real difference can never be rounded away.
  - 01, 07 and 11 now also store the integer counts.
  - The cache check (11 + `run_check cache`) moved from a 0.05-point tolerance to exact integer equality (4,632 = 4,632 of 5,000).
- **Audit of the remaining comparisons:** none needed changes.
  - **Thresholds:** test acc ≥ 90 %, head val acc > 60 %, flag rate ±1 pt of alpha, ρ ≥ 0.6, `acc_sel ≥ acc_min`.
  - **Integer counts:** SDC, flags, runs, mismatches.
  - **Checksums.**
  - **`10_analyze` replay:** checks `value == logged old value`. That's an intentional exact identity (a float32 value round-trips JSON exactly).
- **NAS feasibility margin:** the closest any TPE or random trial came to the boundary is 0.04 points = 1 image of 2,500 (hand-placed: 17 images), so float rounding could not have flipped feasibility.
- The J4 check was re-run on J4's existing outputs (no retraining) and passed.

**D40. Menu heads in the proxy.**
`cfg_key` accepts menu names ("A1".."A5") so the 16/A4 + 23/A3 model can be proxy-scored the same way as NAS trials. Keys for config dicts are byte-identical to before, so existing memo entries are unchanged.

**D41. Extra sampler seeds (TPE 11, 21; random 12, 22).**
- **Same everything else:** same budget (250), same proxy (base seed 42) and **the same memoised proxy heads**, the same VAL_CAL/VAL_SEL, and the same reference point (`17_hv_seeds.py` imports it from `14_pick_nas.py`).
- **So the spread is sampler variance only.** It does not include proxy-training noise.
- **Outputs:** `trials_<sampler>_s<seed>.csv`; the original `trials_tpe.csv` / `trials_random.csv` are never touched.
- **The chosen model is not changed** by these runs.
- **Scheduling:** lowest priority (`--nice=1000`), after J7.

**D42. Best 2-exit point (analysis only).**
Taken from the Pareto front of the feasible TPE trials whose effective exit count is 2, with the same pick rule (within 1 point of the backbone on VAL_SEL, lowest FLOPs, then fewer params). It is reported next to the chosen model; it does not replace it.

**Limitation (D26 addendum).**
The proxy memo key doesn't include the backbone checkpoint. That's correct here, because the backbone is fixed since Step 1b. If the backbone were ever retrained, `cache/proxy_logits/` must be deleted first.

---

# NAS variant study (early_exit_new_3, 2026-10-22)

## Source protection

**N1. The source stays read-only.** It's enforced at six levels:
- **Git:** `early_exit_new_3` is a `git clone` of `early_exit_new_2`, tagged `v1-from-new2`, and the push URL is set to `DISABLED-read-only-source`.
- **Writes:** `safe_path` allows writes only under `early_exit_new_3/`, and a test checks that writes under the source raise.
- **Checkpoints:** `ckpt_path` returns a source checkpoint only for reading. Saving to it is refused (tested).
- **Feature cache:** read through `CACHE_DIR = early_exit_new_2/cache` with `mmap_mode="r"`.
- **Proxy memo:** copied (1,406 + 43 files, sha256-verified). New entries go only to `early_exit_new_3/cache/proxy_logits`.
- **Legacy job files** pointing at the source were removed from the new repo. Jobs use the source venv read-only.
- **Manifest:** `tools/source_manifest.py verify` runs at the end of every job; any change fails the job.

## Reuse and search variants

**N2. A1 and A3 reuse the source's runs.**
- TPE seeds 1/11/21: `outputs/nas/trials_tpe*.csv`. Random seeds 2/12/22: `trials_random*.csv`.
- Same evaluator, proxy, memo, split and reference point.
- **Check:** the summary reproduces the source's hypervolume exactly (TPE 7.038 ± 0.453).

**N3. NSGA-II** (A2): `optuna.samplers.NSGAIISampler(population_size=50)` with the same `constraints_func`, sampler seeds 3/13/23 (250 trials = 5 generations). For conditional or dynamic parameters outside its intersection search space, Optuna falls back to independent sampling (documented Optuna behaviour).

**N4. Exact exit counts** (C variants): a single categorical over all sorted position combinations (17 / 136 / 680 choices), so "exactly K exits" holds. B3 fixes positions 16 and 23. Variants with 1–3 exits keep the source's per-exit sampling with sorting and dedupe.

**N5. Menu space** (B1): heads `A1..A5` exactly as in `models/exit_heads.py`. The proxy memo keys them separately (`p<pos>_A<k>`).

## Fault-aware search (E)

**N6. Head bank and probe.**
- **Head bank:** in-memory heads, retrained with the same seed and recipe as the memo, never written to disk.
- **Which logits:** accuracy, FLOPs and params come from the **memo logits**, so they're comparable across variants. The probe uses the bank heads' weights; tiny GPU non-determinism between the two is possible.
- **Probe setup:** golden routing uses the bank heads' clean logits with the deployed rule (α = 1 %); clean-correct is computed on VAL_SEL.
- **Fault count:** N single exponent faults (bits 23–30) spread uniformly over the head weights; fault i uses seed 777000+i in every trial.
  - NaN/Inf faults are rejected and redrawn (at most 50).
  - N = 40, or 20 if the first 5 trials (bank training included) average more than 20 s. The choice is recorded in `outputs/nas_variants/probe_faults.txt`.

**N7. E2 objectives (revised 2026-10-22, replaces the first version).**
E2 has **3 objectives**: maximise accuracy, minimise expected FLOPs, minimise exposure (= faultable head weights × head_fault_sdc / 1e6).
- head_fault_sdc is **not** a separate objective in E2; it only enters through the exposure.
- E1 is unchanged: (accuracy, FLOPs, head params, head_fault_sdc).
- Implemented in `objective_spec()` in `scripts/32_nas_variants.py` (tested in `tests/test_variants.py`).

**N8. Hypervolumes for E.**
- **Common HV:** every variant, E included, gets the common 3-objective HV (accuracy, FLOPs, params) on the shared reference point. That's the comparison basis in T1.
- **Own HV**, reported separately:
  - **E1:** 4 objectives; reference = common reference + head_fault_sdc 20 %.
  - **E2:** its own 3 objectives (accuracy error, FLOPs/F_base, exposure); reference = (common accuracy, common FLOPs, params bound × 20 % SDC = 9.312 × 20 = 186.2).
- **Computation:** `optuna._hypervolume.compute_hypervolume`, which matches our `hv3d` exactly in 3-D (tested).

**N9. Probe validation.**
- **Configs:** 10 configs spread evenly by probe SDC over the feasible 4-objective E1 Pareto front (union of seeds).
- **Comparison:** probe (N faults, seeds 777000+) vs a 200-fault head campaign with different seeds (991000+).
- **Statistics:** Spearman and Kendall.

## Proxy fidelity and picks

**N10. Proxy fidelity.**
- **Configs:** 20 = the 17 positions plus 16, 23 and 33 again; heads drawn with `random.Random(2026)` from the 156 configs.
- **Full training:** 60 epochs, 05 recipe; all 20 heads in one frozen-backbone pass; best epoch per head.
- **Pick stability:** the source TPE seed-1 feasible Pareto front is re-scored with 4- and 16-epoch proxies and the pick rule re-applied. Only the 8-epoch front is re-scored; this is a stated limitation.

**N11. Picks.**
- **Pool:** union of each variant's seeds, duplicates removed.
- **"Best A variant":** highest mean final 3-objective HV.
- **P3:** E1 4-objective Pareto points with acc ≥ acc_min and saved ≥ 39.81 %, the 16/23 model's saving under the same proxy scoring (source `results/nas/extra_val_table.csv`). Lowest head_fault_sdc wins; ties go to lower FLOPs.
- **References:** every pick and reference design gets a recomputed probe value.

## Statistics and jobs

**N12. Phase 0b statistics.**
- **Pairing:** by (k, run). Fault seeds are identical; the fault lists are identical in 784/800 (G1), 782/800 (G2) and 800/800 (G3) runs, differing only through each model's own NaN/Inf rejections.
- **Sensitivity check:** also run on identical-fault pairs only.
- **Rates:** each on the model's own clean-correct subset (9,125 vs 9,127 images).
- **HEADS:** Mann-Whitney U.
- **Holm correction:** over all (location, k) tests per metric.

**N13. Job chain.**
- **Order:** strict `afterok` chain (J0 → J6), plus the summary J7 `afterany` J6. With MaxJobs = 1 this is sequential anyway.
- **Gaps:** if a check fails, the remaining searches stay pending, by design. J7 then also waits; cancel it or resubmit.

**N14. GPU and Phase 0 location.**
- **GPU:** full A100 as instructed (`--gres=gpu:a100:1`). At submission all 6 were busy (jobs with 12–48 h limits plus a queued chain of eight 20-hour A100 jobs). The free MIG slice (3g.20gb) was **not** used without approval.
- **Phase 0a–0c:** computed on the login node; CPU only, reading finished CSVs, no training or GPU.

## Overnight 3 (2026-10-22 evening)

**N15. GPU: 20 GB MIG slice.**
- **Smoke test** (job 47770): `--gres=gpu:a100_3g.20gb:1` (`NVIDIA A100-SXM4-40GB MIG 3g.20gb`, 21.1 GB visible), with the **full feature cache preloaded** (all 17 positions, train + val).
  - **Peak memory:** 15.57 GB (NSGA-II) and 16.41 GB (E1 with probe and head bank), under the 18 GB limit.
  - **Cache mode:** the feature cache stays on the GPU. The pinned-CPU fallback was **not** needed, so arithmetic is unchanged (FP32, TF32 off).
- **Time limits:** max(measured per-trial time × trials × 1.5, 2 × old limit), so the 2× minimum dominated every job. J0 1:20, J1 4:00, J2 12:00, J3 5:00, J4 7:00, J5 6:00, J6 3:00, J7 3:00.
  - Measured on the slice: about 4 s per newly trained head; E1 about 9.4 s per trial in the first trials.
- **Expected GPU use:** about 10–11 h, under the 24 h budget, so D/B/C were **not** reduced.

**N16. J0 runs Phase 0d only.** 0a–0c were already computed on the login node (CPU only, finished CSVs). J0's check still verifies all Phase 0 files.

**N17. `34_proxy_fidelity`** frees the feature cache before the 60-epoch training, and each re-scoring evaluator before the next one, so peak GPU memory stays at the single-cache level.

**N18. Monitor.**
- **Where:** tmux session `nasvar_monitor` on the login node (CPU only), running `scripts/monitor_loop.sh`.
- **Every 15 min:** appends `check_status.sh` output and `df -h /home1` to `logs/monitor.log`.
- **On failure:** if a job ends FAILED / TIMEOUT / CANCELLED, or its log shows a failed check, it writes the job and log path to `reports/FAILURE.md` once. No resubmission.
- The cancelled job table from earlier today was renamed to `logs/slurm/jobs_variants_cancelled_47755-47762.tsv`.

**N19. Optuna deprecation warning.** Optuna 5 warns that `constraints_func` is deprecated (it will be removed in v7); it still works in 5.0.0. Behaviour is unchanged, so the API was kept for comparability with the source runs.

## Paper experiments (2026-10-23): everything except HLS

**P1. Scope.** E3 (INT8 faults) included; C4 (longer NSGA-II) skipped. HLS (D1) is deferred to the end. GPU: full A100 (`--gres=gpu:a100:1`); 4 of 6 were free at submission. Order is strict `afterok` (MaxJobs = 1); K8 (evaluation + tables) is `afterany` K7.

**P2. Store logits, evaluate rules offline.**
- `40_campaign_store.py` saves the float32 logits of every exit for every fault run on the **whole** split. `41_offline_eval.py` evaluates all rules on the identical faults.
- **Seeds:** the same fault seeds as the source campaign; val runs add 50,000,000.
- **Rejection:** a fault is rejected when it makes logits non-finite on any of the 10,000 test images. The source used only its clean-correct subset, so a few runs can differ from the source; only identical-fault runs are compared (P5).

**P3. Rules (`utils/rules2.py`).**
- **Families:** plain DNN, SDN-style max-softmax (no detection), msp / entropy / energy / margin with a low-tail final gate, energy / maxlogit / margin bands, margin multi-band (min logit + logit sum), and margin_band_agree. The agree rule exits only if the next exit agrees, so it pays for the next exit's FLOPs.
- **Calibration:** all rules use the same accuracy-aware threshold calibration (slack 0.5) on the full clean val set.
- **α sweep:** {0.25, 0.5, 1, 2, 5, 10} %.
- **Ablations (NAS model):** band upper-only, lower-only or none; slack 0 or 1.

**P4. MENDNet (DAC 2024).**
- **Pseudocode deviation (M3):** in Algorithm 1 as printed, `continue` makes the fault branch unreachable. We implement the stated intent: if no exit passes τ by the final exit, the sample is flagged and given the lowest-uncertainty label.
- **Calibrations:**
  - *clean:* τ = α-percentile of max-over-exits −E on clean val; γ from a grid maximising clean val accuracy.
  - *faulty_val:* the paper's (γ, τ) grid search for the best average accuracy under faults, run on **validation** exponent-fault runs (10 runs per location × k), never on test.
- **MENDNet's own fault model:** sign-bit flips of FR = 10/30/50 % of one group, 10 runs each.
  - FP1: final classifier Linear layers.
  - FP2: exit-2 head (exit-1 head if there is a single exit).
  - FP3: first conv after exit 1.
  - FP4: first conv.

**P5. Reproduction check (pre-registered tolerance).**
- **Method:** margin_band with the source calibration, applied to the stored logits.
- **Pass criteria:**
  - the clean-correct subset size equals the source;
  - at least 95 % of runs drew identical faults;
  - at least 99 % of identical-fault runs give exactly the same SDC and flag counts;
  - the total absolute count difference is at most 0.01 % of evaluated images.
- **Why a tolerance:** the batch composition differs (whole split vs subset), so float differences at the threshold boundary are possible.

**P6. Metric definitions (41).**
- **Per-rule subset:** SDC / flag / masked are measured on each rule's own clean-correct subset, as in the source.
- **Image level:** detection ratio = flagged / (flagged + SDC). Precision = flagged images that are corrupted (non-finite or argmax changed at any exit) / all flagged images.
- **M0 and run classes:** M0 = the unprotected early-exit model (margin thresholds, no band, no final gate). A run is **critical** if M0 all-image accuracy drops by ≥ 1.0 point, **benign** if by < 0.1 point.
- **Run detection:** flag rate > 3 × clean FAR, or binomial P(X ≥ flags | N, FAR) < 0.01.
- **Latency:** the first non-overlapping window of 100 / 500 test images with a significant flag count.

**P7. INT8 model (E3).**
- **Quantisation:** weight-only, symmetric per-tensor, codes in [−127, 127]; activations FP32.
- **Faults:** one bit (0–7) of the two's-complement code per faulty weight, k = 1..8.
- **Calibration:** on the INT8 model's own clean val logits.

**P8. Top-5 selection (C1, "unstable pick").**
- **Candidates:** the 5 lowest-FLOPs feasible Pareto points of the three TPE seeds, plus s1 trials 94 and 239. Each is fully trained (60 epochs, seed 42). s1_t94 is the source `ee_nas.pth`, which used the same recipe and seed, so it is not retrained.
- **Pick rule (unchanged, VAL_SEL only):** lowest FLOPs with accuracy ≥ backbone − 1.0; ties go to fewer head params.
- **Seeds:** the final configuration is retrained with seeds 1 and 2; their campaigns use 20 runs per (location, k) to save time.

**P9. P3 / P4.** Trained with 05 from `picks.json` (`cand_p3.json`, `cand_p4.json`). The separate 06/07 steps are not run: 41 calibrates on clean val and evaluates clean test for every rule.

**P10. Latency (43).**
- **Setup:** batch 1, margin_band α 1 %, 50 warm-up images. GPU: 2,000 images with a synchronize per image. CPU: 500 images, 8 threads.
- **Modes:** plain VGG, all exits computed, and real early stopping.
- **Check:** at least 99.5 % prediction agreement with the offline (batch-1000) routing, since batch-size-dependent float differences are possible.

**P11. Loading stored logits.** Stored-logit `.pt` files are read through `utils/logit_store.py`. The existing guard test (no `torch.load` in scripts, so checkpoints always go through `load_checkpoint`) is unchanged.

**P12. K1 disk-full failure and the tmux driver (2026-10-23 evening).**
- **What happened:** K1 (47789) failed at 17:40 with `OSError: 30000000 requested and 18612192 written` while saving `old1623/exponent_test_G3_k8.npy`. The shared 4 TB `/home1` had briefly filled up because of other users; our whole work folder is about 8 GB, and the filesystem had 50 GB free again by 19:15.
- **Before the failure:** the NAS campaign and its reproduction check had passed (3,200/3,200 runs identical, 0 count difference).
- **Fix:**
  - `40_campaign_store.py` now writes `.npy` files atomically (`.part`, fsync, rename), so an interrupted write can't leave a truncated file that the resume would skip.
  - The one truncated file was deleted; the campaign resumes from it.
- **Driver (at your request: "use tmux and complete the task"):** the dependency chain (47790–47795, stuck at DependencyNeverSatisfied) was cancelled and replaced by `scripts/paper_driver.sh` in tmux session `paper_driver`.
  - It submits K1..K8 one at a time.
  - It resubmits **only** on disk-full errors, after ≥ 30 GB is free again (max 3 retries).
  - Any other failure or failed check still stops the chain and is written to `reports/FAILURE.md`.
  - Job table: `logs/slurm/jobs_paper2.tsv`.

**P13. Move to early_exit_final, and the real cause of the K1 failures (2026-10-24).**
- **Move:** this project now lives in `/home1/jtt/early_exit_final/stage2_paper`; its source (`early_exit_new_2`) is `../stage1_base_nas`, CIFAR-10 is `../data`, the venv is `../.venv`. Only hard-coded paths changed (code, scripts, sbatch, tools); no behaviour change. Manifest verify: 1534 files, 0 changed. Tests: 680 passed.
- **Correction to P12:** K1 (47789 and 47797) failed with *Disk quota exceeded* because user `jtt` reached its own quota (~50 GB), not because other users filled `/home1` (`df` showed 49 GB free while no file could be created). `need_disk` checks `df /home1` and cannot see this quota. Freed 4.9 GB (pip cache) on 2026-10-24; the remaining chain needs roughly 15–20 GB more.

**P14. Large static data on /scratch (2026-10-24).**
- **Why:** user `jtt` has ~50 GB on `/home1` and ~10 GB on `/scratch` (both per-user quotas, invisible to `df`).
- **Moved to `/scratch/jtt/early_exit_final/` (checksum-verified, linked back at the same path):** `../data` (CIFAR-10), `../stage0_original/checkpoints`, `cache/` (proxy-logit memo), `results/campaigns/nas/` (finished). Paths in code are unchanged.
- **Stays in home:** code, git, reports, result tables/figures, all stage-1/2 checkpoints, `results/campaigns/old1623` and every new campaign (scratch has only ~2.7 GB left), stage-1 `cache/` (15.4 GB, does not fit on scratch).
- **`safe_path`:** also accepts this stage's scratch mirror (`SCRATCH_ROOT`); stage 1's scratch folder and anything else stays refused (new test).

**P15. Quota-aware space checks (2026-10-24).**
- `need_disk` (scripts/slurm/common.sh) and `free_gb` (paper_driver.sh) now measure 50 GB minus `du /home1/jtt` instead of `df /home1`. The thresholds per job are unchanged (K1 20, K2 15, K5 25, K6 10 GB).
- The driver also treats "Disk quota exceeded" as disk-full.
- Effect with ~21 GB left: K1–K3 can run; K5 refuses to start (and the chain stops with reports/FAILURE.md) unless more quota is available.

**P16. Stage-1 feature cache deleted to run K5–K8 (2026-10-25, user's choice).**
- **Why:** K5 refused to start (19 GB left, needs 25); K5+K6 write ~20 GB. The alternative was more quota from the admin.
- **What:** the 40 `../stage1_base_nas/cache/*.npy` files (15.1 GB). Kept: `cache/meta.json`, `cache/quick/`, `cache/proxy_logits/`.
- **Safe because:** no K5–K8 script (40–43, checks) reads the cache; it was only in the per-job manifest verify. Their 40 manifest lines moved to `reports/REMOVED_FEATURE_CACHE.txt` (sha256 kept), and `cache/*.npy` was dropped from `tools/source_manifest.py`. Verify: 1494 files OK. Tests 653 + 681 pass.
- **Rebuild if ever needed:** stage 1 `scripts/11_cache_features.py --seed 42` (~1 min GPU), then compare against `REMOVED_FEATURE_CACHE.txt`. Needed again only for re-running NAS / proxy steps (11–13, 34).

**P17. Old checkpoint paths inside result files mapped at load time (2026-10-25).**
- **Why:** K7 (47874) failed: `clean_test.pt` of the K1 campaigns stores `ee_ckpt=/home1/jtt/early_exit_new_2/...` (written before the P13 move), and that folder no longer exists. The P13 rewrite changed code, not stored results. K8 ran anyway (afterany) and left T11 (latency) as "missing input".
- **What:** `utils/paths.current_path` maps `early_exit_new_2/` → `stage1_base_nas/` and `early_exit_new_3/` → `stage2_paper/`; `load_searched_model` applies it. Result files are left unchanged. All 5 K7 tags resolve to existing checkpoints. Test added; 682 pass.
- **Re-run:** K7, then K8 again so T11 is filled (`K8` removed from `logs/paper_driver.log.done`).
