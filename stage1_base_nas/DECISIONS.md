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
