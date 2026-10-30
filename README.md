# early_exit_final

Self-contained home of the early-exit VGG16-BN / CIFAR-10 fault-tolerance project (journal paper).
Built 2026-10-24 from `early_exit_new`, `early_exit_new_2` and `early_exit_new_3` (all three deleted the same day). Nothing here reads from
those folders any more.

## Layout

| Folder | Was | Content |
|---|---|---|
| `data/` | `early_exit_new/data` | CIFAR-10 (`cifar-10-batches-py` + `cifar-10-python.tar.gz`). Read-only. |
| `.venv/` | `early_exit_new_2/.venv` | Shared venv for all stages (pytest, fvcore, tabulate, …). |
| `stage0_original/` | `early_exit_new` (minus `data/`) | The original project before the rebuild: scripts 01–09, models, utils, and its checkpoints/results. `data` is a symlink to `../data`. No git history (it never had one). Kept as an archive; stages 1–2 do not use it. |
| `stage1_base_nas/` | `early_exit_new_2` | Steps 0–6: backbone, exit heads 16/23, exit rules + calibration, fault campaigns (margin_band / energy), feature cache, Optuna NAS, NAS-model eval, extra analyses. Own git history. |
| `stage2_paper/` | `early_exit_new_3` | NAS variant study (Phase 0, variants A–F) and the paper experiment chain K1–K8. Own git history (includes stage 1's commits). Reads stage 1 **read-only**. |

Dependency direction: `stage2_paper` → `stage1_base_nas` → `data`. Never the other way round.

## Storage on /scratch

Static large data is on `/scratch/jtt/early_exit_final/` and linked back at the same paths (stage2 DECISIONS P14):
`data/`, `stage0_original/checkpoints/`, `stage2_paper/cache/`, `stage2_paper/results/campaigns/nas/`.
Scratch is usually not backed up. Quotas: ~50 GB on /home1, ~10 GB on /scratch (not shown by `df`).

## External dependency (only one)

- Python: `.venv` is built on `/home1/jtt/haripriya/env` (torch, torchvision, numpy, optuna, …; see
  `stage*/environment.txt`). Keep that env.

## Where the results are

### Stage 0 (`stage0_original/`, original project)
| What | Files |
|---|---|
| Code | `scripts/01_train_backbone.py` … `09_summarise.py`, `models/`, `faults/inject.py`, `utils/`, `dataset/` (run from this folder; reads `./data`) |
| Checkpoints | `checkpoints/vgg16_bn_cifar10_base.pth`, `supernet_heads.pth`, `ee_searched.pth`, `cached_logits/` |
| Results | `checkpoints/search_all_configs.csv`, `chosen_config.json`, `calibrated_thresholds.json`, `fault_campaign_mode_A.csv`, `fault_campaign_summary.csv`, `figure_fault_resilience.png` |

### Stage 1 (`stage1_base_nas/`)
| What | Files |
|---|---|
| Checkpoints | `checkpoints/vgg16_bn_cifar10_base.pth` (backbone), `ee_old16_23.pth` (exits 16/23), `ee_nas.pth` (NAS exits) |
| Clean accuracy, calibration | `outputs/clean_eval.{csv,json}`, `outputs/calib_*.json`, `results/clean_results_table.md`; NAS model: `outputs/nas/` |
| Fault campaigns 16/23 | `results/faults_{margin_band,energy}_{G1,G2,G3,HEADS}.csv`, `results/fault_summary.{csv,md}` |
| Fault analysis | `results/analysis_*.csv`, `results/analysis_tables.md`, `results/paired_rules.csv`, `results/figures/` |
| NAS | `outputs/nas/` (trials, Pareto fronts, hypervolume, seed repeats, chosen config), `results/nas/` (NAS-model faults, model comparison, extra analyses) |
| Feature cache | `cache/`: the 15.1 GB feature files were **deleted** 2026-10-25 to free quota (stage2 DECISIONS P16); `meta.json`, `quick/`, `proxy_logits/` kept. Rebuild with `scripts/11_cache_features.py --seed 42`; checksums in `stage2_paper/reports/REMOVED_FEATURE_CACHE.txt`. |
| Reports, decisions | `reports/`, `DECISIONS.md` |

### Stage 2 (`stage2_paper/`)
| What | Files |
|---|---|
| Phase 0 (statistics, per-bit) | `results/phase0/` (`phase0_tables.md`, CSVs, figures) |
| NAS variants A–F | `results/nas_variants/` (`variants_tables.md`, `T1_search_variants.csv`, figures), `outputs/nas_variants/` (all trials, picks) |
| Top-5 candidates (P8) | `outputs/final/cand_*.json`, `candidates.json` |
| Logit-storing campaigns | `results/campaigns/nas/` (**complete**, 100 runs, reproduction check passed 3200/3200), `results/campaigns/old1623/` (**partial**, see below) |
| Paper tables T2–T14 | `results/paper/paper_tables.md` (+ CSVs, `figures/`) |
| Reports, decisions | `reports/` (`PAPER_EXPERIMENTS_REPORT.md`, `NAS_VARIANTS_REPORT.md`, `OVERNIGHT_*`), `DECISIONS.md` (N*, P1–P12) |

## Status of the paper chain (as of 2026-10-25 18:00)

- **All jobs K1-K8 COMPLETED**, every check and manifest verify passed. Tables T2-T14 + R in
  `stage2_paper/results/paper/paper_tables.md` (CSVs and `figures/` next to it).
- Final model: s11_t229, exits 13/16/26 (`stage2_paper/checkpoints/ee_top_s11_t229.pth`).
- K7 needed one re-run (47874 -> 47882): old checkpoint paths stored in result files, fixed by DECISIONS P17.
- Not done: HLS synthesis (T12 is theoretical).

## How this folder was made

- Files were first hard-linked from the originals; after the old folders were deleted every file here is an ordinary, independent file.
- Not copied: `tmp/`, `__pycache__/`, the `.part` file above. `stage0_original` is unchanged (28 files,
  sha256-identical to `early_exit_new`).
- Path rewrite (code, scripts, sbatch, tools, venv only; `reports/`, `logs/`, `DECISIONS.md` keep the old
  paths as historical records):

  | Old | New |
  |---|---|
  | `/home1/jtt/early_exit_new/data` | `/home1/jtt/early_exit_final/data` |
  | `/home1/jtt/early_exit_new_2` | `/home1/jtt/early_exit_final/stage1_base_nas` |
  | `/home1/jtt/early_exit_new_3` | `/home1/jtt/early_exit_final/stage2_paper` |
  | `early_exit_new_2/.venv` | `early_exit_final/.venv` |

- Checks done: CIFAR-10 archive md5 = official `c58f30108f718f92721af3b95e74349a` and all 8 extracted
  files byte-identical to it; `stage2_paper/tools/source_manifest.py verify` → 1534 stage-1 files,
  0 changed/missing; file counts equal to the sources; both test suites (see commit messages).
