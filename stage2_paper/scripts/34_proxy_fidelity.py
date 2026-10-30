"""Phase 1F: proxy fidelity (no search).

20 head configurations spanning the space (every one of the 17 candidate positions once + positions 16, 23, 33 again;
heads drawn with a fixed seed from the 156 configurable heads). Proxy val accuracy after 4, 8 and 16 epochs on the
cached features vs the full 05 recipe (60 epochs, augmented 45k train, frozen backbone; all 20 heads trained in one
pass = MultiProbeVGG, best epoch per head). Spearman / Kendall per proxy length.
Pick stability: the source TPE run's feasible Pareto front (outputs/nas/pareto_tpe.csv, 8-epoch proxy) is re-scored
with 4- and 16-epoch proxies; the fixed pick rule is re-applied to each re-scored front.
Writes outputs/nas_variants/proxy_fidelity.csv / .json and proxy_pick_stability.csv.
"""
import argparse
import importlib.util
import itertools
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from scipy import stats

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import BACKBONE_CKPT, SRC_OUTPUT_DIR
from dataset.datamodule import get_cifar10_dataloaders
from models.backbone import build_backbone
from models.head_builder import CONV_TYPES, HIDDEN, POOLINGS, REDUCTIONS, cfg_key
from models.multi_exit import CANDIDATE_POSITIONS
from models.searched_exit_model import MultiProbeVGG
from utils.ckpt import load_checkpoint
from utils.hypervolume import pareto_mask
from utils.proxy import FeatureCache, proxy_val_logits
from utils.runmeta import output_path
from utils.seed import set_seed
from utils.train_heads import train_heads

spec = importlib.util.spec_from_file_location("nas13", root_dir / "scripts" / "13_nas_search.py")
nas13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nas13)
spec = importlib.util.spec_from_file_location("pick14", root_dir / "scripts" / "14_pick_nas.py")
pick14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pick14)

EPOCHS = (4, 8, 16)


def all_cfgs():
    out = [{"n_conv": 0, "pooling": p, "hidden_fc": h} for p in POOLINGS for h in HIDDEN]
    for n, t, r, p, h in itertools.product((1, 2), CONV_TYPES, REDUCTIONS, POOLINGS, HIDDEN):
        out.append({"n_conv": n, "conv_type": t, "channel_reduction": r, "pooling": p, "hidden_fc": h})
    return out


def configs():
    rng = random.Random(2026)
    cfgs = all_cfgs()
    positions = list(CANDIDATE_POSITIONS) + [16, 23, 33]
    return [(p, rng.choice(cfgs)) for p in positions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full-epochs", type=int, default=60)
    args = parser.parse_args()
    q = args.quick
    set_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = FeatureCache(dev, q, BACKBONE_CKPT)
    yv = cache.labels["val"].cpu()
    cfg_list = configs()[:4] if q else configs()

    rows = []
    for pos, cfg in cfg_list:
        row = {"position": pos, "cfg": json.dumps(cfg), "key": cfg_key(pos, cfg)}
        for e in EPOCHS:
            t0 = time.time()
            z, _ = proxy_val_logits(cache, pos, cfg, e, 42)
            row[f"proxy{e}_val_acc"] = 100.0 * (z.argmax(1) == yv).float().mean().item()
            row[f"proxy{e}_seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        print(f"[proxy] p{pos} {cfg} -> " + " ".join(f"{e}ep {row[f'proxy{e}_val_acc']:.2f}%" for e in EPOCHS), flush=True)

    del cache                                   # free the cached features before the full training
    torch.cuda.empty_cache()
    backbone = build_backbone()
    load_checkpoint(BACKBONE_CKPT if not q else Path(str(BACKBONE_CKPT).replace("/checkpoints/", "/checkpoints/quick/")),
                    backbone)
    model = MultiProbeVGG(backbone.features, backbone.avgpool, backbone.classifier,
                          [p for p, _ in cfg_list], [c for _, c in cfg_list]).to(dev)
    train_loader, val_loader, _ = get_cifar10_dataloaders(batch_size=128, seed=42, num_workers=4, quick=q)
    t0 = time.time()
    _, best = train_heads(model, train_loader, val_loader, dev, 2 if q else args.full_epochs,
                          log=lambda s: print(s, flush=True))
    full_minutes = (time.time() - t0) / 60
    for r, b in zip(rows, best):
        r["full_val_acc_best"] = b
    df = pd.DataFrame(rows)
    corr = {}
    for e in EPOCHS:
        rho, p = stats.spearmanr(df[f"proxy{e}_val_acc"], df.full_val_acc_best)
        tau, _ = stats.kendalltau(df[f"proxy{e}_val_acc"], df.full_val_acc_best)
        corr[f"{e}"] = {"spearman_rho": float(rho), "spearman_p": float(p), "kendall_tau": float(tau),
                        "mean_seconds_per_head": float(df[f"proxy{e}_seconds"].mean())}
    df.to_csv(output_path("proxy_fidelity.csv", q, "nas_variants"), index=False)

    # Pick stability on the source TPE front
    front = pd.read_csv(SRC_OUTPUT_DIR / "nas" / "pareto_tpe.csv")
    if q:
        front = front.head(3)
    with open(SRC_OUTPUT_DIR / "nas" / "meta_tpe.json") as f:
        acc_min = json.load(f)["acc_min"]
    stab = []
    for e in EPOCHS:
        ev = nas13.Evaluator(SimpleNamespace(quick=q, proxy_epochs=e, seed=42, alpha=1.0), dev)
        res = []
        for r in front.itertuples():
            out = ev(list(zip(json.loads(r.positions), json.loads(r.cfgs))), r.rule)
            res.append({"proxy_epochs": e, "source_trial": r.trial, **out})
        rf = pd.DataFrame(res)
        feas = rf[rf.feasible]
        fr = feas[pareto_mask(pick14.objectives(feas))] if len(feas) else feas
        chosen = pick14.pick(fr, acc_min) if len(fr) else None
        for x in res:
            x["picked"] = chosen is not None and x["source_trial"] == chosen.source_trial
        stab += res
        del ev
        torch.cuda.empty_cache()
        print(f"[pick @ {e} epochs] source trial {None if chosen is None else int(chosen.source_trial)}", flush=True)
    stab = pd.DataFrame(stab)
    stab.to_csv(output_path("proxy_pick_stability.csv", q, "nas_variants"), index=False)
    picks = {str(e): (stab[(stab.proxy_epochs == e) & stab.picked].source_trial.tolist() or [None])[0] for e in EPOCHS}
    out = {"n_configs": len(df), "full_epochs": 2 if q else args.full_epochs, "full_training_minutes": round(full_minutes, 2),
           "correlation_vs_full": corr, "pick_source_trial_by_proxy_epochs": picks,
           "pick_changes_vs_8_epochs": {e: picks[e] != picks["8"] for e in picks}}
    with open(output_path("proxy_fidelity.json", q, "nas_variants"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
