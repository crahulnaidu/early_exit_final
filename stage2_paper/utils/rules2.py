"""Generic exit rules for the method comparison (vectorised over a batch; calibrated on clean VAL only).

A standard rule = (score, band variable, final gate, extras):
  intermediate exit k: EXIT if logits finite AND score >= t_k AND [L_k <= band <= U_k] AND [extra bands] AND
                       [agree: argmax_k == argmax of the next exit]; otherwise continue
  final classifier   : gate "band" (accept if L_fin <= band <= U_fin, alpha/2 per tail), "low" (accept if
                       score >= q_fin, alpha in the low tail) or "none" (always accept: no detection)
  t_k: accuracy-aware calibration exactly as utils/calibrate.py (smallest percentile threshold such that the head is
  at most ACC_SLACK points worse than the final classifier on the exiting samples, for it and all larger thresholds).
Special rules: plain_dnn (final classifier only, no exits, no detection) and mendnet (Algorithm 1 of MENDNet,
DAC 2024: max-softmax threshold gamma + negative-energy threshold tau + fallback list; see route_mendnet).
Every route returns exit_id (0..K-1 early exit, K final, -1 flagged), pred (-1 if flagged) and mitig (the label a
mitigating system would output for every sample: pred if not flagged, otherwise the fallback label).
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch

ACC_SLACK = 0.5
MIN_COUNT = 25
BAND_PCT = (0.1, 99.9)
EXTRA_PCT_EXIT = (0.1, 99.9)
EXTRA_PCT_FINAL = (0.05, 99.95)


# ---------------- signals ----------------
def msp(z):
    return torch.softmax(z.float(), dim=-1).max(dim=-1).values


def neg_entropy(z):
    p = torch.softmax(z.float(), dim=-1)
    return (p * torch.log(p.clamp_min(1e-30))).sum(dim=-1)


def neg_energy(z):                   # -E(z) = logsumexp(z)
    return torch.logsumexp(z.float(), dim=-1)


def energy(z):
    return -neg_energy(z)


def margin(z):
    t = torch.topk(z, 2, dim=-1).values
    return t[..., 0] - t[..., 1]


def maxlogit(z):
    return z.max(dim=-1).values


def minlogit(z):
    return z.min(dim=-1).values


def logit_sum(z):
    return z.sum(dim=-1)


def finite(z):
    return torch.isfinite(z).all(dim=-1)


@dataclass
class RuleSpec:
    name: str
    kind: str = "standard"            # standard | plain | mendnet
    score: object = None
    band: object = None
    final: str = "band"               # band | low | none
    extras: list = field(default_factory=list)
    agree: bool = False
    ops: str = ""                     # hardware operations needed per exit decision (for the paper table)


RULES = {
    "plain_dnn": RuleSpec("plain_dnn", kind="plain", final="none", ops="none (no exits)"),
    "sdn_msp": RuleSpec("sdn_msp", score=msp, final="none", ops="softmax (exp, div), max"),
    "msp_low": RuleSpec("msp_low", score=msp, final="low", ops="softmax (exp, div), max"),
    "entropy_low": RuleSpec("entropy_low", score=neg_entropy, final="low", ops="softmax, log, mult, add"),
    "energy_low": RuleSpec("energy_low", score=neg_energy, final="low", ops="exp, log (logsumexp)"),
    "margin_low": RuleSpec("margin_low", score=margin, final="low", ops="comparators, 1 subtraction"),
    "energy_band": RuleSpec("energy_band", score=neg_energy, band=energy, final="band", ops="exp, log + comparators"),
    "maxlogit_band": RuleSpec("maxlogit_band", score=maxlogit, band=maxlogit, final="band", ops="comparators"),
    "margin_band": RuleSpec("margin_band", score=margin, band=maxlogit, final="band",
                            ops="comparators, 1 subtraction"),
    "margin_multiband": RuleSpec("margin_multiband", score=margin, band=maxlogit, final="band",
                                 extras=[minlogit, logit_sum], ops="comparators, 1 subtraction, 9 additions"),
    "margin_band_agree": RuleSpec("margin_band_agree", score=margin, band=maxlogit, final="band", agree=True,
                                  ops="comparators, 1 subtraction (+ next exit computed)"),
    "mendnet": RuleSpec("mendnet", kind="mendnet", ops="softmax (exp, div), max, logsumexp (exp, log)"),
}


def _np(x):
    return x.detach().double().cpu().numpy()


def _f32(v):
    return float(np.float32(v)) if math.isfinite(v) else v


def _pct(x, q):
    return [_f32(float(v)) for v in np.percentile(x, q)]


# ---------------- calibration (clean VAL) ----------------
def calibrate(rule, val_logits, y, alphas=(1.0,), slack=ACC_SLACK, band_pct=BAND_PCT, use_band=True,
              band_sides="both"):
    """Thresholds for a standard rule. use_band / band_sides ("both" | "upper" | "lower") serve the ablations."""
    spec = RULES[rule]
    if spec.kind == "plain":
        return {"rule": rule, "kind": "plain"}
    if spec.kind == "mendnet":
        raise ValueError("use calibrate_mendnet_clean / calibrate_mendnet_faulty")
    K = len(val_logits) - 1
    n = y.shape[0]
    yv = y
    final_correct = _np(val_logits[K].argmax(-1).eq(yv))
    eligible = np.ones(n, dtype=bool)
    exits = []
    for k in range(K):
        z = val_logits[k]
        s = _np(spec.score(z))
        head_correct = _np(z.argmax(-1).eq(yv))
        cand = eligible.copy()
        e = {}
        if spec.band is not None and use_band:
            b = _np(spec.band(z))
            lo, hi = _pct(b, band_pct)
            lo = lo if band_sides in ("both", "lower") else -math.inf
            hi = hi if band_sides in ("both", "upper") else math.inf
            cand &= (b >= lo) & (b <= hi)
            e.update(L=lo, U=hi)
        ex = []
        for fn in spec.extras:
            v = _np(fn(z))
            lo2, hi2 = _pct(v, EXTRA_PCT_EXIT)
            cand &= (v >= lo2) & (v <= hi2)
            ex.append([lo2, hi2])
        e["extras"] = ex
        if spec.agree:
            cand &= _np(z.argmax(-1).eq(val_logits[k + 1].argmax(-1))).astype(bool)
        grid = np.unique(np.percentile(s[cand], np.arange(0, 100.5, 0.5))) if cand.any() else np.array([])
        thresholds = list(grid) + [math.inf]
        ok = []
        for t in thresholds:
            sel = cand & (s >= t)
            ok.append(sel.sum() < MIN_COUNT or 100 * head_correct[sel].mean() >= 100 * final_correct[sel].mean() - slack)
        last_fail = max((i for i, v in enumerate(ok) if not v), default=-1)
        e["t"] = _f32(float(thresholds[last_fail + 1]))
        sel = cand & (s >= e["t"])
        e["val_exit_count"] = int(sel.sum())
        exits.append(e)
        eligible &= ~sel

    zf = val_logits[K]
    finals = {}
    for alpha in alphas:
        reach = eligible
        f = {}
        if spec.final == "none":
            f["gate"] = "none"
        elif spec.final == "band":
            b = _np((spec.band if spec.band is not None else maxlogit)(zf))[reach]
            q = 100.0 * (alpha / 100.0 * n / 2.0) / max(len(b), 1)
            if len(b) == 0 or q >= 50:
                f.update(L=math.inf, U=-math.inf)
            else:
                lo, hi = _pct(b, (q, 100.0 - q))
                lo = lo if (band_sides in ("both", "lower") and use_band) else -math.inf
                hi = hi if (band_sides in ("both", "upper") and use_band) else math.inf
                f.update(L=lo, U=hi)
            f["extras"] = [_pct(_np(fn(zf))[reach], EXTRA_PCT_FINAL) if reach.any() else [-math.inf, math.inf]
                           for fn in spec.extras]
        else:  # low
            s = _np(spec.score(zf))[reach]
            q = 100.0 * (alpha / 100.0 * n) / max(len(s), 1)
            f["q"] = math.inf if (len(s) == 0 or q >= 100) else _pct(s, (q,))[0]
        finals[str(float(alpha))] = f
    return {"rule": rule, "kind": "standard", "exits": exits, "final": finals, "slack": slack}


# ---------------- routing ----------------
def _passes_final(spec, f, z):
    fin = finite(z)
    if f.get("gate") == "none":
        return torch.ones_like(fin)
    if "q" in f:
        return fin & (spec.score(z) >= f["q"])
    b = (spec.band if spec.band is not None else maxlogit)(z)
    ok = fin & (b >= f["L"]) & (b <= f["U"])
    for fn, (lo, hi) in zip(spec.extras, f.get("extras", [])):
        v = fn(z)
        ok &= (v >= lo) & (v <= hi)
    return ok


def _fallback(spec, cal, logits):
    """Mitigation label for flagged samples: the most confident finite, in-band exit (score), else the earliest
    finite exit, else -1."""
    n = logits[0].shape[0]
    dev = logits[0].device
    best = torch.full((n,), -math.inf, device=dev)
    label = torch.full((n,), -1, dtype=torch.long, device=dev)
    earliest = torch.full((n,), -1, dtype=torch.long, device=dev)
    K = len(logits) - 1
    for k, z in enumerate(logits):
        fin = finite(z)
        am = torch.nan_to_num(z, nan=-math.inf).argmax(-1)
        earliest = torch.where((earliest < 0) & fin, am, earliest)
        s = (spec.score or maxlogit)(torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)).float()
        ok = fin.clone()
        if spec.band is not None:
            b = spec.band(torch.nan_to_num(z, nan=0.0))
            if k < K and "L" in cal["exits"][k]:
                ok &= (b >= cal["exits"][k]["L"]) & (b <= cal["exits"][k]["U"])
        better = ok & (s > best)
        best = torch.where(better, s, best)
        label = torch.where(better, am, label)
    return torch.where(label >= 0, label, earliest)


def route(cal, logits, alpha=1.0):
    """logits: [E1..EK, final] tensors [N, C]. Returns exit_id, pred, mitig."""
    rule = cal["rule"]
    spec = RULES[rule]
    K = len(logits) - 1
    n = logits[0].shape[0]
    dev = logits[0].device
    if spec.kind == "plain":
        p = torch.nan_to_num(logits[K], nan=-math.inf).argmax(-1)
        return torch.full((n,), K, dtype=torch.long, device=dev), p, p
    if spec.kind == "mendnet":
        return route_mendnet(logits, cal["gamma"], cal["tau"])
    exit_id = torch.full((n,), -1, dtype=torch.long, device=dev)
    pred = torch.full((n,), -1, dtype=torch.long, device=dev)
    undecided = torch.ones(n, dtype=torch.bool, device=dev)
    for k in range(K):
        z = logits[k]
        e = cal["exits"][k]
        ok = undecided & finite(z) & (spec.score(z) >= e["t"])
        if "L" in e:
            b = spec.band(z)
            ok &= (b >= e["L"]) & (b <= e["U"])
        for fn, (lo, hi) in zip(spec.extras, e.get("extras", [])):
            v = fn(z)
            ok &= (v >= lo) & (v <= hi)
        if spec.agree:
            ok &= z.argmax(-1).eq(torch.nan_to_num(logits[k + 1], nan=-math.inf).argmax(-1))
        exit_id[ok] = k
        pred[ok] = z.argmax(-1)[ok]
        undecided &= ~ok
    f = cal["final"][str(float(alpha))]
    ok = undecided & _passes_final(spec, f, logits[K])
    exit_id[ok] = K
    pred[ok] = logits[K].argmax(-1)[ok]
    flagged = exit_id == -1
    mitig = pred.clone()
    if flagged.any():
        fb = _fallback(spec, cal, logits)
        mitig[flagged] = fb[flagged]
    return exit_id, pred, mitig


# ---------------- MENDNet (Algorithm 1, DAC 2024) ----------------
def route_mendnet(logits, gamma, tau):
    """Vectorised Algorithm 1. Per sample, exits i = 1..N (N includes the final classifier):
      track the label of the exit with the highest negative energy (lowest uncertainty);
      if msp > gamma and -E > tau: return argmax_i (exit)
      elif -E > tau: push argmax_i to the list
      else (-E <= tau): if list empty or i <= 2: continue, else return the last pushed label
      at i == N: list empty -> return the lowest-uncertainty label with fault flag True; else return the last label.
    The printed pseudocode's `continue` would skip the i == N block (the fault branch could never run); we implement
    the stated intent (DECISIONS.md M3): at the final exit, an empty list means "fault detected".
    exit_id: the exit whose label is returned (fault-flagged samples: -1); pred: -1 for flagged; mitig: label."""
    N = len(logits)
    n = logits[0].shape[0]
    dev = logits[0].device
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    exit_id = torch.full((n,), -1, dtype=torch.long, device=dev)
    out = torch.full((n,), -1, dtype=torch.long, device=dev)
    flag = torch.zeros(n, dtype=torch.bool, device=dev)
    last = torch.full((n,), -1, dtype=torch.long, device=dev)
    last_exit = torch.full((n,), -1, dtype=torch.long, device=dev)
    min_unc = torch.full((n,), -math.inf, device=dev)
    low_label = torch.full((n,), -1, dtype=torch.long, device=dev)
    for i in range(N):
        z = torch.nan_to_num(logits[i].float(), nan=-math.inf)
        ne = torch.nan_to_num(neg_energy(logits[i]), nan=-math.inf)
        conf = torch.nan_to_num(msp(logits[i]), nan=0.0)
        am = z.argmax(-1)
        upd = ~done & (ne > min_unc)
        min_unc = torch.where(upd, ne, min_unc)
        low_label = torch.where(upd, am, low_label)
        c1 = ~done & (conf > gamma) & (ne > tau)
        out = torch.where(c1, am, out)
        exit_id = torch.where(c1, torch.full_like(exit_id, i), exit_id)
        done |= c1
        c2 = ~done & (ne > tau)
        last = torch.where(c2, am, last)
        last_exit = torch.where(c2, torch.full_like(last_exit, i), last_exit)
        c3 = ~done & ~c2 & (last >= 0) & (i + 1 > 2)
        out = torch.where(c3, last, out)
        exit_id = torch.where(c3, last_exit, exit_id)
        done |= c3
        if i == N - 1:
            rem = ~done
            empty = rem & (last < 0)
            out = torch.where(empty, low_label, out)
            flag |= empty
            ret = rem & (last >= 0)
            out = torch.where(ret, last, out)
            exit_id = torch.where(ret, last_exit, exit_id)
            done |= rem
    pred = torch.where(flag, torch.full_like(out, -1), out)
    exit_id = torch.where(flag, torch.full_like(exit_id, -1), exit_id)
    return exit_id, pred, out


def mendnet_reference(rows, gamma, tau):
    """Slow per-sample version of the same semantics (for tests). rows: list of N logit lists."""
    import math as m
    lst, low, min_unc = [], -1, -m.inf
    N = len(rows)
    for i, z in enumerate(rows, start=1):
        mx = max(z)
        ne = mx + m.log(sum(m.exp(v - mx) for v in z))
        conf = 1.0 / sum(m.exp(v - mx) for v in z)
        a = max(range(len(z)), key=lambda j: z[j])
        if ne > min_unc:
            min_unc, low = ne, a
        if conf > gamma and ne > tau:
            return i - 1, a, a
        elif ne > tau:
            lst.append((i - 1, a))
        elif not (not lst or i <= 2):
            e, lab = lst.pop()
            return e, lab, lab
        if i == N:
            if not lst:
                return -1, -1, low
            e, lab = lst.pop()
            return e, lab, lab
    raise AssertionError("unreachable")


def calibrate_mendnet_clean(val_logits, y, alpha=1.0, gammas=None):
    """Clean-only calibration (no fault knowledge): tau so that the clean val fault-flag rate is alpha
    (flag <=> every exit has -E <= tau); gamma from a grid maximising clean val accuracy (flagged -> fallback label)."""
    gammas = gammas or [round(g, 2) for g in np.arange(0.50, 0.96, 0.05)]
    ne_max = torch.stack([neg_energy(z) for z in val_logits], 0).max(0).values
    tau = _pct(_np(ne_max), (alpha,))[0]
    best = None
    for g in gammas:
        _, _, lab = route_mendnet(val_logits, g, tau)
        acc = float(lab.eq(y).float().mean())
        if best is None or acc > best[0]:
            best = (acc, g)
    return {"rule": "mendnet", "kind": "mendnet", "gamma": best[1], "tau": tau, "variant": "clean",
            "val_acc": 100 * best[0]}


def calibrate_mendnet_faulty(faulty_val_runs, y, gammas=(0.5, 0.6, 0.7, 0.8, 0.9), taus=tuple(range(0, 17, 2))):
    """As in the MENDNet paper: (gamma, tau) with the best average accuracy over fault-injected runs - here on
    VALIDATION fault runs (never test). faulty_val_runs: list of logits lists."""
    best = None
    for g in gammas:
        for t in taus:
            acc = np.mean([float(route_mendnet(r, g, float(t))[2].eq(y).float().mean()) for r in faulty_val_runs])
            if best is None or acc > best[0]:
                best = (acc, g, float(t))
    return {"rule": "mendnet", "kind": "mendnet", "gamma": best[1], "tau": best[2], "variant": "faulty_val_tuned",
            "val_faulty_acc": 100 * best[0]}
