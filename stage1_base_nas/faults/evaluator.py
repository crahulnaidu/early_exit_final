"""Batched evaluation of a SearchedMultiExitVGG on a fixed image subset held on the device.

For speed, a fault in G2 only recomputes features[p_1+1 ..] from the cached clean activation at exit 1, a fault
in G3 only recomputes features[p_K+1 ..] from the cached activation at the last exit (earlier exits are
unaffected), and a HEADS fault only recomputes the heads from the cached clean features. tests/test_faults.py checks that this is bit-identical to a full forward pass.
"""
import torch

from faults.inject import location_recompute


class SubsetEvaluator:
    def __init__(self, model, images, batch_size=1000):
        self.model = model.eval()
        self.images = images
        self.batch_size = batch_size
        self.positions = list(model.positions)
        self.clean_feats, self.clean_logits = self._full(cache=True)

    def _chunks(self, n):
        return [(i, min(i + self.batch_size, n)) for i in range(0, n, self.batch_size)]

    @torch.no_grad()
    def _run(self, x, start, head_idx):
        """features[start:] on x, applying heads from head_idx on; returns (logits list, cached feats)."""
        m = self.model
        logits, feats = [], {}
        for idx in range(start, len(m.features)):
            x = m.features[idx](x)
            if head_idx < len(self.positions) and idx == self.positions[head_idx]:
                feats[idx] = x
                logits.append(m.exit_heads[head_idx](x))
                head_idx += 1
        logits.append(m.classifier(torch.flatten(m.avgpool(x), 1)))
        return logits, feats

    def _full(self, cache=False):
        outs, feats = None, {p: [] for p in self.positions}
        for a, b in self._chunks(self.images.shape[0]):
            logits, f = self._run(self.images[a:b], 0, 0)
            if outs is None:
                outs = [[] for _ in logits]
            for i, z in enumerate(logits):
                outs[i].append(z)
            if cache:
                for p in self.positions:
                    feats[p].append(f[p])
        logits = [torch.cat(o) for o in outs]
        return ({p: torch.cat(v) for p, v in feats.items()} if cache else None), logits

    @torch.no_grad()
    def logits(self, location):
        """Logits [E1..EK, final] on the subset with the model's CURRENT (possibly faulty) weights."""
        m = self.model
        if location == "HEADS":
            heads = [torch.cat([m.exit_heads[k](self.clean_feats[p][a:b])
                                for a, b in self._chunks(self.images.shape[0])])
                     for k, p in enumerate(self.positions)]
            return heads + [self.clean_logits[-1]]
        start, n_clean = location_recompute(m, location)
        if start == 0:
            return self._full()[1]
        cached = self.clean_feats[self.positions[n_clean - 1]]
        outs = None
        for a, b in self._chunks(cached.shape[0]):
            logits, _ = self._run(cached[a:b], start, n_clean)
            if outs is None:
                outs = [[] for _ in logits]
            for i, z in enumerate(logits):
                outs[i].append(z)
        return list(self.clean_logits[:n_clean]) + [torch.cat(o) for o in outs]

    @torch.no_grad()
    def full_logits(self):
        return self._full()[1]
