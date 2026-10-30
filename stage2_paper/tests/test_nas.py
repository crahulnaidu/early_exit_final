import importlib.util
from pathlib import Path

import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nas = load("13_nas_search")
pick = load("14_pick_nas")


def test_suggest_sorts_dedupes_and_drops_unused_conditionals():
    params = {"n_exits": 3, "rule": "margin_band",
              "pos_1": 23, "n_conv_1": 0, "pooling_1": 4, "hidden_fc_1": 256,
              "pos_2": 6, "n_conv_2": 1, "conv_type_2": "standard", "channel_reduction_2": 0.5, "pooling_2": 1,
              "hidden_fc_2": 0,
              "pos_3": 23, "n_conv_3": 2, "conv_type_3": "depthwise_separable", "channel_reduction_3": 0.125,
              "pooling_3": 2, "hidden_fc_3": 64}
    k, exits, rule = nas.suggest(optuna.trial.FixedTrial(params))
    assert k == 3 and rule == "margin_band"
    assert [p for p, _ in exits] == [6, 23]                         # sorted, duplicate 23 keeps exit 1's head
    assert exits[1][1] == {"n_conv": 0, "pooling": 4, "hidden_fc": 256}
    assert exits[0][1]["conv_type"] == "standard"


def test_pick_rule_lowest_flops_then_fewer_params():
    front = pd.DataFrame({"acc_sel": [91.0, 91.5, 92.0, 89.0], "avg_flops": [300, 300, 400, 100],
                          "head_params": [2000, 1000, 500, 10]})
    chosen = pick.pick(front, acc_min=90.5)
    assert chosen.avg_flops == 300 and chosen.head_params == 1000      # 89.0 % point is below the constraint


def test_reference_point_bounds_every_config():
    assert pick.max_head_params() > 1_000_000
