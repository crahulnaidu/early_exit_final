import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nv = load("32_nas_variants")
summ = load("35_variants_summary")


def test_objective_spec_per_variant():
    assert nv.objective_spec("none") == (["maximize", "minimize", "minimize"],
                                         ["acc_sel", "avg_flops", "head_params"])
    assert nv.objective_spec("sdc") == (["maximize", "minimize", "minimize", "minimize"],
                                        ["acc_sel", "avg_flops", "head_params", "head_fault_sdc"])
    d, k = nv.objective_spec("exposure")                       # E2: 3 objectives, no separate head_fault_sdc
    assert d == ["maximize", "minimize", "minimize"] and k == ["acc_sel", "avg_flops", "exposure"]


def frame():
    return pd.DataFrame({"acc_sel": [92.0, 91.0], "avg_flops": [3e8, 2e8], "head_params": [4e5, 1e5],
                         "head_fault_sdc": [0.1, 0.5], "exposure": [0.04, 0.05], "feasible": [True, True]})


def test_own_objectives_shapes():
    df = frame()
    assert summ.own_objectives("E1_sdc", df).shape == (2, 4)
    e2 = summ.own_objectives("E2_exposure", df)
    assert e2.shape == (2, 3) and np.allclose(e2[:, 2], df.exposure)


def test_hv_own_matches_reference_implementation():
    from utils.hypervolume import hv3d
    df = frame()
    ref = np.array([10.0, 1.1, 1.0])
    assert summ.hv_own("E2_exposure", df, ref) == pytest.approx(hv3d(summ.own_objectives("E2_exposure", df), ref))
