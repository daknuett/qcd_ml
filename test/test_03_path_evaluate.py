import torch
import numpy as np 

import pytest

from qcd_ml.base.hop import v_hop
from qcd_ml.base.paths import v_evaluate_path
from qcd_ml.base.paths import v_ng_evaluate_path, slow_v_ng_evaluate_path

def test_v_evaluate_path_against_v_hop(config_1500, psi_test):
    path = [(0, 1)]
    assert torch.allclose(v_hop(config_1500, 0, 1, psi_test), v_evaluate_path(config_1500, path, psi_test))

    path = [(0, -1)]
    assert torch.allclose(v_hop(config_1500, 0, -1, psi_test), v_evaluate_path(config_1500, path, psi_test))

    path = [(0, 2)]
    hopp0 = lambda v: v_hop(config_1500, 0, 1, v)
    assert torch.allclose(hopp0(hopp0(psi_test)), v_evaluate_path(config_1500, path, psi_test))

    path = [(0, 1), (1, -2)]
    hopm1 = lambda v: v_hop(config_1500, 1, -1, v)
    assert torch.allclose(hopm1(hopm1(v_hop(config_1500, 0, 1, psi_test))), v_evaluate_path(config_1500, path, psi_test))


def test_v_ng_evaluate_path_against_slow(psi_test):
    psi_test = torch.randn_like(psi_test)

    path = [(0, 1), (1, -2)]

    assert torch.allclose(v_ng_evaluate_path(path, psi_test), slow_v_ng_evaluate_path(path, psi_test))

    path = [(0, 1), (1, -2), (0, 3)]

    assert torch.allclose(v_ng_evaluate_path(path, psi_test), slow_v_ng_evaluate_path(path, psi_test))
