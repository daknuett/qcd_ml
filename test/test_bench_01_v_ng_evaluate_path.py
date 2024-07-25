import pytest
import torch
from qcd_ml.base.paths import v_ng_evaluate_path, slow_v_ng_evaluate_path


def test_v_ng_evaluate_path(benchmark, psi_test):
    psi_test = torch.randn_like(psi_test)

    path = [(0, 1), (1, -2), (0, 3)]
    expect = slow_v_ng_evaluate_path(path, psi_test)
    got = benchmark(v_ng_evaluate_path, path, psi_test)

    assert torch.allclose(expect, got)

def test_slow_v_ng_evaluate_path(benchmark, psi_test):
    psi_test = torch.randn_like(psi_test)

    path = [(0, 1), (1, -2), (0, 3)]
    expect = v_ng_evaluate_path(path, psi_test)
    got = benchmark(slow_v_ng_evaluate_path, path, psi_test)

    assert torch.allclose(expect, got)
