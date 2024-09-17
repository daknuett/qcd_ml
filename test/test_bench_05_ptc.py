import torch 
import numpy as np
import pytest

from qcd_ml.nn.ptc import v_PTC
from qcd_ml.nn.lptc import v_LPTC
from qcd_ml.base.operations import v_gauge_transform, link_gauge_transform

@pytest.mark.benchmark(group="nn.ptc", warmup=True)
def test_v_PTC(config_1500, psi_test, benchmark):
    paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
    layer = v_PTC(1, 1, paths, config_1500)

    psibar_ngt = benchmark(layer.forward, torch.stack([psi_test]))

@pytest.mark.benchmark(group="nn.ptc", warmup=True)
def test_v_LPTC(config_1500, psi_test, benchmark):
    paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
    layer = v_LPTC(1, 1, paths, config_1500)

    psibar_ngt = benchmark(layer.forward, torch.stack([psi_test]))
