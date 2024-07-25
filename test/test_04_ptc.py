import torch 
import numpy as np
import pytest

from qcd_ml.nn.ptc import v_PTC
from qcd_ml.nn.lptc import v_LPTC
from qcd_ml.base.operations import v_gauge_transform, link_gauge_transform

def test_v_PTC_equivariance(config_1500, psi_test, V_1500mu0_1500mu2):
    V = V_1500mu0_1500mu2
    paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
    layer = v_PTC(1, 1, paths, config_1500)

    psibar_ngt = layer.forward(torch.stack([psi_test]))[0]
    psibar_gta = v_gauge_transform(V, psibar_ngt)

    layer.U = link_gauge_transform(config_1500, V)
    psi_test_gt = v_gauge_transform(V, psi_test)

    psibar_gtb = layer.forward(torch.stack([psi_test_gt]))[0]
    
    assert torch.allclose(psibar_gtb, psibar_gta)

def test_v_LPTC_equivariance(config_1500, psi_test, V_1500mu0_1500mu2):
    V = V_1500mu0_1500mu2
    paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
    layer = v_LPTC(1, 1, paths, config_1500)

    psibar_ngt = layer.forward(torch.stack([psi_test]))[0]
    psibar_gta = v_gauge_transform(V, psibar_ngt)

    layer.U = link_gauge_transform(config_1500, V)
    psi_test_gt = v_gauge_transform(V, psi_test)

    psibar_gtb = layer.forward(torch.stack([psi_test_gt]))[0]
    
    assert torch.allclose(psibar_gtb, psibar_gta)
