import numpy as np
import pytest
import torch

from qcd_ml.base.operations import link_gauge_transform, v_gauge_transform
from qcd_ml.nn.pt import v_PT


def test_v_PTC_equivariance(config_1500, psi_test, V_1500mu0_1500mu2):
    V = V_1500mu0_1500mu2
    paths = (
        [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
    )
    layers = [v_PT([path], config_1500) for path in paths]

    psibar_ngts = [
        layer.forward(torch.stack([psi_test]))[0] for layer in layers
    ]
    psibar_gtas = [
        v_gauge_transform(V, psibar_ngt) for psibar_ngt in psibar_ngts
    ]

    for layer in layers:
        layer.gauge_transform_using_transformed(
            link_gauge_transform(config_1500, V)
        )
    psi_test_gt = v_gauge_transform(V, psi_test)

    psibar_gtbs = [
        layer.forward(torch.stack([psi_test_gt]))[0] for layer in layers
    ]

    assert all(
        torch.allclose(psibar_gtb, psibar_gta)
        for psibar_gtb, psibar_gta in zip(psibar_gtbs, psibar_gtas)
    )
