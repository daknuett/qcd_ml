import torch
import qcd_ml
from qcd_ml.qcd.gauge.smear import stout, wilson_flow, improved_flow

import pytest

def test_stout_smear_static(config_1500, config_1500_smeared_stout_rho0p1):
    smearer = stout.constant_rho(0.1)

    U_bar = smearer(config_1500)

    assert torch.allclose(U_bar, config_1500_smeared_stout_rho0p1)


@pytest.mark.slow
def test_stout_smear_static10(config_1500, config_1500_smeared_stout_rho0p1_10):
    smearer = stout.constant_rho(0.1)

    U_bar = config_1500
    for _ in range(10):
        U_bar = smearer(U_bar)

    assert torch.allclose(U_bar, config_1500_smeared_stout_rho0p1_10)


def test_wilson_flow(config_1500, config_1500_wilson_flow_eps0p1):
    flow = wilson_flow(0.1)

    U_bar = flow(config_1500)

    assert torch.allclose(U_bar, config_1500_wilson_flow_eps0p1)


@pytest.mark.slow
def test_wilson_flow10(config_1500, config_1500_wilson_flow_eps0p1_10):
    flow = wilson_flow(0.1)

    U_bar = config_1500
    for _ in range(10):
        U_bar = flow(U_bar)

    assert torch.allclose(U_bar, config_1500_wilson_flow_eps0p1_10)


def test_improved_flow(config_1500, config_1500_symanzik_flow_eps0p1):
    flow = improved_flow(0.1, -1.0 / 12.0)

    U_bar = flow(config_1500)

    assert torch.allclose(U_bar, config_1500_symanzik_flow_eps0p1)