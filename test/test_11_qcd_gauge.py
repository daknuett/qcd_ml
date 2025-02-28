import torch
import qcd_ml
from qcd_ml.qcd.gauge.smear import stout

import pytest


def test_stout_smear_static(config_1500, config_1500_smeared_stout_rho0p1):
    alg = stout.constant_rho(0.1)
    smearer = alg(config_1500)

    U_bar = smearer(config_1500)

    assert torch.allclose(U_bar, config_1500_smeared_stout_rho0p1)


@pytest.mark.slow
def test_stout_smear_static10(config_1500, config_1500_smeared_stout_rho0p1_10):
    alg = stout.constant_rho(0.1)
    smearer = alg(config_1500)

    U_bar = config_1500
    for _ in range(10):
        U_bar = alg(U_bar)(U_bar)

    assert torch.allclose(U_bar, config_1500_smeared_stout_rho0p1_10)
