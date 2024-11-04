import torch
import qcd_ml
from qcd_ml.qcd.gauge.smear import stout


def test_stout_smear_static(config_1500, config_1500_smeared_stout_rho0p1):
    alg = stout.constant_rho(0.1)
    smearer = alg(config_1500)

    U_bar = smearer(config_1500)

    assert torch.allclose(U_bar, config_1500_smeared_stout_rho0p1)
