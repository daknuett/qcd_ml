import torch 

from qcd_ml.dirac import dirac_wilson


def test_dirac_wilson_precomputed(config_1500, psi_test, psi_Dw1500_m0p5_psitest):
    w = dirac_wilson(config_1500, -0.5)
    expect = psi_Dw1500_m0p5_psitest

    got = w(psi_test)

    assert torch.allclose(expect, got)
