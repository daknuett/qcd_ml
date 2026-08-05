import torch 

from qcd_ml.qcd.dirac import dirac_wilson, dirac_wilson_clover


def test_dirac_wilson_precomputed(config_1500, psi_test, psi_Dw1500_m0p5_psitest):
    w = dirac_wilson(config_1500, -0.5)
    expect = psi_Dw1500_m0p5_psitest

    got = w(psi_test)

    assert torch.allclose(expect, got)


def test_dirac_wilson_clover_precomputed(config_1500, psi_test, psi_Dwc1500_m0p5_psitest):
    w = dirac_wilson_clover(config_1500, -0.5, 1)
    expect = psi_Dwc1500_m0p5_psitest

    got = w(psi_test)

    assert torch.allclose(expect, got)


def apply_decomposed(w, v):
    result = w.apply_diag(v)
    for mu in range(4):
        result = result + w.apply_pos_hop(v, mu) + w.apply_neg_hop(v, mu)

    return result


def test_dirac_wilson_decomposition(config_1500, psi_test):
    w = dirac_wilson(config_1500, -0.5)
    expect = w(psi_test)

    got = apply_decomposed(w, psi_test)

    assert torch.allclose(expect, got)


def test_dirac_wilson_dag_decomposition(config_1500, psi_test):
    w = dirac_wilson(config_1500, -0.5, dag=True)
    expect = w(psi_test)

    got = apply_decomposed(w, psi_test)

    assert torch.allclose(expect, got)


def test_dirac_wilson_clover_decomposition(config_1500, psi_test):
    w = dirac_wilson_clover(config_1500, -0.5, 1)
    expect = w(psi_test)

    got = apply_decomposed(w, psi_test)

    assert torch.allclose(expect, got)


def test_dirac_wilson_clover_dag_decomposition(config_1500, psi_test):
    w = dirac_wilson_clover(config_1500, -0.5, 1, dag=True)
    expect = w(psi_test)

    got = apply_decomposed(w, psi_test)

    assert torch.allclose(expect, got)
