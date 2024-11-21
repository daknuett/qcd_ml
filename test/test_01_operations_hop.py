import torch
import numpy as np 

from qcd_ml.base.operations import SU3_group_compose, v_gauge_transform, link_gauge_transform, m_gauge_transform
from qcd_ml.base.hop import v_hop, m_hop

def test_SU3_group_compose(config_1500, V_1500mu0_1500mu2):
    expect = V_1500mu0_1500mu2
    got = SU3_group_compose(config_1500[0], config_1500[2])

    assert torch.allclose(expect, got)


def test_U_adjoint_1500(config_1500, config_1500_adj):
    assert torch.allclose(config_1500.adjoint(), config_1500_adj)


def test_U_Udg_id_1500(config_1500):
    U = config_1500
    U_id = torch.zeros_like(config_1500)
    U_id[:, :,:,:,:] = torch.complex(torch.tensor(np.eye(3)), torch.tensor(np.zeros((3,3))))

    assert torch.allclose(SU3_group_compose(U[0].adjoint(), U[0]), U_id[0])


def test_v_gauge_transform(psi_test, config_1500, psi_1500mu0_psitest):
    psibar_got = v_gauge_transform(config_1500[0], psi_test)

    assert torch.allclose(psibar_got, psi_1500mu0_psitest)


def test_link_gauge_transform(config_1500, config_1200, config_1500_gtrans_1200mu0):
    got = link_gauge_transform(config_1500, config_1200[0])

    for mu in range(4):
        assert torch.allclose(got[mu], config_1500_gtrans_1200mu0[mu])


def test_v_hop_Uid(config_1500, psi_test):
    U_id = torch.zeros_like(config_1500)
    U_id[:, :,:,:,:] = torch.complex(torch.tensor(np.eye(3)), torch.tensor(np.zeros((3,3))))

    psi_tt = torch.zeros_like(psi_test)
    psi_tt[0,0,0,0, :,:] = 1
    psi_tt_hop_expect = torch.zeros_like(psi_test)
    psi_tt_hop_expect[0,1,0,0, :,:] = 1

    assert torch.allclose(psi_tt_hop_expect, v_hop(U_id, 1, 1, psi_tt))


def test_v_hop_switch_color(config_1500, psi_test):
    U_test = torch.zeros_like(config_1500)
    U_test[:, :,:,:,:] = torch.complex(torch.tensor(np.eye(3)), torch.tensor(np.zeros((3,3))))
    U_test[0, 0,0,0,0] = torch.complex(torch.tensor(np.array([[0,1.,0], [1,0,0], [0,0,1]])), torch.tensor(np.zeros((3,3))))

    psi_tt = torch.zeros_like(psi_test)
    psi_tt[0,0,0,0, 0,0] = 1
    psi_tt_hop_expect = torch.zeros_like(psi_test)
    psi_tt_hop_expect[1,0,0,0, 0,1] = 1

    assert torch.allclose(psi_tt_hop_expect, v_hop(U_test, 0, 1, psi_tt))


def test_v_hop_hopinv(config_1500, psi_test):
    assert torch.allclose(psi_test, v_hop(config_1500, 0, 1, v_hop(config_1500, 0, -1, psi_test)))


def test_v_hop_equivariance(config_1500, config_1200, psi_test):
    U = config_1500
    V = config_1200[0]
    psi = psi_test

    psibar_trns_first = v_hop(link_gauge_transform(U, V), 1, 1, v_gauge_transform(V, psi))
    psibar_hop_first = v_gauge_transform(V, v_hop(U, 1, 1, psi))

    assert torch.allclose(psibar_trns_first, psibar_hop_first)


def test_m_hop_equivariance(config_1500, config_1200):
    U = config_1500
    V = config_1200[0]
    M = torch.randn_like(U[0])

    M_trns_first = m_hop(link_gauge_transform(U, V), 1, 1, m_gauge_transform(V, M))
    M_hop_first = m_gauge_transfor(V, m_hop(U, 1, 1, M))

    assert torch.allclose(M_trns_first, M_hop_first)
