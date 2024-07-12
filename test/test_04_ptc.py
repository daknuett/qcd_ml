import torch 
import numpy as np

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


def test_v_PTC_reverse(config_1500, psi_test):
    paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
    layer = v_PTC(1, 1, paths, config_1500)

    idty = torch.complex(torch.tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), dtype=torch.double)
                     , torch.zeros((4,4), dtype=torch.double))

    # set weights to unitary random matrices
    for i in range(layer.n_feature_in):
        for j in range(layer.n_feature_out):
            for k in range(len(layer.paths)):
                randm = torch.randn_like(layer.weights.data[i,j,k])
                antihermitian = randm - randm.adjoint()
                unitary = torch.matrix_exp(antihermitian)

                assert torch.allclose(torch.einsum("ij,jk->ik", unitary, unitary.adjoint())
                                      , idty)

                layer.weights.data[i,j,k] = unitary
            

    psi2 = layer.forward(torch.stack([psi_test]))
    psi3 = layer.reverse(psi2)

    assert torch.allclose(psi_test, psi3)


def test_v_PTC_reverse_idty_weights(config_1500, psi_test):
    paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
    layer = v_PTC(1, 1, paths, config_1500)

    idty = torch.complex(torch.tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), dtype=torch.double)
                     , torch.zeros((4,4), dtype=torch.double))

    # set weights to unitary random matrices
    layer.weights.data[:,:,:] = idty

    psi2 = layer.forward(torch.stack([psi_test]))
    psi3 = layer.reverse(psi2)

    assert torch.allclose(psi_test, psi3[0])
