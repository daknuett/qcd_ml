import pytest
import torch

from qcd_ml.nn.matrix_layers import LGE_Convolution, LGE_Bilinear, LGE_ReTrAct, LGE_Exp
from qcd_ml.nn.matrix_layers.bilinear import LGE_BilinearLM, Apply_LGE_Bilinear
from qcd_ml.base.paths import PathBuffer
from qcd_ml.base.operations import m_gauge_transform, link_gauge_transform


def test_LGE_Convolution_equivariance(config_1500, V_1500mu0_1500mu2):
    n_input = 1
    n_output = 1

    paths = ([[]]
             + [[(mu, 1)] for mu in range(4)]
             + [[(mu, -1)] for mu in range(4)])

    layer = LGE_Convolution(n_input, n_output, paths)

    features_in = torch.randn(1, 8,8,8,16, 3,3, dtype=torch.cdouble)
    features_out = layer.forward(config_1500, features_in)
    transformed_after = m_gauge_transform(V_1500mu0_1500mu2, features_out[0])

    transformed_U = link_gauge_transform(config_1500, V_1500mu0_1500mu2)

    features_out_gt = layer.forward(transformed_U, torch.stack([m_gauge_transform(V_1500mu0_1500mu2, features_in[0])]))

    assert torch.allclose(transformed_after, features_out_gt[0])


def test_LGE_Bilinear_equivariance(config_1500, V_1500mu0_1500mu2):
    n_input1 = 2
    n_input2 = 2
    n_output = 1

    layer = LGE_Bilinear(n_input1, n_input2, n_output)

    input1_features = torch.randn(n_input1, 8,8,8,16, 3,3, dtype=torch.cdouble)
    input2_features = torch.randn(n_input2, 8,8,8,16, 3,3, dtype=torch.cdouble)

    features_out = layer.forward(input1_features, input2_features)
    transformed_after = m_gauge_transform(V_1500mu0_1500mu2, features_out[0])

    input1_features_gt = torch.stack([m_gauge_transform(V_1500mu0_1500mu2, ifl) for ifl in input1_features])
    input2_features_gt = torch.stack([m_gauge_transform(V_1500mu0_1500mu2, ifl) for ifl in input2_features])

    features_out = layer.forward(input1_features_gt, input2_features_gt)

    assert torch.allclose(transformed_after, features_out[0])


def test_LGE_ReTrAct_equivariance(config_1500, V_1500mu0_1500mu2):
    n_input = 3
    activation = torch.nn.functional.relu

    layer = LGE_ReTrAct(activation, n_input)
    input_features = torch.randn(n_input, 8,8,8,16, 3,3, dtype=torch.cdouble)

    features_out = layer.forward(input_features)
    transformed_after = m_gauge_transform(V_1500mu0_1500mu2, features_out[0])

    transformed_U = link_gauge_transform(config_1500, V_1500mu0_1500mu2)
    features_out_gt = layer.forward(torch.stack([m_gauge_transform(V_1500mu0_1500mu2, input_features[i]) for i in range(n_input)]))
    assert torch.allclose(transformed_after, features_out_gt[0])


def test_LGE_Exp_equivariance(config_1500, V_1500mu0_1500mu2):
    n_input = 1
    layer = LGE_Exp(n_input)
    input_features = config_1500
    input_GE = torch.stack([PathBuffer(config_1500, [(0,1), (1,1), (0,-1), (1,-1)]).gauge_transport_matrix])

    features_out = layer.forward(input_features, input_GE)
    transformed_after = torch.stack(link_gauge_transform(features_out, V_1500mu0_1500mu2))

    gauge_transformed_U = link_gauge_transform(config_1500, V_1500mu0_1500mu2)
    input_features = torch.stack(gauge_transformed_U)
    input_GE = torch.stack([PathBuffer(gauge_transformed_U, [(0,1), (1,1), (0,-1), (1,-1)]).gauge_transport_matrix])
    transformed_before = layer.forward(input_features, input_GE)

    assert torch.allclose(transformed_after, transformed_before)


@pytest.mark.slow
def test_LGE_BilinearLM_autograd():
    n_input1 = 2
    n_input2 = 2
    n_output = 1

    layer1 = LGE_Bilinear(n_input1, n_input2, n_output)
    layer2 = LGE_BilinearLM(n_input1, n_input2, n_output)

    layer2.weights.data = layer1.weights.data

    input1_features = torch.randn(n_input1, 4,4,2,2, 3,3, dtype=torch.cdouble)
    input2_features = torch.randn(n_input2, 4,4,2,2, 3,3, dtype=torch.cdouble)

    features_out1 = layer1.forward(input1_features, input2_features)
    features_out2 = layer2.forward(input1_features, input2_features)

    assert torch.allclose(features_out1, features_out2)

    input1_features.requires_grad = True
    input2_features.requires_grad = True

    assert torch.autograd.gradcheck(Apply_LGE_Bilinear.apply, (input1_features, input2_features, layer2.weights))
