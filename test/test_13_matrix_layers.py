import torch

from qcd_ml.nn.matrix_layers import LGE_Convolution, LGE_Bilinear
from qcd_ml.base.paths import PathBuffer
from qcd_ml.base.operations import m_gauge_transform, link_gauge_transform


def test_LGE_Convolution_equivariance(config_1500, V_1500mu0_1500mu2):
    n_input = 1
    n_output = 1

    paths = ([[]]
             + [[(mu, 1)] for mu in range(4)]
             + [[(mu, -1)] for mu in range(4)])
    path_buffers = [PathBuffer(config_1500, path) for path in paths]

    layer = LGE_Convolution(n_input, n_output, path_buffers)

    features_in = torch.randn(1, 8,8,8,16, 3,3, dtype=torch.cdouble)
    features_out = layer.forward(features_in)
    transformed_after = m_gauge_transform(V_1500mu0_1500mu2, features_out[0])

    transformed_U = link_gauge_transform(config_1500, V_1500mu0_1500mu2)
    layer.path_buffers = [PathBuffer(transformed_U, path) for path in paths]

    features_out_gt = layer.forward(torch.stack([m_gauge_transform(V_1500mu0_1500mu2, features_in[0])]))

    assert torch.allclose(transformed_after, features_out_gt[0])


def test_LGE_Bilinear_equivariance(config_1500, V_1500mu0_1500mu2):
    n_input1 = 2,
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
