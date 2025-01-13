import torch

from qcd_ml.nn.matrix_layers import LGE_Convolution
from qcd_ml.base.paths import PathBuffer
from qcd_ml.base.operations import SU3_group_compose, link_gauge_transform


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
    transformed_after = SU3_group_compose(V_1500mu0_1500mu2, features_out[0])

    transformed_U = link_gauge_transform(config_1500, V_1500mu0_1500mu2)
    layer.path_buffers = [PathBuffer(transformed_U, path) for path in paths]

    features_out_gt = layer.forward(features_in)

    assert torch.allclose(transformed_after, features_out_gt[0])
