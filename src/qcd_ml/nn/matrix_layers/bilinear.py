"""
This module contains lattice gauge equvariant bilinear layers.
"""

import torch


class LGE_Bilinear(torch.nn.Module):
    r"""
    This class provides lattice gauge equivariant bilinear layers.
    .. math::
        W_{x,i} \rightarrow \sum_{j,k} \alpha_{i,j,k} W_{x,j} W'_{x,k}

    See 10.1103/PhysRevLett.128.032003 for more details.
    """

    def __init__(self, n_input1, n_input2, n_output):
        super(LGE_Bilinear, self).__init__()
        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.n_output = n_output

        self.weights = torch.nn.Parameter(
                torch.randn(n_input1, n_input2, n_output, dtype=torch.cdouble))

    def forward(self, features_in1, features_in2):
        return torch.einsum("ijk,jabcdnr,kabcdrm->iabcdnm", self.weights, features_in1, features_in2)
