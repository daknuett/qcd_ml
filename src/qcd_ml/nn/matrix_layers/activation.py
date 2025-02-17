r"""
Activation functions for matrix-like fields, i.e., fields that transform as 

.. math::
    M(x) \rightarrow \Omega(x) M(x) \Omega(x)

"""

import torch

class LGE_ReTrAct(torch.nn.Module):
    r"""
    Given an activation function ``activation`` (:math:`F`) applies
    .. math::
        W_j(x) \rightarrow F(\omega_j \mbox{Re}\mbox{Tr}(W_j(x)) \alpha_j) W_j(x)
    """

    def __init__(self, activation, n_features):
        super(LGE_ReTrAct, self).__init__()
        self.activation = activation
        self.biases = torch.nn.Parameter(torch.randn(n_features, 1, 1, 1, 1, dtype=torch.double))
        self.weights = torch.nn.Parameter(torch.randn(n_features, 1, 1, 1, 1, dtype=torch.double))

    def forward(self, features):
        re_tr = torch.einsum("...ii->...", features.real)
        prefactor = self.activation(self.weights.expand_as(re_tr) * re_tr + biases.expand_as(re_tr))

        return torch.einsum("fabcd, fabcdij->fabcdij"
                            , prefactor, features)

