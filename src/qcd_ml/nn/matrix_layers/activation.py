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

    Attributes:
        activation (torch.nn.Module): The activation function to use.
        biases (torch.nn.Parameter): Learnable bias parameters.
        weights (torch.nn.Parameter): Learnable weight parameters.
    """

    def __init__(self, activation: torch.nn.Module, n_features: int) -> None:
        """Initialize the LGE_ReTrAct layer.

        Args:
            activation: The activation function (e.g., torch.nn.ReLU, torch.nn.Sigmoid).
            n_features: Number of features.
        """
        super(LGE_ReTrAct, self).__init__()
        self.activation = activation
        self.biases = torch.nn.Parameter(torch.randn(n_features, 1, 1, 1, 1, dtype=torch.double))
        self.weights = torch.nn.Parameter(torch.randn(n_features, 1, 1, 1, 1, dtype=torch.double))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the activation to the input features.

        .. math::
            W_j(x) \rightarrow F(\omega_j \mbox{Re}\mbox{Tr}(W_j(x)) \alpha_j) W_j(x)

        Args:
            features: Input features tensor of shape (n_features, ...).

        Returns:
            Activated features tensor.
        """
        re_tr = torch.einsum("...ii->...", features.real)
        prefactor = self.activation(self.weights.expand_as(re_tr) * re_tr + self.biases.expand_as(re_tr))

        return torch.einsum("fabcd, fabcdij->fabcdij"
                            , prefactor, features)
