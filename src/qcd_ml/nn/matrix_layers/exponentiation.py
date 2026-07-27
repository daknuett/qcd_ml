"""
Exponentiation layers for matrix-like fields.
"""
import torch
from typing import Literal


class LGE_Exp(torch.nn.Module):
    r"""
    Provides an exponentiation layer for matrix-like fields acting on
    gauge links, i.e.,

    .. math::
        U_\mu(x) \rightarrow \exp\left(\sum\limits_j \beta_{\mu,i} W_{i}(x)\right) U_\mu(x)
    """

    def __init__(self, n_features_in: int, matrix_mode: Literal['a', 'h', 'ah'] = 'a') -> None:
        """Initialize the LGE_Exp layer.

        Args:
            n_features_in: Number of input features.
            matrix_mode: Type of anti-hermitian matrix used in the exponent.
                - 'a': exp(anti-hermitian)
                - 'h': exp(i * hermitian)
                - 'ah': exp(anti-hermitian + i * hermitian)
        """
        super(LGE_Exp, self).__init__()
        self.matrix_mode = matrix_mode
        if matrix_mode == "a" or matrix_mode == "ah":
            self.ah_weights = torch.nn.Parameter(torch.randn(4, n_features_in, dtype=torch.double))
        if matrix_mode == "h" or matrix_mode == "ah":
            self.h_weights = torch.nn.Parameter(torch.randn(4, n_features_in, dtype=torch.double))
        if matrix_mode not in ["a", "h", "ah"]:
            raise ValueError("invalid matrix_mode. The possibilities are:\n"
                            "a -> exp(anti-hermitian)\n"
                            "h -> exp(i * hermitian)\n"
                            "ah -> exp(anti-hermitian + i * hermitian)")

    def forward(self, U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        transform_matrix = torch.zeros(U.shape, dtype=torch.cdouble)
        W_adj = W.adjoint()

        if self.matrix_mode == "a" or self.matrix_mode == "ah":
            anti_hermitian = W - W_adj
            traceless_anti_hermitian = anti_hermitian - torch.einsum("itxyzaa,bc->itxyzbc", anti_hermitian, torch.eye(3)) / 3

            transform_matrix += torch.matrix_exp(torch.einsum("ui,i...->u...",
                                                              1j * self.ah_weights, -1j * traceless_anti_hermitian))

        if self.matrix_mode == "h" or self.matrix_mode == "ah":
            hermitian = W + W_adj
            traceless_hermitian = hermitian - torch.einsum("itxyzaa,bc->itxyzbc", hermitian, torch.eye(3)) / 3

            transform_matrix += torch.matrix_exp(torch.einsum("ui,i...->u...",
                                                              1j * self.h_weights, -1j * traceless_hermitian))

        return torch.einsum("uabcdij,uabcdjk->uabcdik", transform_matrix, U)
