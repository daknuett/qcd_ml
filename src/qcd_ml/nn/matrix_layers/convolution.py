r"""
Convolutions for matrix-like fields, i.e., fields that transform as

.. math::

    M(x) \rightarrow \Omega(x) M(x) \Omega(x)
"""

import torch


class LGE_Convolution(torch.nn.Module):
    r"""
    Provides a convolution for matrix-like fields, i.e., fields that transform as
    .. math::
        M(x) \rightarrow \Omega(x) M(x) \Omega(x)

    The convolution is defined as
    .. math::

        W_i(x) \rightarrow \sum_{j\mu k} \omega_{i\mu k j} U_{\mu k}(x) W_j(x+k\mu) U_{\mu k}^\dagger(x)
    
    See 10.1103/PhysRevLett.128.032003 for more details.

    We implement this convolution differently: We define a gauge transporter along an arbitrary path :math:`T_p` as
    .. math::

        (T_p(M))(x) = ((\prod\limits_{\mu_k \in p} H_{\mu_k}) M)(x)

    where
    .. math::
        (H_{\mu} M)(x) = U_{\mu}(x) M(x + \mu) U_{\mu}(x+\mu)^\dagger

    Then, the convolution is defined as
    .. math::

        W_i(x) \rightarrow \sum_{jik} \omega_{j i k} T_{p_k}(W_j)(x)
    """
    def __init__(self, n_input, n_output, path_buffers):
        super(LGE_Convolution, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.path_buffers = path_buffers

        self.weights = torch.nn.Parameter(
                torch.randn(n_input
                             , n_output
                             , len(path_buffers)
                             , dtype=torch.cdouble))

    def forward(self, features_in):
        transported = torch.stack([torch.stack([pi.m_transport(fj) for pi in self.path_buffers]) for fj in features_in])
        return torch.einsum("ikl,il...->k...", self.weights, transported)
