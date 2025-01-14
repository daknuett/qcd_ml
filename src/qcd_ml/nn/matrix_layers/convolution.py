r"""
Convolutions for matrix-like fields, i.e., fields that transform as

.. math::

    M(x) \rightarrow \Omega(x) M(x) \Omega(x)
"""

import torch
from ...base.paths import PathBuffer


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

    def __init__(self, n_input, n_output, paths, disable_caching=False):
        super(LGE_Convolution, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.paths = paths
        self.disable_caching = disable_caching

        # Store path buffers by link field.
        # We expect that the link field is a torch tensor. In this case
        # use id(U) as a key for the hash. This seems OK, since it is
        # recommended here:
        # https://github.com/pytorch/pytorch/issues/7733#issuecomment-390912112
        # See also the entire issue discussion
        # https://github.com/pytorch/pytorch/issues/7733.
        self.path_buffer_cache = {}

        self.weights = torch.nn.Parameter(
                torch.randn(n_input
                             , n_output
                             , len(paths)
                             , dtype=torch.cdouble))

    def clear_path_buffers(self):
        self.path_buffer_cache = {}

    def forward(self, U, features_in):
        if id(U) in self.path_buffer_cache:
            path_buffers = self.path_buffer_cache[id(U)]
        else:
            path_buffers = [PathBuffer(U, path) for path in self.paths]
            if not self.disable_caching:
                self.path_buffer_cache[id(U)] = path_buffers

        transported = torch.stack([
                torch.stack([pi.m_transport(fj) for pi in path_buffers])
                for fj in features_in])
        return torch.einsum("ikl,il...->k...", self.weights, transported)
