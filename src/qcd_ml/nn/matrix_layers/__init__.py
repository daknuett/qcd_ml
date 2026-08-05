r"""
qcd_ml.nn.matrix_layers
=======================

Layers for matrix valued fields, i.e., fields that transform as
.. math::
    M(x) \rightarrow \Omega(x) M(x) \Omega(x).

Provides the following layers:

- ``LGE_Convolution``
- ``LGE_Bilinear``
- ``LGE_ReTrAct``
- ``LGE_Exp``
- ``PolyakovLoopGenerator`` and ``PositiveOrientationPlaquetteGenerator``

See [10.1103/PhysRevLett.128.032003].
"""

import torch
import itertools
from typing import List, Tuple, Dict, Any
from .convolution import LGE_Convolution
from .bilinear import LGE_Bilinear
from .loop_generator import PolyakovLoopGenerator, PositiveOrientationPlaquetteGenerator
from .activation import LGE_ReTrAct
from .exponentiation import LGE_Exp

from ...base.paths import PathBuffer

class LGE_CB(torch.nn.Module):
    r"""
    A combined lattice gauge equivariant convolution-bilinear layer.
    Originally described in the supplemental material of https://link.aps.org/doi/10.1103/PhysRevLett.128.032003.

    The input features are mapped as

    .. math::

        W_{jk}'(x) &= (T_{p_k}W_j)(x) \\
        W^a(x) &= (W_j(x), W_j^\dagger(x), \mathbb{1}) \\
        W^b(x) &= (W_{jk}'(x), W_{jk}^{\prime\dagger}(x), \mathbb{1})\\
        W_i^o(x) &= \sum\limits_{ijj'} \alpha_{ijj'} W_j^a(x) W_{j'}^b(x)

    and :math:`W^o` is returned.
    """

    def __init__(
        self, n_features_in: int, n_features_out: int, paths: list, disable_cache: bool = True
    ) -> None:
        super(LGE_CB, self).__init__()
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.paths = paths
        self.cache = {}
        self.disable_cache = disable_cache

        self.weights = torch.nn.Parameter(torch.randn(n_features_in*2 + 1
                                                      , (n_features_in * len(paths))*2 + 1
                                                      , n_features_out, dtype=torch.cdouble))

    def get_path_buffers(self, U: torch.Tensor) -> List[Any]:
        """Get path buffers for the paths used in the model and the given link field U, using cache if available.
        """
        if id(U) not in self.cache:
            path_buffers = [PathBuffer(U, path) for path in self.paths]
        else:
            return self.cache[id(U)]
        if not self.disable_cache:
            self.cache[id(U)] = path_buffers
        return path_buffers

    def forward(self, U: torch.Tensor, features_in: torch.Tensor) -> torch.Tensor:
        path_buffers = self.get_path_buffers(U)
        transported_features = torch.stack([pk.m_transport(fj) for pk, fj in itertools.product(path_buffers, features_in)])
        identity = torch.stack([torch.zeros_like(features_in[0])])
        identity[:,:,:,:,:] = torch.eye(3,3, dtype=torch.cdouble)

        features_a = torch.concatenate((features_in, features_in.adjoint(), identity))
        features_b = torch.concatenate((transported_features, transported_features.adjoint(), identity))

        return torch.einsum("nmo,nabcdij,mabcdjk->oabcdik", self.weights, features_a, features_b)
