"""
This module provides loop generator layers, i.e., layers that take a
link field :math:`U` and compute a set of gauge equivariant loops.
"""

import torch
from ...base.paths import PathBuffer


class PolyakovLoopGenerator(torch.nn.Module):
    r"""
    Generates the Polyakov loops
    .. math::
        P_\mu(x) = \prod\limits_{k=0}^{L_\mu} U_\mu(x + k\mu)
    for :math:`\mu = 0,1,2,3`.
    """

    def __init__(self, disable_cache=False):
        super(PolyakovLoopGenerator, self).__init__()
        self.cache = {}
        self.disable_cache = disable_cache

    def clear_cache(self):
        self.cache = {}

    def forward(self, U):
        if id(U) in self.cache:
            loops = self.cache[id(U)]
        else:
            paths = [[(mu, 1) for _ in L_mu] for mu, L_mu in U[0].shape]
            loops = torch.stack([PathBuffer(U, path).gauge_transport_matrix for path in paths])
            if not self.disable_cache:
                self.cache[id(U)] = loops
        return loops


class PositiveOrientationPlaquetteGenerator(torch.nn.Module):
    r"""
    Generates the positivly oriented Plaquettes, i.e.,
    .. math::
        P_{\mu\nu}(x) = U_\mu(x) U_\nu(x+\mu) U_\mu^\dagger(x+\nu) U_\nu^\dagger(x)
    """

    def __init__(self, disable_cache=False):
        super(PositiveOrientationPlaquetteGenerator, self).__init__()

        self.cache = {}
        self.disable_cache = disable_cache

    def clear_cache(self):
        self.cache = {}

    def forward(self, U):
        if id(U) in self.cache:
            loops = self.cache[id(U)]
        else:

            paths = [[(mu, 1), (nu, 1), (mu, -1), (nu, -1)]
                     for mu in range(4) for nu in range(mu)]
            # Remove the empty path generated above.
            paths = [pi for pi in paths if len(pi)]

            loops = torch.stack([PathBuffer(U, path).gauge_transport_matrix for path in paths])
            if not self.disable_cache:
                self.cache[id(U)] = loops
        return loops
