"""
------------

This module provides loop generator layers, i.e., layers that take a
link field :math:`U` and compute a set of gauge equivariant loops.

All layers inherit from ``AbstractLoopGenerator`` and provide the class property
``nfeatures_out``. This allows to access the number of generated loops programatically::

    features_in = PositiveOrientationPlaquetteGenerator.nfeatures_out
    generator = PositiveOrientationPlaquetteGenerator()
    ...
"""

from abc import ABCMeta, abstractmethod
import torch
from ...base.paths import PathBuffer

class AbstractLoopGenerator(metaclass=ABCMeta):
    @classmethod
    @property
    def nfeatures_out(cls):
        return cls.property_nfeatures_out

    @property
    @abstractmethod
    def property_nfeatures_out(self):
        pass


class PolyakovLoopGenerator(torch.nn.Module, AbstractLoopGenerator):
    r"""
    Generates the Polyakov loops

    .. math::
        P_\mu(x) = \prod\limits_{k=0}^{L_\mu} U_\mu(x + k\mu)

    for :math:`\mu = 0,1,2,3`.
    """

    property_nfeatures_out = 4  # XXX This expects 4D fields.

    def __init__(self, disable_cache=False):
        super(PolyakovLoopGenerator, self).__init__()
        self.cache = {}
        self.disable_cache = disable_cache

    def clear_cache(self):
        self.cache = {}

    def forward(self, U):
        """
        Compite the Polyakov loops.
        """
        if id(U) in self.cache:
            loops = self.cache[id(U)]
        else:
            paths = [[(mu, 1) for _ in L_mu] for mu, L_mu in U[0].shape]
            loops = torch.stack([PathBuffer(U, path).gauge_transport_matrix for path in paths])
            if not self.disable_cache:
                self.cache[id(U)] = loops
        return loops


class PositiveOrientationPlaquetteGenerator(torch.nn.Module, AbstractLoopGenerator):
    r"""
    Generates the positivly oriented plaquettes, i.e.,

    .. math::
        P_{\mu\nu}(x) = U_\mu(x) U_\nu(x+\mu) U_\mu^\dagger(x+\nu) U_\nu^\dagger(x)

    """

    property_nfeatures_out = 6

    def __init__(self, disable_cache=False):
        super(PositiveOrientationPlaquetteGenerator, self).__init__()

        self.cache = {}
        self.disable_cache = disable_cache

    def clear_cache(self):
        self.cache = {}

    def forward(self, U):
        """
        Compute the positivly oriented plaquettes.
        """
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
