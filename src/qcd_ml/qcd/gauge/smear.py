"""This module provides smearing of gauge links. Currenltly the following smearing algorithms are implemented:

    - Stout smearing (``stout``) [1]_ [2]_
    - Wilson gradient flow (``wilson_flow``) [3]_


.. [1]: 10.1103/PhysRevD.69.054501
.. [2]: 10.1007/978-3-642-01850-3
.. [3]: 10.1007/JHEP08(2010)071


"""

import torch

from ...base.paths  import PathBuffer
from ...base.operations import SU3_group_compose


def _staple_generator(U: torch.Tensor, rho: torch.Tensor = None) -> torch.Tensor:
    r"""Compute the traceless Hermitian staple generator :math:`Q_\mu(x)`.

    The generator is constructed from the weighted sum of staples as

    .. math::

        C_{\mu\nu}(x) &=
            U_\nu(x) U_\mu(x+\hat\nu) U_\nu^\dagger(x+\hat\mu) \\
            &\quad + U_\nu^\dagger(x-\hat\nu) U_\mu(x-\hat\nu)
            U_\nu(x-\hat\nu+\hat\mu), \\
        C_\mu(x) &= \sum_{\nu \neq \mu} \rho_{\mu\nu} C_{\mu\nu}(x), \\
        \Omega_\mu(x) &= C_\mu(x) U_\mu^\dagger(x), \\
        Q_\mu(x) &= \frac{i}{2} \left[
            \Omega_\mu^\dagger(x) - \Omega_\mu(x)
            - \frac{1}{N_c} \mathrm{Tr}\left(
                \Omega_\mu^\dagger(x) - \Omega_\mu(x)
            \right) \mathbb{1}
        \right].

    If ``rho`` is ``None``, unit weights are used for all
    :math:`\mu \neq \nu`.
    """
    Hp = lambda mu, lst: lst + [(mu, 1)]
    Hm = lambda mu, lst: lst + [(mu, -1)]

    staple_paths1 = [[Hm(nu, Hp(mu, Hp(nu, [])))
                     for nu in range(4)] for mu in range(4)]
    staple_paths2 = [[Hp(nu, Hp(mu, Hm(nu, [])))
                     for nu in range(4)] for mu in range(4)]
    staple1 = [[PathBuffer(U, pnu).accumulated_U.adjoint() for pnu in pmu]
               for pmu in staple_paths1]
    staple2 = [[PathBuffer(U, pnu).accumulated_U.adjoint() for pnu in pmu]
               for pmu in staple_paths2]

    staples = (torch.stack([torch.stack(stapmu) for stapmu in staple1]) 
               + torch.stack([torch.stack(stapmu) for stapmu in staple2]))

    weights = None
    if rho is None:
        weights = torch.ones(4, 4, dtype=torch.cdouble, device=U.device)
        weights.fill_diagonal_(0)
    else:
        weights = rho

    Omega_mu = torch.einsum("mn,mnabcdij,mabcdjk->mabcdik"
                                , weights, staples, U.adjoint())

    hermitian = Omega_mu.adjoint() - Omega_mu
    trace = torch.einsum("mabcdii->mabcd", hermitian)

    identity = torch.clone(hermitian)
    identity[:, :, :, :, :] = torch.eye(3, dtype=torch.cdouble)
    trace_removal = torch.einsum("mabcdij,mabcd->mabcdij"
                                 , identity, trace)
    return 1j/2 * (hermitian - trace_removal / 3)


def _exp_update(
        U: torch.Tensor, Q: torch.Tensor, epsilon: float = 1.0
    ) -> torch.Tensor:
        r"""Update the gauge links as

        .. math::

        U_\mu'(x) = \exp\left(i\epsilon Q_\mu(x)\right) U_\mu(x).

        """

        exp_iQ = torch.matrix_exp(1j * epsilon * Q)

        return torch.stack([
            SU3_group_compose(exp_iQ[mu], U[mu])
            for mu in range(4)
        ])


class stout:
    r"""This class is used to construct the stout smearing operation.
    See [1]_ and [2]_ for details of the smearing algorithm.

    This class is an abstract representation of the smearing operation with only
    the parameter ..math:`\rho` being specified. The actual smearing operation is
    performed as such::

        smearer = stout(rho)

        for i in range(n_smearing_steps):
            U = smearer(U)

    Classmethods are provided for common ..math:`\rho` matrices:

    - ``constant_rho``: ..math:`\rho` is a constant matrix.
    - ``spatial_only``: ..math:`\rho` is a constant matrix 
                        with the temporal components set to zero.
    """

    def __init__(self, rho: torch.Tensor) -> None:
        if rho.shape != (4, 4):
            raise ValueError(f"expected 4x4 tensor, got {rho.shape}")
        self.rho = rho

        # rho_mumu != 0 is technically forbidden. See
        # Gattringer Lang 2012 eq. (6.52).
        for mu in range(4):
            self.rho[mu, mu] = 0

    def __call__(self, U: torch.Tensor) -> torch.Tensor:
        """Apply stout smearing to the gauge field.
        """

        Q = _staple_generator(U, self.rho)
        return _exp_update(U, Q)

    @classmethod
    def constant_rho(cls, rho: float) -> 'stout':
        r"""
        Create a stout smearer with :math:`\rho_{ij} = \rho`.
        """
        rho_tensor = rho * torch.ones(4, 4, dtype=torch.cdouble)
        return cls(rho_tensor)

    @classmethod
    def spatial_only(cls, rho: float) -> 'stout':
        """
        The same as ``constant_rho`` but temporal direction is zero.
        """
        rho_tensor = rho * torch.ones(4, 4, dtype=torch.cdouble)
        rho_tensor[:, -1] *= 0
        rho_tensor[-1, :] *= 0
        return cls(rho_tensor)


class wilson_flow:
    r"""This class is used to construct the Wilson gradient flow.
    See [3]_ for details of the flow equation and integration algorithm.

    performed as such::

        flow = wilson_flow(epsilon)

        for i in range(n_flow_steps):
            U = flow(U)
    """

    def __init__(self, epsilon: float) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"epsilon > 0 required, got {epsilon}")
        self.epsilon = epsilon

    def __call__(self, U: torch.Tensor) -> torch.Tensor:
        """Apply Wilson flow to the gauge field.
        """
        V = U.clone()
        eps = self.epsilon

        Q0 = _staple_generator(V)
        V1 = _exp_update(V, Q0, eps / 4.0)

        Q1 = _staple_generator(V1)
        V2 = _exp_update(
            V1, (8.0 / 9.0) * Q1 - (17.0 / 36.0) * Q0, eps
        )

        Q2 = _staple_generator(V2)
        V = _exp_update(
            V2,
            (3.0 / 4.0) * Q2
            - (8.0 / 9.0) * Q1
            + (17.0 / 36.0) * Q0,
            eps,
        )

        return V