"""This module provides smearing of gauge links. Currenltly the following smearing algorithms are implemented:

    - Stout smearing (``stout``) [1]_ [2]_
    - Wilson gradient flow (``wilson_flow``) [3]_
    - Plaquette-plus-rectangle gradient flow (``improved_flow``) [3]_ [4]_ [5]_ [6]_


.. [1] 10.1103/PhysRevD.69.054501
.. [2] 10.1007/978-3-642-01850-3
.. [3] 10.1007/JHEP08(2010)071
.. [4] 10.1007/BF01206178
.. [5] arXiv:1111.7054 
.. [6] 10.1016/S0550-3213(00)00145-0


"""

import torch

from ...base.paths  import PathBuffer
from ...base.operations import SU3_group_compose


# Shared helper functions


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

    antihermitian = Omega_mu.adjoint() - Omega_mu
    trace = torch.einsum("mabcdii->mabcd", antihermitian)

    identity = torch.eye(3, dtype=U.dtype, device=U.device)
    trace_removal = trace[..., None, None] * identity

    return 1j/2 * (antihermitian - trace_removal / 3)


def _rectangle_staple_generator(U: torch.Tensor) -> torch.Tensor:
    r"""Compute the traceless Hermitian rectangle generator :math:`Q_\mu(x)`.

    The generator is constructed from the sum of rectangular staples as

    .. math::

        C_\mu(x) &= \sum_{\nu \neq \mu} C_{\mu\nu}^{\mathrm{rect}}(x), \\
        \Omega_\mu(x) &= C_\mu(x) U_\mu^\dagger(x), \\
        Q_\mu(x) &= \frac{i}{2} \left[
            \Omega_\mu^\dagger(x) - \Omega_\mu(x)
            - \frac{1}{N_c} \mathrm{Tr}\left(
                \Omega_\mu^\dagger(x) - \Omega_\mu(x)
            \right) \mathbb{1}
        \right].

    Here, :math:`C_{\mu\nu}^{\mathrm{rect}}(x)` is the sum of the six
    five-link staples associated with the :math:`1\times2` and
    :math:`2\times1` rectangles in the :math:`\mu\nu`-plane.
    """

    Hp = lambda mu, lst: lst + [(mu, 1)]
    Hm = lambda mu, lst: lst + [(mu, -1)]

    rectangle_paths = [[[
        Hm(mu, Hm(nu, Hp(mu, Hp(mu, Hp(nu, []))))),
        Hm(nu, Hp(mu, Hp(mu, Hp(nu, Hm(mu, []))))),
        Hm(mu, Hp(nu, Hp(mu, Hp(mu, Hm(nu, []))))),
        Hp(nu, Hp(mu, Hp(mu, Hm(nu, Hm(mu, []))))),
        Hm(nu, Hm(nu, Hp(mu, Hp(nu, Hp(nu, []))))),
        Hp(nu, Hp(nu, Hp(mu, Hm(nu, Hm(nu, [])))))
    ] for nu in range(4) if nu != mu] for mu in range(4)]

    rectangle_staples = [[
        sum(PathBuffer(U, p).accumulated_U.adjoint() for p in pnu)
        for pnu in pmu
    ] for pmu in rectangle_paths]

    C_mu = torch.stack([
        torch.stack(rectmu).sum(dim=0)
        for rectmu in rectangle_staples
    ])

    Omega_mu = torch.einsum(
        "mabcdij,mabcdjk->mabcdik", C_mu, U.adjoint()
    )

    antihermitian = Omega_mu.adjoint() - Omega_mu
    trace = torch.einsum("mabcdii->mabcd", antihermitian)

    identity = torch.eye(3, dtype=U.dtype, device=U.device)
    trace_removal = trace[..., None, None] * identity

    return 1j/2 * (antihermitian - trace_removal / 3)


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


# Stout smearing


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


# Gradient flows


class improved_flow:
    r"""Gradient flow using a plaquette-plus-rectangle gauge action.

    The flow action is defined by

    .. math::

        S = c_0 S_{\mathrm{plaq}} + c_1 S_{\mathrm{rect}},
        \qquad c_0 = 1 - 8c_1.

    Common choices for :math:`c_1` are:

    - Wilson: :math:`c_1 = 0` [3]_
    - Tree-level Symanzik: :math:`c_1 = -1/12` [4]_
    - Iwasaki: :math:`c_1 = -0.331` [5]_
    - DBW2: :math:`c_1 = -1.4088` [6]_

    The flow is integrated using a third-order Runge-Kutta integrator.
    """

    def __init__(self, epsilon: float, c1: float) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"epsilon > 0 required, got {epsilon}")

        self.epsilon = epsilon
        self.c1 = c1
        self.c0 = 1.0 - 8.0 * c1

    def _generator(self, U: torch.Tensor) -> torch.Tensor:
        """Compute the generator of the flow action.
        """

        Q = self.c0 * _staple_generator(U)

        if self.c1 != 0.0:
            Q += self.c1 * _rectangle_staple_generator(U)

        return Q

    def __call__(self, U: torch.Tensor) -> torch.Tensor:
        """Apply one gradient-flow step to the gauge field.
        """

        eps = self.epsilon

        Q0 = self._generator(U)
        V1 = _exp_update(U, Q0, eps / 4.0)

        Q1 = self._generator(V1)
        V2 = _exp_update(
            V1, (8.0 / 9.0) * Q1 - (17.0 / 36.0) * Q0, eps
        )

        Q2 = self._generator(V2)
        V = _exp_update(
            V2,
            (3.0 / 4.0) * Q2
            - (8.0 / 9.0) * Q1
            + (17.0 / 36.0) * Q0,
            eps,
        )

        return V


class wilson_flow(improved_flow):
    r"""Wilson gradient flow.

    See [3]_ for details of the flow equation and integration algorithm.
    """

    def __init__(self, epsilon: float) -> None:
        super().__init__(epsilon, c1=0.0)