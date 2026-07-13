"""
qcd_ml.qcd.dirac
================

Dirac operators.
"""

import torch
from typing import List, Union

from ...base.hop import v_hop
from ...base.operations import mspin_const_group_compose, v_spin_const_transform
from ...base.paths import PathBuffer
from ...util.comptime import comptime
from ..static import gamma

"""
qcd_ml.qcd.dirac
================

Dirac operators.
"""


@comptime([(mu, nu) for mu in range(4) for nu in range(4)])
def sigmamunu(mu: int, nu: int) -> torch.Tensor:
    """Compute the sigma_{mu,nu} matrix from gamma matrices.

    The sigma matrices are defined as:
    sigma_{mu,nu} = (gamma_mu * gamma_nu - gamma_nu * gamma_mu) / 2

    This is the generator of the Lorentz group in spinor space.

    Args:
        mu (int): First Lorentz index (0-3).
        nu (int): Second Lorentz index (0-3).

    Returns:
        torch.Tensor: The sigma_{mu,nu} matrix as a spinor space tensor.

    Note:
        This function is decorated with @comptime for compile-time evaluation.
    """
    return (
        mspin_const_group_compose(gamma[mu], gamma[nu])
        - mspin_const_group_compose(gamma[nu], gamma[mu])
    ) / 2

class dirac_wilson:
    """Dirac Wilson operator. See arXiv:2302.05419.

    The Wilson Dirac operator is the discretized version of the Dirac operator
    in lattice QCD, including a Wilson term to remove fermion doubling.

    Attributes:
        U (Union[torch.Tensor, List[torch.Tensor]]): Gauge field configuration.
        mass_parameter (float): Bare mass parameter.
        gamma (torch.Tensor): Stacked gamma matrices on the same device as U.
        dag (bool): Whether to use the dagger (adjoint) operator.
    """

    def __init__(self, U: Union[torch.Tensor, List[torch.Tensor]], mass_parameter: float, dag: bool = False) -> None:
        """Initialize the Wilson Dirac operator.

        Args:
            U (Union[torch.Tensor, List[torch.Tensor]]): Gauge field configuration.
                Can be a single tensor or a list of tensors, one for each direction.
            mass_parameter (float): Bare mass parameter in lattice units.
            dag (bool, optional): If True, use the adjoint operator. Defaults to False.
        """
        self.U = U
        self.mass_parameter = mass_parameter

        # copy gamma to local device.
        self.gamma = torch.stack(gamma).to(U[0].device)

        self.dag = dag

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the Wilson Dirac operator to a spinor field.

        Computes: D_w * v = (4 + m) * v - 1/2 * sum_mu [ (U_mu v_{x+mu} + U_{-mu} v_{x-mu}) 
        - gamma_mu (U_mu v_{x+mu} - U_{-mu} v_{x-mu}) ]

        Args:
            v (torch.Tensor): Input spinor field.

        Returns:
            torch.Tensor: Result of applying the Wilson Dirac operator to v.
        """
        sign = 1 if not self.dag else -1
        result = (4 + self.mass_parameter) * v
        for mu in range(4):
            hopped_pos = v_hop(self.U, mu, 1, v)
            hopped_neg = v_hop(self.U, mu, -1, v)

            result -= (hopped_pos + hopped_neg) / 2
            result += sign * (
                v_spin_const_transform(gamma[mu], hopped_neg - hopped_pos) / 2
            )

        return result

    def apply_diag(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the diagonal part of the Wilson Dirac operator.

        This is the part that acts only on the local site without
        any hopping terms.

        Args:
            v (torch.Tensor): Input spinor field.

        Returns:
            torch.Tensor: Result of applying the diagonal part of the operator to v.
        """
        result = (4 + self.mass_parameter) * v

        return result

    def apply_pos_hop(self, v: torch.Tensor, mu: int) -> torch.Tensor:
        """Apply the positive hopping term in direction mu.

        Computes the contribution from hopping in the positive mu direction.

        Args:
            v (torch.Tensor): Input spinor field.
            mu (int): Direction index (0-3).

        Returns:
            torch.Tensor: Result of the positive hop in direction mu.
        """
        sign = 1 if not self.dag else -1
        hopped = v_hop(self.U, mu, 1, v)
        result = -hopped / 2 - sign * (
            v_spin_const_transform(self.gamma[mu], hopped) / 2
        )

        return result

    def apply_neg_hop(self, v: torch.Tensor, mu: int) -> torch.Tensor:
        """Apply the negative hopping term in direction mu.

        Computes the contribution from hopping in the negative mu direction.

        Args:
            v (torch.Tensor): Input spinor field.
            mu (int): Direction index (0-3).

        Returns:
            torch.Tensor: Result of the negative hop in direction mu.
        """
        sign = 1 if not self.dag else -1
        hopped = v_hop(self.U, mu, -1, v)
        result = -hopped / 2 + sign * (
            v_spin_const_transform(self.gamma[mu], hopped) / 2
        )

        return result

class dirac_wilson_clover:
    """Dirac Wilson operator with clover term improvement.

    The clover-improved Wilson Dirac operator adds a clover term to
    improve the action and reduce O(a) errors. See arXiv:2302.05419.

    Attributes:
        U (Union[torch.Tensor, List[torch.Tensor]]): Gauge field configuration.
        mass_parameter (float): Bare mass parameter.
        csw (float): Clover improvement coefficient.
        gamma (torch.Tensor): Stacked gamma matrices on the same device as U.
        sigmamunu (torch.Tensor): Stacked sigma matrices on the same device as U.
        plaquette_path_buffers (list): Path buffers for plaquette computations.
        dag (bool): Whether to use the dagger (adjoint) operator.
    """

    def __init__(self, U: Union[torch.Tensor, List[torch.Tensor]], mass_parameter: float, csw: float, dag: bool = False) -> None:
        """Initialize the clover-improved Wilson Dirac operator.

        Args:
            U (Union[torch.Tensor, List[torch.Tensor]]): Gauge field configuration.
                Can be a single tensor or a list of tensors, one for each direction.
            mass_parameter (float): Bare mass parameter in lattice units.
            csw (float): Clover improvement coefficient.
            dag (bool, optional): If True, use the adjoint operator. Defaults to False.
        """
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

        # copy both gamma and sigma to local device.
        self.gamma = torch.stack(gamma).to(U[0].device)

        self.sigmamunu = torch.stack(
            [
                torch.stack([sigmamunu(mu, nu) for nu in range(4)])
                for mu in range(4)
            ]
        ).to(U[0].device)

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]

        plaquette_paths = [
            [
                [
                    Hm(mu, Hm(nu, Hp(mu, Hp(nu, [])))),
                    Hm(nu, Hp(mu, Hp(nu, Hm(mu, [])))),
                    Hp(nu, Hm(mu, Hm(nu, Hp(mu, [])))),
                    Hp(mu, Hp(nu, Hm(mu, Hm(nu, [])))),
                ]
                for nu in range(4)
            ]
            for mu in range(4)
        ]

        self.plaquette_path_buffers = [
            [[PathBuffer(U, pi) for pi in pnu] for pnu in pmu]
            for pmu in plaquette_paths
        ]

        self.dag = dag

    def Qmunu(self, mu: int, nu: int, v: torch.Tensor) -> torch.Tensor:
        """Compute the sum of plaquette paths for directions mu and nu.

        Computes the sum of the four plaquette paths that contribute
        to the field strength tensor F_{mu,nu}.

        Args:
            mu (int): First Lorentz index (0-3).
            nu (int): Second Lorentz index (0-3).
            v (torch.Tensor): Input spinor field to transport.

        Returns:
            torch.Tensor: Sum of transported spinor fields along all four plaquette paths.
        """
        paths = self.plaquette_path_buffers[mu][nu]
        return (
            paths[0].v_transport(v)
            + paths[1].v_transport(v)
            + paths[2].v_transport(v)
            + paths[3].v_transport(v)
        )

    def field_strength(self, mu: int, nu: int, v: torch.Tensor) -> torch.Tensor:
        """Compute the field strength tensor F_{mu,nu} acting on a spinor field.

        The field strength is computed as:
        F_{mu,nu} = (Q_{mu,nu} - Q_{nu,mu}) / 8

        where Q_{mu,nu} is the sum of plaquette paths.

        Args:
            mu (int): First Lorentz index (0-3).
            nu (int): Second Lorentz index (0-3).
            v (torch.Tensor): Input spinor field.

        Returns:
            torch.Tensor: Field strength tensor F_{mu,nu} acting on v.
        """
        return (self.Qmunu(mu, nu, v) - self.Qmunu(nu, mu, v)) / 8

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the clover-improved Wilson Dirac operator to a spinor field.

        The operator includes both the Wilson term and the clover improvement term:
        D_clover = D_wilson - (c_sw / 4) * sum_{mu < nu} sigma_{mu,nu} F_{mu,nu}

        Args:
            v (torch.Tensor): Input spinor field.

        Returns:
            torch.Tensor: Result of applying the clover-improved Wilson Dirac operator to v.
        """
        sign = 1 if not self.dag else -1
        result = (4 + self.mass_parameter) * v
        for mu in range(4):
            hopped_pos = v_hop(self.U, mu, 1, v)
            hopped_neg = v_hop(self.U, mu, -1, v)

            result -= (hopped_pos + hopped_neg) / 2
            result += sign * (
                v_spin_const_transform(gamma[mu], hopped_neg - hopped_pos) / 2
            )

        improvement = 0
        for mu in range(4):
            for nu in range(mu):
                # sigma and field_strength are both anti symmetric.
                improvement = improvement + 2 * v_spin_const_transform(
                    self.sigmamunu[mu, nu], self.field_strength(mu, nu, v)
                )

        return result - self.csw / 4 * improvement

    def apply_diag(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the diagonal part of the clover-improved Wilson Dirac operator.

        This includes both the diagonal Wilson term and the diagonal part
        of the clover improvement.

        Args:
            v (torch.Tensor): Input spinor field.

        Returns:
            torch.Tensor: Result of applying the diagonal part to v.
        """
        result = (4 + self.mass_parameter) * v

        improvement = 0
        for mu in range(4):
            for nu in range(mu):
                # sigma and field_strength are both anti-symmetric.
                improvement = improvement + 2 * v_spin_const_transform(
                    self.sigmamunu[mu, nu], self.field_strength(mu, nu, v)
                )

        return result - self.csw / 4 * improvement

    def apply_pos_hop(self, v: torch.Tensor, mu: int) -> torch.Tensor:
        """Apply the positive hopping term in direction mu.

        Computes the contribution from hopping in the positive mu direction.

        Args:
            v (torch.Tensor): Input spinor field.
            mu (int): Direction index (0-3).

        Returns:
            torch.Tensor: Result of the positive hop in direction mu.
        """
        sign = 1 if not self.dag else -1
        hopped = v_hop(self.U, mu, 1, v)
        result = -hopped / 2 - sign * (
            v_spin_const_transform(self.gamma[mu], hopped) / 2
        )

        return result

    def apply_neg_hop(self, v: torch.Tensor, mu: int) -> torch.Tensor:
        """Apply the negative hopping term in direction mu.

        Computes the contribution from hopping in the negative mu direction.

        Args:
            v (torch.Tensor): Input spinor field.
            mu (int): Direction index (0-3).

        Returns:
            torch.Tensor: Result of the negative hop in direction mu.
        """
        sign = 1 if not self.dag else -1
        hopped = v_hop(self.U, mu, -1, v)
        result = -hopped / 2 + sign * (
            v_spin_const_transform(self.gamma[mu], hopped) / 2
        )

        return result
