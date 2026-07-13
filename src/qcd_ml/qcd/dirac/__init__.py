"""
qcd_ml.qcd.dirac
================

Dirac operators.
"""

import torch

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
def sigmamunu(mu, nu):
    return (
        mspin_const_group_compose(gamma[mu], gamma[nu])
        - mspin_const_group_compose(gamma[nu], gamma[mu])
    ) / 2


class dirac_wilson:
    """
    Dirac Wilson operator. See arXiv:2302.05419.
    """

    def __init__(self, U, mass_parameter, dag=False):
        self.U = U
        self.mass_parameter = mass_parameter

        # copy gamma to local device.
        self.gamma = torch.stack(gamma).to(U[0].device)

        self.dag = dag

    def __call__(self, v):
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

    def apply_diag(self, v):
        result = (4 + self.mass_parameter) * v

        return result

    def apply_pos_hop(self, v, mu):
        sign = 1 if not self.dag else -1
        hopped = v_hop(self.U, mu, 1, v)
        result = -hopped / 2 - sign * (
            v_spin_const_transform(self.gamma[mu], hopped) / 2
        )

        return result

    def apply_neg_hop(self, v, mu):
        sign = 1 if not self.dag else -1
        hopped = v_hop(self.U, mu, -1, v)
        result = -hopped / 2 + sign * (
            v_spin_const_transform(self.gamma[mu], hopped) / 2
        )

        return result


class dirac_wilson_clover:
    """
    Dirac Wilson operator with clover term improvement.

    See arXiv:2302.05419.
    """

    def __init__(self, U, mass_parameter, csw, dag=False):
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

    def Qmunu(self, mu, nu, v):
        paths = self.plaquette_path_buffers[mu][nu]
        return (
            paths[0].v_transport(v)
            + paths[1].v_transport(v)
            + paths[2].v_transport(v)
            + paths[3].v_transport(v)
        )

    def field_strength(self, mu, nu, v):
        return (self.Qmunu(mu, nu, v) - self.Qmunu(nu, mu, v)) / 8

    def __call__(self, v):
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

    def apply_diag(self, v):
        result = (4 + self.mass_parameter) * v

        improvement = 0
        for mu in range(4):
            for nu in range(mu):
                # sigma and field_strength are both anti-symmetric.
                improvement = improvement + 2 * v_spin_const_transform(
                    self.sigmamunu[mu, nu], self.field_strength(mu, nu, v)
                )

        return result - self.csw / 4 * improvement

    def apply_pos_hop(self, v, mu):
        sign = 1 if not self.dag else -1
        hopped = v_hop(self.U, mu, 1, v)
        result = -hopped / 2 - sign * (
            v_spin_const_transform(self.gamma[mu], hopped) / 2
        )

        return result

    def apply_neg_hop(self, v, mu):
        sign = 1 if not self.dag else -1
        hopped = v_hop(self.U, mu, -1, v)
        result = -hopped / 2 + sign * (
            v_spin_const_transform(self.gamma[mu], hopped) / 2
        )

        return result
