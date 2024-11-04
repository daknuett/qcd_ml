import torch

from ..static import gamma
from ...base.operations import v_spin_const_transform, mspin_const_group_compose
from ...base.hop import v_hop
from ...base.paths import PathBuffer

from ...util.comptime import comptime

"""
qcd_ml.qcd.dirac
================

Dirac operators.
"""


@comptime([(mu, nu) for mu in range(4) for nu in range(4)])
def sigmamunu(mu, nu):
    return (mspin_const_group_compose(gamma[mu], gamma[nu]) 
            - mspin_const_group_compose(gamma[nu], gamma[mu])) / 2

class dirac_wilson:
    """
    Dirac Wilson operator. See arXiv:2302.05419.
    """
    def __init__(self, U, mass_parameter):
        self.U = U
        self.mass_parameter = mass_parameter


    def __call__(self, v):
        result = (4 + self.mass_parameter) * v 
        for mu in range(4):
            result -= v_hop(self.U, mu, 1, v) / 2
            result -= v_hop(self.U, mu, -1, v) / 2

            result += v_spin_const_transform(gamma[mu], v_hop(self.U, mu, -1, v)) / 2
            result -= v_spin_const_transform(gamma[mu], v_hop(self.U, mu, 1, v)) / 2

        return result


class dirac_wilson_clover:
    """
    Dirac Wilson operator with clover term improvement.

    See arXiv:2302.05419.
    """
    def __init__(self, U, mass_parameter, csw):
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        self.plaquette_path_buffers = [[[PathBuffer(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

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
        result = (4 + self.mass_parameter) * v 
        for mu in range(4):
            result -= v_hop(self.U, mu, 1, v) / 2
            result -= v_hop(self.U, mu, -1, v) / 2

            result += v_spin_const_transform(gamma[mu], v_hop(self.U, mu, -1, v)) / 2
            result -= v_spin_const_transform(gamma[mu], v_hop(self.U, mu, 1, v)) / 2

        improvement = 0
        for mu in range(4):
            for nu in range(mu):
                # sigma and field_strength are both anti symmetric.
                improvement = (improvement 
                               + 2*v_spin_const_transform(sigmamunu(mu, nu), self.field_strength(mu, nu, v))
                               )

        return result - self.csw / 4 * improvement

