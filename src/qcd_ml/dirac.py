from .static import gamma
from .operations import v_spin_const_transform, mspin_const_group_compose

from .hop import v_hop

class dirac_wilson:
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
    def __init__(self, U, mass_parameter, csw):
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

    def Qmunu(self, mu, nu, v):
        Hp = lambda mu, vec: v_hop(self.U, mu, 1, vec)
        Hm = lambda mu, vec: v_hop(self.U, mu, -1, vec)

        return (
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, v))))
                + Hm(nu, Hp(mu, Hp(nu, Hm(mu, v))))
                + Hp(nu, Hm(mu, Hm(nu, Hp(mu, v))))
                + Hp(mu, Hp(nu, Hm(mu, Hm(nu, v))))
                )

    def field_strength(self, mu, nu, v):
        return (self.Qmunu(mu, nu, v) - self.Qmunu(nu, mu, v)) / 8

    def sigmamunu(self, mu, nu):
        return (mspin_const_group_compose(gamma[mu], gamma[nu]) 
                - mspin_const_group_compose(gamma[nu], gamma[mu])) / 2

    def __call__(self, v):
        result = (4 + self.mass_parameter) * v 
        for mu in range(4):
            result -= v_hop(self.U, mu, 1, v) / 2
            result -= v_hop(self.U, mu, -1, v) / 2

            result += v_spin_const_transform(gamma[mu], v_hop(self.U, mu, -1, v)) / 2
            result -= v_spin_const_transform(gamma[mu], v_hop(self.U, mu, 1, v)) / 2

        improvement = 0
        for mu in range(4):
            for nu in range(4):
                improvement = (improvement 
                               + v_spin_const_transform(self.sigmamunu(mu, nu), self.field_strength(mu, nu, v))
                               )

        return result - self.csw / 4 * improvement
