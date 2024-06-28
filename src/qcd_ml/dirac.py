from .static import gamma
from .operations import v_spin_const_transform

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
