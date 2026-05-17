import torch

from ..static import gamma
from ...base.operations import v_spin_const_transform, mspin_const_group_compose
from ...base.hop import v_hop
from ...base.paths import PathBuffer

from ...util.comptime import comptime
from ...util import get_device_by_reference

from .dwf import dirac_dwf5_None

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

    ``boundary_phases`` is a list of 4 elements, each being either 1 or -1.
    It is used to set the boundary phases of the fermion fields. Usually,
    fermions are anti-periodic in the time direction, and periodic in the spatial
    directions, which gives [1, 1, 1, -1] for (x, y, z, t). The default is
    [1, 1, 1, 1], which corresponds to periodic boundary conditions in all directions.
    """
    def __init__(self, U, mass_parameter, boundary_phases=[1,1,1,1]):
        self.U = U
        self.mass_parameter = mass_parameter

        self.boundary_phases = boundary_phases

        # copy gamma to local device.
        self.gamma = torch.stack(gamma).to(get_device_by_reference(U[0]))

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]

        self.hop_buffers_forward = [PathBuffer(U, Hp(mu, [])) for mu in range(4)]
        self.hop_buffers_backward = [PathBuffer(U, Hm(mu, [])) for mu in range(4)]

        for mu, bf in enumerate(self.boundary_phases):
            def get_slice_forward(mu):
                return [slice(None,None,None)]*mu + [-1] + [slice(None,None,None)]*(3-mu)
            def get_slice_backward(mu):
                return [slice(None,None,None)]*mu + [0] + [slice(None,None,None)]*(3-mu)

            self.hop_buffers_forward[mu].accumulated_U[ get_slice_forward(mu) ] *= bf
            self.hop_buffers_backward[mu].accumulated_U[ get_slice_backward(mu) ] *= bf


    def __call__(self, v):
        result = (4 + self.mass_parameter) * v 
        for mu in range(4):
            forward = self.hop_buffers_forward[mu].v_transport(v)
            backward = self.hop_buffers_backward[mu].v_transport(v)
            result -= forward / 2
            result -= backward / 2

            result += v_spin_const_transform(self.gamma[mu], backward) / 2
            result -= v_spin_const_transform(self.gamma[mu], forward) / 2

        return result


class dirac_wilson_clover:
    """
    Dirac Wilson operator with clover term improvement.

    See arXiv:2302.05419.

    ``boundary_phases`` is a list of 4 elements, each being either 1 or -1.
    It is used to set the boundary phases of the fermion fields. Usually,
    fermions are anti-periodic in the time direction, and periodic in the spatial
    directions, which gives [1, 1, 1, -1] for (x, y, z, t). The default is
    [1, 1, 1, 1], which corresponds to periodic boundary conditions in all directions.
    """
    def __init__(self, U, mass_parameter, csw, boundary_phases=[1,1,1,1]):
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

        self.boundary_phases = boundary_phases

        # copy both gamma and sigma to local device.
        self.gamma = torch.stack(gamma).to(get_device_by_reference(U[0]))

        self.sigmamunu = torch.stack([
                                torch.stack([sigmamunu(mu, nu) for nu in range(4)])
                            for mu in range(4)]).to(get_device_by_reference(U[0]))

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        self.plaquette_path_buffers = [[[PathBuffer(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

        self.hop_buffers_forward = [PathBuffer(U, Hp(mu, [])) for mu in range(4)]
        self.hop_buffers_backward = [PathBuffer(U, Hm(mu, [])) for mu in range(4)]

        for mu, bf in enumerate(self.boundary_phases):
            def get_slice_forward(mu):
                return [slice(None,None,None)]*mu + [-1] + [slice(None,None,None)]*(3-mu)
            def get_slice_backward(mu):
                return [slice(None,None,None)]*mu + [0] + [slice(None,None,None)]*(3-mu)

            self.hop_buffers_forward[mu].accumulated_U[ get_slice_forward(mu) ] *= bf
            self.hop_buffers_backward[mu].accumulated_U[ get_slice_backward(mu) ] *= bf

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
            forward = self.hop_buffers_forward[mu].v_transport(v)
            backward = self.hop_buffers_backward[mu].v_transport(v)
            result -= forward / 2
            result -= backward / 2

            result += v_spin_const_transform(self.gamma[mu], backward) / 2
            result -= v_spin_const_transform(self.gamma[mu], forward) / 2

        improvement = 0
        for mu in range(4):
            for nu in range(mu):
                # sigma and field_strength are both anti symmetric.
                improvement = (improvement
                               + 2*v_spin_const_transform(self.sigmamunu[mu, nu], self.field_strength(mu, nu, v))
                               )

        return result - self.csw / 4 * improvement


class dirac_dwf5_kwc(dirac_dwf5_None):
    r"""
    5D domain wall fermion dirac operator using Wilson-clover kernel.

    M5 is the mass parameter of the 4D kernel operator.
    m is the mass parameter of the 5D operator.

    See
    - http://arxiv.org/abs/1206.5214
    - 10.1016/j.nuclphysbps.2004.11.180
    """
    def __init__(self, U, m, M5, b, c, Ls, csw, boundary_phases=[1,1,1,1]):
        super().__init__(U, m, M5, b, c, Ls)
        self.csw = csw
        self.boundary_phases = boundary_phases
        self.kernel = dirac_wilson_clover(U, self.M5, csw, boundary_phases)


class dirac_dwf5_kw(dirac_dwf5_None):
    r"""
    5D domain wall fermion dirac operator using Wilson kernel.

    M5 is the mass parameter of the 4D kernel operator.
    m is the mass parameter of the 5D operator.

    See
    - http://arxiv.org/abs/1206.5214
    - 10.1016/j.nuclphysbps.2004.11.180
    """
    def __init__(self, U, m, M5, b, c, Ls, boundary_phases=[1,1,1,-1]):
        super().__init__(U, m, M5, b, c, Ls)

        self.boundary_phases = boundary_phases
        self.kernel = dirac_wilson(U, self.M5, boundary_phases)
