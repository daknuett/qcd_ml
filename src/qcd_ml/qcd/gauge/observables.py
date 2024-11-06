"""
QCD observables that are computed on the gauge field.

"""
import torch
import numpy
from ...base.paths import PathBuffer
from ...base.operations import SU3_group_compose
from ...util.tensor import levi_civita_index_and_sign_iterator


def plaquette_field(U, _gpt_compat=False):
    """
    Plaquette field of a gauge field. See [1]_ [2]_.

    If ``_gpt_compat=True``, the field is rescaled to match gpt's conventions.
    
    .. [1]: 10.1103/PhysRevD.10.2445
    .. [2]: 10.1007/978-3-642-01850-3
    """
    Nd = 4
    ndims = 3
    Hp = lambda mu, lst: lst + [(mu, 1)]
    Hm = lambda mu, lst: lst + [(mu, -1)]

    gpt_rescale_factor = 2 / Nd / (Nd - 1) / ndim
    
    plaquette_paths = [[
            list(reversed(Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))))
             for nu in range(4)] for mu in range(4)]
    
    untraced_plaquettes = [[PathBuffer(U, pmunu).accumulated_U  for pmunu in pmu] for pmu in plaquette_paths]
    untraced_plaquette = torch.zeros_like(U[0])
    
    for mu, pmu in enumerate(untraced_plaquettes):
        for pmunu in pmu[:mu]:
            untraced_plaquette += pmunu
    plaquette_field = torch.einsum("abcdii->abcd", untraced_plaquette).real

    if _gpt_compat:
        return plaquette_field * gpt_rescale_factor
    else:
        return plaquette_field


def _mul(iterable):
    res = 1
    for i in iterable:
        res *= i
    return res

def topological_charge_density_clover(U, _gpt_compat=False):
    """
    The topological charge density field :math:`q(n)` [1]_ [2]_ using the clover 
    field strength [3]_.

    .. [1]: 10.1007/BF02029132
    .. [2]: 10.1007/978-3-642-01850-3
    .. [3]: 10.1016/0550-3213(85)90002-1
    """
    Hp = lambda mu, lst: lst + [(mu, 1)]
    Hm = lambda mu, lst: lst + [(mu, -1)]

    clover_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

    clover_terms = [[[PathBuffer(U, pi).accumulated_U for pi in pnu] for pnu in pmu] for pmu in clover_paths]
    Qmunu = [[sum(pnu) / 4 for pnu in pmu] for pmu in clover_terms]
    Fmunu = [[Qmunu[mu][nu] - Qmunu[nu][mu] for nu in range(4)] for mu in range(4)]

    identity = torch.zeros_like(Fmunu[0][0])
    identity[:,:,:,:] = torch.eye(3,3, dtype=torch.cdouble)
    
    q_field = 0
    for (mu,nu,rho,sigma), sgn in levi_civita_index_and_sign_iterator(4):
        q_field += sgn * torch.einsum("abcdii->abcd", identity - SU3_group_compose(Fmunu[mu][nu], Fmunu[rho][sigma]))
        
    if not _gpt_compat:
        rescale = 1 / 32 / numpy.pi**2
    else:
        rescale = 16.0 / (32.0 * numpy.pi**2) * (0.125**2.0) * _mul(U[0].shape[0:4])
    return q_field * rescale
