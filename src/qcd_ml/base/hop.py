#!/usr/bin/env python3
"""
qcd_ml.base.hop
===============

Gauge-equivariant hops. Mostly for internal use.

Generally the hop functions take the following arguments:

U: Gauge field tensor of shape (4, Lx, Ly, Lz, Lt, 3, 3) where 4 is the number
    of spacetime directions.
mu: The direction index (0-3 for 4D spacetime).
direction: The direction of the hop (1 for forward, -1 for backward).
v or m: The object to be gauge-transported by a single hop
"""

import torch
from typing import List

from .operations import v_gauge_transform, m_gauge_transform

def v_hop(U: torch.Tensor, mu: int, direction: int, v: torch.Tensor) -> torch.Tensor:
    """
    Gauge-equivariant hop for a vector-like field, i.e., fields that transform as 

    .. math::
        v(x) \rightarrow \Omega(x) v(x).
    
    """
    if direction == -1:
        result = torch.roll(v, -1, mu)
        return v_gauge_transform(U[mu], result)
    else:
        Umudg = U[mu].adjoint()
        result = v_gauge_transform(Umudg, v)
        return torch.roll(result, 1, mu)


def v_ng_hop(mu: int, direction: int, v: torch.Tensor) -> torch.Tensor:
    """
    Hop for a vector-like field without gauge degrees of freedom.
    """
    return torch.roll(v, direction,  mu)


def m_hop(U: torch.Tensor, mu: int, direction: int, m: torch.Tensor) -> torch.Tensor:
    r"""
    Gauge-equivariant hop for a matrix-like field, i.e., fields that transform as 

    .. math::
        M(x) \rightarrow \Omega(x) M(x) \Omega^\dagger(x).
    """
    if direction == -1:
        result = torch.roll(m, -1, mu)
        return m_gauge_transform(U[mu], result)
    else:
        Umudg = U[mu].adjoint()
        result = m_gauge_transform(Umudg, m)
        return torch.roll(result, 1, mu)
