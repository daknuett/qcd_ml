#!/usr/bin/env python3
"""
qcd_ml.base.hop
===============

Gauge-equivariant hops.

"""

import torch
from typing import List

from .operations import v_gauge_transform, m_gauge_transform

def v_hop(U: list[torch.Tensor], mu: int, direction: int, v: torch.Tensor) -> torch.Tensor:
    """
    Gauge-equivariant hop for a vector-like field.
    
    Args:
        U: List of gauge field tensors, one for each direction.
        mu: The direction index (0-3 for 4D spacetime).
        direction: The direction of the hop (1 for forward, -1 for backward).
        v: The vector-like field tensor.
        
    Returns:
        The hopped vector field tensor.
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
    
    Args:
        mu: The direction index (0-3 for 4D spacetime).
        direction: The direction of the hop (1 for forward, -1 for backward).
        v: The vector-like field tensor.
        
    Returns:
        The hopped vector field tensor.
    """
    return torch.roll(v, direction,  mu)


def m_hop(U: list[torch.Tensor], mu: int, direction: int, m: torch.Tensor) -> torch.Tensor:
    """
    Gauge-equivariant hop for a matrix-like field.
    
    Args:
        U: List of gauge field tensors, one for each direction.
        mu: The direction index (0-3 for 4D spacetime).
        direction: The direction of the hop (1 for forward, -1 for backward).
        m: The matrix-like field tensor.
        
    Returns:
        The hopped matrix field tensor.
    """
    if direction == -1:
        result = torch.roll(m, -1, mu)
        return m_gauge_transform(U[mu], result)
    else:
        Umudg = U[mu].adjoint()
        result = m_gauge_transform(Umudg, m)
        return torch.roll(result, 1, mu)
