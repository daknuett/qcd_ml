#!/usr/bin/env python3
"""
qcd_ml.base.hop
===============

Gauge-equivariant hops.

"""

import torch

from .operations import v_gauge_transform

def v_hop(U, mu, direction, v):
    """
    Gauge-equivariant hop for a vector-like field.
    """
    if direction == -1:
        result = torch.roll(v, -1, mu)
        return v_gauge_transform(U[mu], result)
    else:
        Umudg = U[mu].adjoint()
        result = v_gauge_transform(Umudg, v)
        return torch.roll(result, 1, mu)


def v_ng_hop(mu, direction, v):
    """
    Hop for a vector-like field without gauge degrees of freedom.
    """
    return torch.roll(v, direction,  mu)


def m_hop(U, mu, direction, m):
    """
    Gauge-equivariant hop for a matrix-like field.
    """
    if direction == -1:
        result = torch.roll(m, -1, mu)
        return m_gauge_transform(U[mu], result)
    else:
        Umudg = U[mu].adjoint()
        result = m_gauge_transform(Umudg, m)
        return torch.roll(result, 1, mu)
