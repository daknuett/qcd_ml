#!/usr/bin/env python3

import torch

from .operations import v_gauge_transform

def v_hop(U, mu, direction, v):
    if direction == -1:
        result = torch.roll(v, -1, mu)
        return v_gauge_transform(U[mu], result)
    else:
        Umudg = U[mu].adjoint()
        result = v_gauge_transform(Umudg, v)
        return torch.roll(result, 1, mu)
