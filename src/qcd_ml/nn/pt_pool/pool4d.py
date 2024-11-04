"""
This file provides v_pool4d and v_unpool4d functions.

They are slow pure-python implementations of the pooling and unpooling 
also provided in qcd_ml_accel.pool4d.
"""
import torch
from itertools import product

def v_pool4d(fine_v, block_size):
    L_coarse = [li // bi for li, bi in zip(fine_v.shape[:-2], block_size)]
    res = torch.zeros(*L_coarse, *fine_v.shape[-2:], dtype=torch.cdouble)
    for x,y,z,t in product(*tuple([range(block_size[i]) for i in range(4)])):
        res += fine_v[x::block_size[0], y::block_size[1], z::block_size[2], t::block_size[3]]
    return res


def v_unpool4d(coarse_v, block_size):
    L_coarse = coarse_v.shape[:-2]
    res = torch.zeros(*[li*bi for li,bi in zip(L_coarse, block_size)], *coarse_v.shape[-2:], dtype=torch.cdouble)
    for x,y,z,t in product(*tuple([range(L_coarse[i]) for i in range(4)])):
        res[x*block_size[0]: (x + 1) * block_size[0]
            , y*block_size[1]: (y + 1) * block_size[1]
            , z*block_size[2]: (z + 1) * block_size[2]
            , t*block_size[3]: (t + 1) * block_size[3]] = coarse_v[x,y,z,t]
    return res
