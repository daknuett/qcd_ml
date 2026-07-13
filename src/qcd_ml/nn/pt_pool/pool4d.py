"""
This file provides v_pool4d and v_unpool4d functions.

They are slow pure-python implementations of the pooling and unpooling
also provided in qcd_ml_accel.pool4d.
"""
import torch
from itertools import product
from typing import Tuple


def v_pool4d(fine_v: torch.Tensor, block_size: Tuple[int, ...]) -> torch.Tensor:
    """Performs 4D volume pooling by summing over blocks.

    Reduces the spatial dimensions of a 4D tensor by pooling over blocks of
    specified size. The last two dimensions (channel-like) are preserved.

    Args:
        fine_v: Input tensor with at least 4 spatial dimensions plus any trailing
            dimensions. Shape: (D1, D2, D3, D4, ...).
        block_size: Size of the pooling block for each of the 4 spatial dimensions.
            Must be a tuple of 4 integers.

    Returns:
        Pooled tensor with reduced spatial dimensions. Shape:
            (D1//block_size[0], D2//block_size[1], D3//block_size[2],
             D4//block_size[3], ...). The last two dimensions are preserved from
            the input. dtype is torch.cdouble.

    Note:
        This is a slow pure-Python implementation. For better performance,
        use the accelerated version in qcd_ml_accel.pool4d.
    """
    L_coarse = [li // bi for li, bi in zip(fine_v.shape[:-2], block_size)]
    res = torch.zeros(*L_coarse, *fine_v.shape[-2:], dtype=torch.cdouble)
    for x,y,z,t in product(*tuple([range(block_size[i]) for i in range(4)])):
        res += fine_v[x::block_size[0], y::block_size[1], z::block_size[2], t::block_size[3]]
    return res


def v_unpool4d(coarse_v: torch.Tensor, block_size: Tuple[int, ...]) -> torch.Tensor:
    """Performs 4D volume unpooling by expanding blocks.

    Expands the spatial dimensions of a 4D tensor by replicating values
    into blocks of specified size. The last two dimensions (channel-like)
    are preserved.

    Args:
        coarse_v: Input tensor with 4 spatial dimensions plus any trailing
            dimensions. Shape: (D1, D2, D3, D4, ...).
        block_size: Size of the unpooling block for each of the 4 spatial dimensions.
            Must be a tuple of 4 integers.

    Returns:
        Unpooled tensor with expanded spatial dimensions. Shape:
            (D1*block_size[0], D2*block_size[1], D3*block_size[2],
             D4*block_size[3], ...). The last two dimensions are preserved from
            the input. dtype is torch.cdouble.

    Note:
        This is a slow pure-Python implementation. For better performance,
        use the accelerated version in qcd_ml_accel.pool4d.
    """
    L_coarse = coarse_v.shape[:-2]
    res = torch.zeros(*[li*bi for li,bi in zip(L_coarse, block_size)], *coarse_v.shape[-2:], dtype=torch.cdouble)
    for x,y,z,t in product(*tuple([range(L_coarse[i]) for i in range(4)])):
        res[x*block_size[0]: (x + 1) * block_size[0]
            , y*block_size[1]: (y + 1) * block_size[1]
            , z*block_size[2]: (z + 1) * block_size[2]
            , t*block_size[3]: (t + 1) * block_size[3]] = coarse_v[x,y,z,t]
    return res
