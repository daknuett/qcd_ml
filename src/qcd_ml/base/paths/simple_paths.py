import torch
from typing import List, Tuple

from ..hop import v_hop, v_ng_hop, m_hop

def v_evaluate_path(U: torch.Tensor, path: List[Tuple[int, int]], v: torch.Tensor) -> torch.Tensor:
    """
    Gauge-equivariantly evaluate a path on a vector-like field.

    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.

    Args:
        U: Gauge field tensor of shape (4, Lx, Ly, Lz, Lt, 3, 3) where 4 is the number
            of spacetime dimensions.
        path: List of tuples (mu, nhops) specifying the path to evaluate.
            mu is the direction index, nhops is the number of hops.
        v: Input vector-like field tensor to evaluate the path on.
    """
    for mu, nhops in path:
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_hop(U, mu, direction, v)
    return v

def v_ng_evaluate_path(path: List[Tuple[int, int]], v: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a path on a vector-like field without gauge degrees of freedom.

    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.

    Args:
        path: List of tuples (mu, nhops) specifying the path to evaluate.
            mu is the direction index, nhops is the number of hops.
        v: Input vector-like field tensor to evaluate the path on.
    """
    if len(path) > 0:
        mus = [mu for mu,_ in path]
        hops = [nhops for _,nhops in path]
        return torch.roll(v, shifts=hops, dims=mus)
    return v


def slow_v_ng_evaluate_path(path: List[Tuple[int, int]], v: torch.Tensor) -> torch.Tensor:
    """
    Deprecated: Use v_ng_evaluate_path instead. Kept for testing.
    
    Evaluate a path on a vector-like field without gauge degrees of freedom.
    This is a slow implementation used only for testing.

    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.

    Args:
        path: List of tuples (mu, nhops) specifying the path to evaluate.
            mu is the direction index, nhops is the number of hops.
        v: Input vector-like field tensor to evaluate the path on.
    """
    for mu, nhops in path:
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_ng_hop(mu, direction, v)
    return v


def v_reverse_evaluate_path(U: torch.Tensor, path: List[Tuple[int, int]], v: torch.Tensor) -> torch.Tensor:
    """
    Gauge-equivariantly evaluate a path on a vector-like field.
    This is the inverse of ``v_evaluate_path``.

    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.

    Args:
        U: Gauge field tensor of shape (4, Lx, Ly, Lz, Lt, 3, 3) where 4 is the number
            of spacetime dimensions.
        path: List of tuples (mu, nhops) specifying the path to evaluate in reverse.
            mu is the direction index, nhops is the number of hops.
        v: Input vector-like field tensor to evaluate the path on.
    """
    for mu, nhops in reversed(path):
        nhops *= -1
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_hop(U, mu, direction, v)
    return v


def v_ng_reverse_evaluate_path(path: List[Tuple[int, int]], v: torch.Tensor) -> torch.Tensor:
    """
    Inverse of ``v_ng_evaluate_path``.
    """
    if len(path) > 0:
        mus = [mu for mu,_ in path]
        hops = [-nhops for _,nhops in path]
        return torch.roll(v, shifts=hops, dims=mus)
    return v


def slow_v_ng_reverse_evaluate_path(path: List[Tuple[int, int]], v: torch.Tensor) -> torch.Tensor:
    """
    Deprecated: Use v_ng_reverse_evaluate_path instead. Kept for testing.
    
    Inverse of ``v_ng_evaluate_path``. This is a slow implementation used only for testing.
    """
    for mu, nhops in reversed(path):
        nhops *= -1
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_ng_hop(mu, direction, v)
    return v


def m_evaluate_path(U: torch.Tensor, path: List[Tuple[int, int]], m: torch.Tensor) -> torch.Tensor:
    """
    Gauge-equivariantly evaluate a path on a matrix-like field.

    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.

    Args:
        U: Gauge field tensor of shape (4, Lx, Ly, Lz, Lt, 3, 3) where 4 is the number
            of spacetime dimensions.
        path: List of tuples (mu, nhops) specifying the path to evaluate.
            mu is the direction index, nhops is the number of hops.
        m: Input matrix-like field tensor to evaluate the path on.
    """
    for mu, nhops in path:
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            m = m_hop(U, mu, direction, m)
    return m


def m_reverse_evaluate_path(U: torch.Tensor, path: List[Tuple[int, int]], m: torch.Tensor) -> torch.Tensor:
    """
    Gauge-equivariantly evaluate a path on a matrix-like field.
    This is the inverse of ``m_evaluate_path``.
    """
    for mu, nhops in reversed(path):
        nhops *= -1
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            m = m_hop(U, mu, direction, m)
    return m
