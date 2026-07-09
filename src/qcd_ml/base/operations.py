#!/usr/bin/env python3
"""
qcd_ml.base.operations
======================

Provides

- matrix-matrix multiplication for
  - SU3 fields
  - spin matrices
- gauge transformation of
  - vector-like fields
  - link-like fields
- group action of
  - spin matrices on vector-like fields
  - spin fields on vector-like fields

See also: :ref:`doc-datatypes:qcd_ml Datatypes`.
"""


import torch
from typing import Iterable


def _mul(iterable: Iterable[int]) -> int:
    """
    Compute the product of all elements in an iterable.
    
    Args:
        iterable: An iterable of integers.
        
    Returns:
        The product of all elements.
    """
    res = 1
    for i in iterable:
        res *= i
    return res


def _es_SU3_group_compose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Einstein summation implementation of SU3 group composition.
    
    Args:
        A: First SU(3) field tensor.
        B: Second SU(3) field tensor.
        
    Returns:
        The composed SU(3) field tensor.
    """
    return torch.einsum("abcdij,abcdjk->abcdik", A, B)


def SU3_group_compose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    :math:`SU(3)` group composition of two :math:`SU(3)` fields.
    
    Args:
        A: First SU(3) field tensor of shape (Lx, Ly, Lz, Lt, 3, 3).
        B: Second SU(3) field tensor of shape (Lx, Ly, Lz, Lt, 3, 3).
        
    Returns:
        The composed SU(3) field tensor of shape (Lx, Ly, Lz, Lt, 3, 3).
    """
    vol = _mul(A.shape[:4])
    old_shape = A.shape
    return torch.bmm(A.reshape((vol, *(A.shape[4:])))
                     , B.reshape((vol, *(A.shape[4:])))).reshape(old_shape)


def _es_v_gauge_transform(Umu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Einstein summation implementation of gauge transformation for vector-like fields.
    
    Args:
        Umu: SU(3) gauge field tensor.
        v: Vector-like field tensor.
        
    Returns:
        Gauge-transformed vector field tensor.
    """
    return torch.einsum("abcdij,abcdSj->abcdSi", Umu, v)


def v_gauge_transform(Umu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Gauge transformation of vector-like fields.
    
    Args:
        Umu: SU(3) gauge field tensor of shape (4, Lx, Ly, Lz, Lt, 3, 3).
        v: Vector-like field tensor of shape (Lx, Ly, Lz, Lt, 4, 3).
        
    Returns:
        Gauge-transformed vector field tensor of shape (Lx, Ly, Lz, Lt, 4, 3).
    """
    vol = _mul(v.shape[:4])
    old_shape = v.shape
    return torch.bmm(Umu.reshape((vol, *(Umu.shape[4:])))
                     , v.reshape((vol, *(v.shape[4:]))).transpose(-1, -2)
                     ).transpose(-1, -2).reshape(old_shape)


def _es_v_spin_transform(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Einstein summation implementation of spin transformation for vector-like fields.
    
    Args:
        M: Spin matrix field tensor.
        v: Vector field tensor.
        
    Returns:
        Transformed vector field tensor.
    """
    return torch.einsum("abcdij,abcdjG->abcdiG", M, v)


def v_spin_transform(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Applies a spin matrix field to a vector field.
    
    Args:
        M: Spin matrix field tensor of shape (Lx, Ly, Lz, Lt, 4, 4).
        v: Vector field tensor of shape (Lx, Ly, Lz, Lt, 4, Nc).
        
    Returns:
        Transformed vector field tensor of shape (Lx, Ly, Lz, Lt, 4, Nc).
    """
    vol = _mul(v.shape[:4])
    old_shape = v.shape
    return torch.bmm(M.reshape((vol, *(M.shape[4:])))
                     , v.reshape((vol, *(v.shape[4:])))
                     ).reshape(old_shape)


def v_spin_const_transform(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Applies a spin matrix to a vector field.
    
    Args:
        M: Spin matrix tensor of shape (4, 4).
        v: Vector field tensor of shape (Lx, Ly, Lz, Lt, 4, Nc).
        
    Returns:
        Transformed vector field tensor of shape (Lx, Ly, Lz, Lt, 4, Nc).
    """
    return torch.einsum("ij,abcdjG->abcdiG", M, v)


def v_ng_spin_transform(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Applies a spin matrix field to a vector field without gauge freedom.
    
    Args:
        M: Spin matrix field tensor of shape (Lx, Ly, Lz, Lt, 4, 4).
        v: Vector field tensor of shape (Lx, Ly, Lz, Lt, 4).
        
    Returns:
        Transformed vector field tensor of shape (Lx, Ly, Lz, Lt, 4).
    """
    return torch.einsum("abcdij,abcdj->abcdi", M, v)


def v_ng_spin_const_transform(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Applies a spin matrix to a vector field without gauge freedom.
    
    Args:
        M: Spin matrix tensor of shape (4, 4).
        v: Vector field tensor of shape (Lx, Ly, Lz, Lt, 4).
        
    Returns:
        Transformed vector field tensor of shape (Lx, Ly, Lz, Lt, 4).
    """
    return torch.einsum("ij,abcdj->abcdi", M, v)


def link_gauge_transform(U: list[torch.Tensor], V: torch.Tensor) -> list[torch.Tensor]:
    """
    Gauge-transforms a link-like field.
    A link-like field is typically a gauge configuration.
    
    Args:
        U: List of gauge field tensors, one for each direction.
        V: Gauge transformation matrix tensor of shape (Lx, Ly, Lz, Lt, 3, 3).
        
    Returns:
        List of gauge-transformed gauge field tensors.
    """
    Vdg = V.adjoint()
    U_trans = [SU3_group_compose(V, Umu) for Umu in U]
    for mu, U_transmu in enumerate(U_trans):
        U_trans[mu] = SU3_group_compose(U_transmu, torch.roll(Vdg, -1, mu))
    return U_trans


def mspin_const_group_compose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix-matrix multiplication for spin matrices.
    
    Args:
        A: First spin matrix tensor of shape (4, 4).
        B: Second spin matrix tensor of shape (4, 4).
        
    Returns:
        The product spin matrix tensor of shape (4, 4).
    """
    return torch.einsum("ij,jk->ik", A, B)


def _es_m_gauge_transform(Umu: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Einstein summation implementation of gauge transformation for matrix-like fields.
    
    Args:
        Umu: SU(3) gauge field tensor.
        m: Matrix-like field tensor.
        
    Returns:
        Gauge-transformed matrix field tensor.
    """
    return torch.einsum("abcdij,abcdjk,abcdkl->abcdil", Umu, m, Umu.adjoint())


def m_gauge_transform(Umu: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Gauge transformation of matrix-like fields.
    
    Args:
        Umu: SU(3) gauge field tensor of shape (4, Lx, Ly, Lz, Lt, 3, 3).
        m: Matrix-like field tensor of shape (Lx, Ly, Lz, Lt, 3, 3).
        
    Returns:
        Gauge-transformed matrix field tensor of shape (Lx, Ly, Lz, Lt, 3, 3).
    """
    vol = _mul(m.shape[:4])
    old_shape = m.shape
    Umu_reshaped = Umu.reshape((vol, *(Umu.shape[4:])))
    return torch.bmm(torch.bmm(Umu_reshaped
                     , m.reshape((vol, *(m.shape[4:])))), Umu_reshaped.adjoint()).reshape(old_shape)
