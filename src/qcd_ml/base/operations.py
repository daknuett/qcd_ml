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

"""


import torch

def _mul(iterable):
    res = 1
    for i in iterable:
        res *= i
    return res

@torch.compile
def _es_SU3_group_compose(A, B):
    return torch.einsum("abcdij,abcdjk->abcdik", A, B)


@torch.compile
def SU3_group_compose(A, B):
    """
    SU(3) group composition
    """
    vol = _mul(A.shape[:4])
    old_shape = A.shape
    return torch.bmm(A.reshape((vol, *(A.shape[4:])))
                     , B.reshape((vol, *(A.shape[4:])))).reshape(old_shape)


@torch.compile
def _es_v_gauge_transform(Umu, v):
    return torch.einsum("abcdij,abcdSj->abcdSi", Umu, v)


@torch.compile
def v_gauge_transform(Umu, v):
    """
    Gauge transformation of vector-like fields.
    """
    vol = _mul(v.shape[:4])
    old_shape = v.shape
    return torch.bmm(Umu.reshape((vol, *(Umu.shape[4:])))
                     , v.reshape((vol, *(v.shape[4:]))).transpose(-1, -2)
                     ).transpose(-1, -2).reshape(old_shape)


@torch.compile
def _es_v_spin_transform(M, v):
    return torch.einsum("abcdij,abcdjG->abcdiG", M, v)

@torch.compile
def v_spin_transform(M, v):
    """
    Applies a spin matrix field to a vector field.
    """
    vol = _mul(v.shape[:4])
    old_shape = v.shape
    return torch.bmm(M.reshape((vol, *(M.shape[4:])))
                     , v.reshape((vol, *(v.shape[4:])))
                     ).reshape(old_shape)

@torch.compile
def v_spin_const_transform(M, v):
    """
    Applies a spin matrix to a vector field.
    """
    return torch.einsum("ij,abcdjG->abcdiG", M, v)


@torch.compile
def v_ng_spin_transform(M, v):
    """
    Applies a spin matrix field to a vector field without gauge freedom.
    """
    return torch.einsum("abcdij,abcdj->abcdi", M, v)


@torch.compile
def v_ng_spin_const_transform(M, v):
    """
    Applies a spin matrix to a vector field without gauge freedom.
    """
    return torch.einsum("ij,abcdj->abcdi", M, v)


@torch.compile
def link_gauge_transform(U, V):
    """
    Gauge-transforms a link-like field.
    A link-like is typically a gauge configuration.
    """
    Vdg = V.adjoint()
    U_trans = [SU3_group_compose(V, Umu) for Umu in U]
    for mu, U_transmu in enumerate(U_trans):
        U_trans[mu] = SU3_group_compose(U_transmu, torch.roll(Vdg, -1, mu))
    return U_trans

@torch.compile
def mspin_const_group_compose(A, B):
    """
    Matrix-matrix multiplication for spin matrices.
    """
    return torch.einsum("ij,jk->ik", A, B)
