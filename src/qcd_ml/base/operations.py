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
    """
    res = 1
    for i in iterable:
        res *= i
    return res


def _es_SU3_group_compose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Einstein summation implementation of SU3 group composition.
    Used for internal testing purposes.
    """
    return torch.einsum("abcdij,abcdjk->abcdik", A, B)


def SU3_group_compose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    :math:`SU(Nc)` group composition of two :math:`SU(Nc)` fields.
    """
    vol = _mul(A.shape[:4])
    old_shape = A.shape
    return torch.bmm(A.reshape((vol, *(A.shape[4:])))
                     , B.reshape((vol, *(B.shape[4:])))).reshape(old_shape)


def _es_v_gauge_transform(Umu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Einstein summation implementation of gauge transformation for vector-like fields.
    Used for internal testing purposes.
    """
    return torch.einsum("abcdij,abcdSj->abcdSi", Umu, v)


def v_gauge_transform(Umu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r"""
    Gauge transformation of vector-like fields, i.e.,

    .. math::

        v(x) \rightarrow \Omega(x) v(x)
    """
    vol = _mul(v.shape[:4])
    old_shape = v.shape
    return torch.bmm(Umu.reshape((vol, *(Umu.shape[4:])))
                     , v.reshape((vol, *(v.shape[4:]))).transpose(-1, -2)
                     ).transpose(-1, -2).reshape(old_shape)


def _es_v_spin_transform(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Einstein summation implementation of spin transformation for vector-like fields.
    Used for internal testing purposes.
    """
    return torch.einsum("abcdij,abcdjG->abcdiG", M, v)


def v_spin_transform(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r"""
    Applies a spin matrix field to a vector field.

    .. math::

        v(x) \rightarrow M(x) v(x)
    """
    vol = _mul(v.shape[:4])
    old_shape = v.shape
    return torch.bmm(M.reshape((vol, *(M.shape[4:])))
                     , v.reshape((vol, *(v.shape[4:])))
                     ).reshape(old_shape)


def v_spin_const_transform(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r"""
    Applies a spin matrix to a vector field, i.e.,

    .. math::

        v(x) \rightarrow M v(x)
    """
    return torch.einsum("ij,abcdjG->abcdiG", M, v)


def v_ng_spin_transform(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Applies a spin matrix field to a vector field without gauge freedom.
    """
    return torch.einsum("abcdij,abcdj->abcdi", M, v)


def v_ng_spin_const_transform(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Applies a spin matrix to a vector field without gauge freedom.
    """
    return torch.einsum("ij,abcdj->abcdi", M, v)


def link_gauge_transform(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    r"""
    Gauge-transforms a link-like field e.g. gauge configuration.

    .. math::

        U_\mu(x) \rightarrow V(x) U_\mu(x) V^\dagger(x + \mu)
    """
    Vdg = V.adjoint()
    # U is already a tensor of shape (4, Lx, Ly, Lz, Lt, Nc, Nc)
    # so U[mu] gives us the mu-th direction
    U_trans = torch.zeros_like(U)
    for mu in range(4):
        U_trans[mu] = SU3_group_compose(V, U[mu])
        U_trans[mu] = SU3_group_compose(U_trans[mu], torch.roll(Vdg, -1, mu))
    return U_trans


def mspin_const_group_compose(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix-matrix multiplication for spin matrices.
    """
    return torch.einsum("ij,jk->ik", A, B)


def _es_m_gauge_transform(Umu: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Einstein summation implementation of gauge transformation for matrix-like fields.
    Used for internal testing purposes.
    """
    return torch.einsum("abcdij,abcdjk,abcdkl->abcdil", Umu, m, Umu.adjoint())


def m_gauge_transform(Umu: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    r"""
    Gauge transformation of matrix-like fields.

    .. math::

        M(x) \rightarrow  U_\mu(x) M(x) U_\mu^\dagger(x)
    """
    vol = _mul(m.shape[:4])
    old_shape = m.shape
    Umu_reshaped = Umu.reshape((vol, *(Umu.shape[4:])))
    return torch.bmm(torch.bmm(Umu_reshaped
                     , m.reshape((vol, *(m.shape[4:])))), Umu_reshaped.adjoint()).reshape(old_shape)
