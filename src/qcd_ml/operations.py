#!/usr/bin/env python3

import torch

def SU3_group_compose(A, B):
    return torch.einsum("abcdij,abcdjk->abcdik", A, B)


def v_gauge_transform(Umu, v):
    return torch.einsum("abcdij,abcdSj->abcdSi", Umu, v)


def v_spin_transform(M, v):
    return torch.einsum("abcdij,abcdjG->abcdiG", M, v)


def v_spin_const_transform(M, v):
    return torch.einsum("ij,abcdjG->abcdiG", M, v)


def link_gauge_transform(U, V):
    Vdg = V.adjoint()
    U_trans = [SU3_group_compose(V, Umu) for Umu in U]
    for mu, U_transmu in enumerate(U_trans):
        U_trans[mu] = SU3_group_compose(U_transmu, torch.roll(Vdg, -1, mu))
    return U_trans
