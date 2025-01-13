import torch
import numpy as np 

import pytest

from qcd_ml.base.paths import v_evaluate_path, m_evaluate_path
from qcd_ml.base.paths import v_reverse_evaluate_path
from qcd_ml.base.paths import PathBuffer
from qcd_ml.base.operations import v_gauge_transform, link_gauge_transform, m_gauge_transform


def test_path_buffer_00(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = []

    pb = PathBuffer(config_1500, path)

    assert torch.allclose(pb.v_transport(psi), v_evaluate_path(config_1500, path, psi))


def test_path_buffer_01(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1)]

    pb = PathBuffer(config_1500, path)

    assert torch.allclose(pb.v_transport(psi), v_evaluate_path(config_1500, path, psi))


def test_path_buffer_02(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,-1)]

    pb = PathBuffer(config_1500, path)

    assert torch.allclose(pb.v_transport(psi), v_evaluate_path(config_1500, path, psi))


def test_path_buffer_03(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1), (1,-2), (3,3), (2,-5)]

    pb = PathBuffer(config_1500, path)

    assert torch.allclose(pb.v_transport(psi), v_evaluate_path(config_1500, path, psi))


def test_path_buffer_m03(config_1500):
    m = torch.randn_like(config_1500[0])
    path = [(0,1), (1,-2), (3,3), (2,-5)]

    pb = PathBuffer(config_1500, path)

    assert torch.allclose(pb.m_transport(m), m_evaluate_path(config_1500, path, m))


def test_path_buffer_reverse(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1), (1,-2), (3,3), (2,-5)]

    pb = PathBuffer(config_1500, path)

    assert torch.allclose(pb.v_reverse_transport(psi), v_reverse_evaluate_path(config_1500, path, psi))


def test_path_buffer_equivariance_v(config_1500, psi_test, V_1500mu0_1500mu2):
    psi_test = torch.randn_like(psi_test)
    V = V_1500mu0_1500mu2

    path = [(0,1), (2, -2), (3,1)]
    pb = PathBuffer(config_1500, path)
    
    psibar_ngt = pb.v_transport(psi_test)
    psibar_gta = v_gauge_transform(V, psibar_ngt)

    U = link_gauge_transform(config_1500, V)
    psi_test_gt = v_gauge_transform(V, psi_test)
    pb_trans = PathBuffer(U, path)
    psibar_gtb = pb_trans.v_transport(psi_test_gt)
    
    assert torch.allclose(psibar_gtb, psibar_gta)


def test_path_buffer_equivariance_m(config_1500, V_1500mu0_1500mu2):
    M_test = torch.randn(8,8,8,16, 3,3, dtype=torch.cdouble)
    V = V_1500mu0_1500mu2

    path = [(0,1), (2, -2), (3,1)]
    pb = PathBuffer(config_1500, path)
    
    Mbar_ngt = pb.m_transport(M_test)
    Mbar_gta = m_gauge_transform(V, Mbar_ngt)

    U = link_gauge_transform(config_1500, V)
    M_test_gt = m_gauge_transform(V, M_test)
    pb_trans = PathBuffer(U, path)
    Mbar_gtb = pb_trans.m_transport(M_test_gt)
    
    assert torch.allclose(Mbar_gtb, Mbar_gta)
