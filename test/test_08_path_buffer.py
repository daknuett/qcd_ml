import torch
import numpy as np 

import pytest

from qcd_ml.base.paths import v_evaluate_path
from qcd_ml.base.paths import v_reverse_evaluate_path
from qcd_ml.base.paths import PathBuffer
from qcd_ml.base.operations import v_gauge_transform, link_gauge_transform


def test_path_buffer_00(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = []

    pb = PathBuffer(config_1500)
    pt = pb.path(path)

    assert torch.allclose(pt.v_transport(psi), v_evaluate_path(config_1500, path, psi))


def test_path_buffer_01(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1)]

    pb = PathBuffer(config_1500)
    pt = pb.path(path)

    assert torch.allclose(pt.v_transport(psi), v_evaluate_path(config_1500, path, psi))


def test_path_buffer_02(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,-1)]

    pb = PathBuffer(config_1500)
    pt = pb.path(path)

    assert torch.allclose(pt.v_transport(psi), v_evaluate_path(config_1500, path, psi))


def test_path_buffer_03(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1), (1,-2), (3,3), (2,-5)]

    pb = PathBuffer(config_1500)
    pt = pb.path(path)

    assert torch.allclose(pt.v_transport(psi), v_evaluate_path(config_1500, path, psi))


def test_path_buffer_reverse(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1), (1,-2), (3,3), (2,-5)]

    pb = PathBuffer(config_1500)
    pt = pb.path(path)

    assert torch.allclose(pt.v_reverse_transport(psi), v_reverse_evaluate_path(config_1500, path, psi))


def test_path_buffer_equivariance(config_1500, psi_test, V_1500mu0_1500mu2):
    psi_test = torch.randn_like(psi_test)
    V = V_1500mu0_1500mu2

    path = [(0,1), (2, -2), (3,1)]
    pb = PathBuffer(config_1500)
    pt = pb.path(path)
    
    psibar_ngt = pt.v_transport(psi_test)
    psibar_gta = v_gauge_transform(V, psibar_ngt)

    U = link_gauge_transform(config_1500, V)
    psi_test_gt = v_gauge_transform(V, psi_test)
    pb_trans = PathBuffer(U)
    psibar_gtb = pb_trans.path(path).v_transport(psi_test_gt)
    
    assert torch.allclose(psibar_gtb, psibar_gta)
