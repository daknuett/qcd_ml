import torch
import numpy as np 

import pytest

from qcd_ml.base.paths import v_evaluate_path
from qcd_ml.base.paths import v_reverse_evaluate_path
from qcd_ml.base.paths import PathBuffer


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


def test_path_buffer_reverse(config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1), (1,-2), (3,3), (2,-5)]

    pb = PathBuffer(config_1500, path)

    assert torch.allclose(pb.v_reverse_transport(psi), v_reverse_evaluate_path(config_1500, path, psi))
