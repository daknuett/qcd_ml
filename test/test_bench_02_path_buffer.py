import torch
import numpy as np 

import pytest

from qcd_ml.base.paths import v_evaluate_path
from qcd_ml.base.paths import v_reverse_evaluate_path
from qcd_ml.base.paths import PathBuffer

def test_path_buffer_00(benchmark, config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = []

    pb = PathBuffer(config_1500, path)
    expect =  v_evaluate_path(config_1500, path, psi)
    got = benchmark(pb.v_transport, psi)

    assert torch.allclose(expect, got,)


def test_path_buffer_01(benchmark, config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1)]

    pb = PathBuffer(config_1500, path)
    expect =  v_evaluate_path(config_1500, path, psi)
    got = benchmark(pb.v_transport, psi)

    assert torch.allclose(expect, got)


def test_path_buffer_02(benchmark, config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1), (1,-1)]

    pb = PathBuffer(config_1500, path)
    expect =  v_evaluate_path(config_1500, path, psi)
    got = benchmark(pb.v_transport, psi)

    assert torch.allclose(expect, got)

def test_path_buffer_03(benchmark, config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1), (1,-1), (3, 3)]

    pb = PathBuffer(config_1500, path)
    expect =  v_evaluate_path(config_1500, path, psi)
    got = benchmark(pb.v_transport, psi)

    assert torch.allclose(expect, got)


def test_path_00(benchmark, config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = []

    pb = PathBuffer(config_1500, path)
    got =  benchmark(v_evaluate_path, config_1500, path, psi)
    expect = pb.v_transport(psi)

    assert torch.allclose(expect, got)


def test_path_01(benchmark, config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1)]

    pb = PathBuffer(config_1500, path)
    got =  benchmark(v_evaluate_path, config_1500, path, psi)
    expect = pb.v_transport(psi)

    assert torch.allclose(expect, got)


def test_path_02(benchmark, config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1), (1,-1)]

    pb = PathBuffer(config_1500, path)
    got =  benchmark(v_evaluate_path, config_1500, path, psi)
    expect = pb.v_transport(psi)

    assert torch.allclose(expect, got)


def test_path_03(benchmark, config_1500, psi_test):
    psi = torch.randn_like(psi_test)
    path = [(0,1), (1,-1), (3, 3)]

    pb = PathBuffer(config_1500, path)
    got =  benchmark(v_evaluate_path, config_1500, path, psi)
    expect = pb.v_transport(psi)

    assert torch.allclose(expect, got)
