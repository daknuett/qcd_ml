import torch 

import pytest

from qcd_ml.base.operations import SU3_group_compose, _es_SU3_group_compose
from qcd_ml.base.operations import v_gauge_transform, _es_v_gauge_transform
from qcd_ml.base.operations import v_spin_transform, _es_v_spin_transform


@pytest.mark.benchmark(group="group_operations")
def test_es_SU3_group_compose(config_1500, benchmark):
    expect = SU3_group_compose(config_1500[0], config_1500[1])

    got = benchmark(_es_SU3_group_compose, config_1500[0], config_1500[1])

    assert torch.allclose(expect, got)


@pytest.mark.benchmark(group="group_operations")
def test_SU3_group_compose(config_1500, benchmark):
    expect = _es_SU3_group_compose(config_1500[0], config_1500[1])

    got = benchmark(SU3_group_compose, config_1500[0], config_1500[1])

    assert torch.allclose(expect, got)


@pytest.mark.benchmark(group="group_operations")
def test_es_v_gauge_transform(psi_test, config_1500, benchmark):
    expect = v_gauge_transform(config_1500[0], psi_test)

    got = benchmark(_es_v_gauge_transform, config_1500[0], psi_test)

    assert torch.allclose(expect, got)


@pytest.mark.benchmark(group="group_operations")
def test_v_gauge_transform(psi_test, config_1500, benchmark):
    expect = _es_v_gauge_transform(config_1500[0], psi_test)

    got = benchmark(v_gauge_transform, config_1500[0], psi_test)

    assert torch.allclose(expect, got)


@pytest.mark.benchmark(group="group_operations")
def test_es_v_spin_transform(psi_test, benchmark):
    W = torch.randn(*psi_test.shape[:-2], 4, 4, dtype=torch.cdouble)
    expect = v_spin_transform(W, psi_test)

    got = benchmark(_es_v_spin_transform, W, psi_test)

    assert torch.allclose(expect, got)


@pytest.mark.benchmark(group="group_operations")
def test_v_spin_transform(psi_test, benchmark):
    W = torch.randn(*psi_test.shape[:-2], 4, 4, dtype=torch.cdouble)
    expect = _es_v_spin_transform(W, psi_test)

    got = benchmark(v_spin_transform, W, psi_test)

    assert torch.allclose(expect, got)
