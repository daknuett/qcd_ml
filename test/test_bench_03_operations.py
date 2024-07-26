import torch 

import pytest

from qcd_ml.base.operations import SU3_group_compose, _es_SU3_group_compose


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
