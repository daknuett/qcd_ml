import torch
import numpy as np
import pytest


from qcd_ml.nn.pt_pool.pool4d import v_pool4d, v_unpool4d


try:
    from qcd_ml_accel.pool4d import v_pool4d as v_pool4d_accel


    @pytest.mark.benchmark(group="pool4d")
    def test_pool4d_py(benchmark, psi_test):
        psi = torch.randn_like(psi_test)
        block_size = (4,4,4,4)
        got = benchmark(v_pool4d, psi, block_size)

        assert got.shape == tuple(np.array(psi.shape) // np.array(block_size + (1,1)))


    @pytest.mark.benchmark(group="pool4d")
    def test_pool4d_accel(benchmark, psi_test):
        psi = torch.randn_like(psi_test)
        block_size = (4,4,4,4)
        got = benchmark(v_pool4d_accel, psi, torch.tensor(block_size))
        expect = v_pool4d(psi, block_size)

        assert torch.allclose(expect, got)

except ImportError:
    @pytest.mark.skip("missing qcd_ml_accel.pool4d")
    def test_pool4d_py():
        pass


    @pytest.mark.skip("missing qcd_ml_accel.pool4d")
    def test_pool4d_accel():
        pass
