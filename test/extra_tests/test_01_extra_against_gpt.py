import qcd_ml
import numpy as np
import pytest

import torch

try:
    import gpt as g
    import qcd_ml.compat.gpt

    def test_innerproduct():
        """
        Test the inner product between two vectors against the gpt implementation.
        The inner product is used for instance in the GMRES algorithm.
        """
        innerproduct = lambda x,y: (x.conj() * y).sum()

        vec = torch.randn(8,8,8,16, 4,3, dtype=torch.cdouble)
        vec2 = torch.randn_like(vec)

        grid = g.grid([8,8,8,16], g.double)
        rng = g.random("deadbeef")


        vec_gpt = qcd_ml.compat.gpt.ndarray2lattice(vec.numpy(), grid, g.vspincolor)
        vec2_gpt = qcd_ml.compat.gpt.ndarray2lattice(vec2.numpy(), grid, g.vspincolor)

        val_qcdml = innerproduct(vec, vec2)

        val_gpt = g.inner_product(vec_gpt, vec2_gpt)

        print(abs(val_qcdml - val_gpt) / abs(val_gpt))
        assert abs(val_qcdml - val_gpt) / abs(val_gpt) < 1e-14


    def test_norm2():
        """
        Test the norm2 of a vector against the gpt implementation.
        The norm2 is used for instance in the GMRES algorithm.
        """
        innerproduct = lambda x,y: (x.conj() * y).sum()

        vec3 = torch.randn(8,8,8,16, 4,3, dtype=torch.cdouble)

        grid = g.grid([8,8,8,16], g.double)
        rng = g.random("deadbeef")

        vec3_gpt = qcd_ml.compat.gpt.ndarray2lattice(vec3.numpy(), grid, g.vspincolor)
        norm2 = lambda x: innerproduct(x, x)

        val_qcdml = norm2(vec3)

        val_gpt = g.norm2(vec3_gpt)

        assert abs(val_qcdml - val_gpt) / abs(val_gpt) < 1e-14


    def test_l2norm():
        def l2norm(v):
            return (v * v.conj()).real.sum()

        vec = torch.randn(8,8,8,16, 4,3, dtype=torch.cdouble)

        grid = g.grid([8,8,8,16], g.double)

        vec_gpt = qcd_ml.compat.gpt.ndarray2lattice(vec.numpy(), grid, g.vspincolor)

        val_qcdml = l2norm(vec)

        val_gpt = g.norm2(vec_gpt)

        assert abs(val_qcdml - val_gpt) / abs(val_gpt) < 1e-14


    def test_mse_loss():
        """
        Tests the complex mean squared error loss against the gpt implementation.
        """
        def complex_mse_loss(output, target):
            err = (output - target)
            return (err * err.conj()).real.sum()

        vec = torch.randn(8,8,8,16, 4,3, dtype=torch.cdouble)
        vec2 = torch.randn_like(vec)

        grid = g.grid([8,8,8,16], g.double)

        vec_gpt = qcd_ml.compat.gpt.ndarray2lattice(vec.numpy(), grid, g.vspincolor)
        vec2_gpt = qcd_ml.compat.gpt.ndarray2lattice(vec2.numpy(), grid, g.vspincolor)

        model = g.ml.model.sequence([])

        rng = g.random("deadbeef")
        W = model.random_weights(rng)

        cost = model.cost([vec_gpt], [vec2_gpt])

        assert abs(cost(W) - complex_mse_loss(vec, vec2)) / abs(cost(W)) < 1e-14

except ImportError:
    @pytest.mark.skip(reason="gpt not available")
    def test_innerproduct():
        pass

    @pytest.mark.skip(reason="gpt not available")
    def test_norm2():
        pass

    @pytest.mark.skip(reason="gpt not available")
    def test_l2norm():
        pass

    @pytest.mark.skip(reason="gpt not available")
    def test_mse_loss():
        pass
