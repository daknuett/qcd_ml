import torch
import numpy as np
import pytest


from qcd_ml.nn.pt_pool import v_ProjectLayer
from qcd_ml.nn.pt_pool.get_paths import get_paths_lexicographic, get_paths_reverse_lexicographic, get_paths_one_step_lexicographic



from qcd_ml.base.paths import PathBuffer
from qcd_ml.base.operations import v_spin_transform
class v_ParallelTransportPool(torch.nn.Module):
    def __init__(self, paths, U):
        super().__init__()
        self.weights = torch.nn.Parameter(
                torch.complex(torch.randn(1, 1, *tuple(U.shape[1:-2]), 4, 4, dtype=torch.double)
                              , torch.randn(1, 1, *tuple(U.shape[1:-2]), 4, 4, dtype=torch.double))
                )

        self.path_buffers = [PathBuffer(U, pi) for pi in paths]

    
    def forward(self, features_in):
        if features_in.shape[0] != 1:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {1}")
        
        features_out = torch.zeros_like(features_in)

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi in self.path_buffers:
                    features_out[io] = features_out[io] + pi.v_transport(v_spin_transform(wfo, fi))

        return features_out

    
    def reverse(self, features_in):
        if features_in.shape[0] != 1:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {1}")
        
        features_out = torch.zeros_like(features_in)

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi in self.path_buffers:
                    features_out[io] = features_out[io] + v_spin_transform(wfo.adjoint(), pi.v_reverse_transport(fi))

        return features_out



class v_PoolingLayer(torch.nn.Module):
    def __init__(self, gauges_and_paths):
        super().__init__()
        self.layers = torch.nn.ModuleList([v_ParallelTransportPool(Pi, Ui) for Ui, Pi in gauges_and_paths])


    def pool(self, features_in):
        result = torch.zeros_like(features_in)
        for li in self.layers:
            result = result + li.forward(features_in)
        return result

    def de_pool(self, features_in):
        result = torch.zeros_like(features_in)
        for li in self.layers:
            result = result + li.reverse(features_in)
        return result
        
class v_SubSampling(torch.nn.Module):
    def __init__(self, L_fine, L_coarse):
        super().__init__()
        self.L_fine = L_fine
        self.L_coarse = L_coarse
        self.block_size = [lf // lc for lf, lc in zip(L_fine, L_coarse)]

    def v_project(self, features):
        res = torch.complex(
            torch.zeros(features.shape[0], *self.L_coarse, *features.shape[-2:], dtype=torch.double)
            , torch.zeros(features.shape[0], *self.L_coarse, *features.shape[-2:], dtype=torch.double)
        )
        res = res + features[:,::self.block_size[0], ::self.block_size[1], ::self.block_size[2], ::self.block_size[3]]
        return res

    def v_prolong(self, features):
        res = torch.complex(
            torch.zeros(features.shape[0], *self.L_fine, *features.shape[-2:], dtype=torch.double)
            , torch.zeros(features.shape[0], *self.L_fine, *features.shape[-2:], dtype=torch.double)
        )
        res[:,::self.block_size[0], ::self.block_size[1], ::self.block_size[2], ::self.block_size[3]] = res[:,::self.block_size[0], ::self.block_size[1], ::self.block_size[2], ::self.block_size[3]] + features
        return res

class v_PathProjectLayer(torch.nn.Module):
    def __init__(self, gauges_and_paths, L_fine, L_coarse):
        super().__init__()
        self.pooling = v_PoolingLayer(gauges_and_paths)
        self.subsampling = v_SubSampling(L_fine, L_coarse)

    def parameters(self):
        yield from self.pooling.parameters()

    def project(self, features_in):
        return self.subsampling.v_project(self.pooling.pool(features_in))

    def prolong(self, features_in):
        return self.pooling.de_pool(self.subsampling.v_prolong(features_in))

def test_ProjectLayer_against_reference_impl(config_1500):
    L_fine = [8, 8, 8, 16]
    L_coarse = [2, 2, 2, 4]
    block_size = [4, 4, 4, 4]

    tfp = v_ProjectLayer([(config_1500, get_paths_lexicographic(block_size))
                          , (config_1500, get_paths_reverse_lexicographic(block_size))], L_fine, L_coarse)
    p_tfp = v_PathProjectLayer([(config_1500, get_paths_lexicographic(block_size))
                            , (config_1500, get_paths_reverse_lexicographic(block_size))], L_fine, L_coarse)

    with torch.no_grad():
        for i, wi in enumerate(p_tfp.parameters()):
            tfp.weights.data[i] = wi[0,0]



    fine_v = torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)
    coarse_v = tfp.v_project(torch.stack([fine_v]))
    p_coarse_v = p_tfp.project(torch.stack([fine_v]))

    assert torch.allclose(coarse_v, p_coarse_v)


try:
    import gpt as g
    from qcd_ml.compat.gpt import ndarray2lattice, lattice2ndarray

    def test_ProjectLayer_against_gpt(config_1500):
        grid = g.grid([8,8,8,16], g.double)
        psi = g.vspincolor(grid)
        coarse_grid = g.grid([2, 2, 2, 4], g.double)
        rng = g.random("test_ProjectLayer_against_gpt")

        ot_ci = g.ot_vector_spin_color(4, 3)
        ot_cw = g.ot_matrix_spin(4)
        U = [ndarray2lattice(Ui.numpy(), grid, g.mcolor) for Ui in config_1500]

        L_fine = [8, 8, 8, 16]
        L_coarse = [2, 2, 2, 4]
        block_size = [4, 4, 4, 4]

        tfp = v_ProjectLayer([(config_1500, get_paths_reverse_lexicographic(block_size))], L_fine, L_coarse, _gpt_compat=True)

        t = g.ml.layer.parallel_transport_pooling.transfer(
            grid,
            coarse_grid,
            ot_ci,
            ot_cw,
            [
                (U, g.ml.layer.parallel_transport_pooling.path.lexicographic)
            ],
            ot_embedding=g.ot_matrix_spin_color(4, 3),
            projector=g.ml.layer.projector_color_trace,
        )

        n = g.ml.model.sequence(g.ml.layer.parallel_transport_pooling.project(t))
        W = n.random_weights(rng)

        rng.cnormal(psi)

        wghts = next(tfp.parameters())
        wghts.data[0] = torch.tensor(lattice2ndarray(W[0]))
        assert len(list(tfp.parameters())) == 1
        assert len(W) == 1


        psi_torch = torch.tensor(lattice2ndarray(psi))


        coarse_v = tfp.v_project(torch.stack([psi_torch]))
        gpt_coarse_v = g.ml.layer.parallel_transport_pooling.project(t)(W, psi)

        assert torch.allclose(coarse_v, torch.tensor(lattice2ndarray(gpt_coarse_v)))



except ImportError:

    @pytest.mark.skip(reason="gpt not available")
    def test_ProjectLayer_against_gpt(config_1500):
        pass
