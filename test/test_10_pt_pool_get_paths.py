import torch
import pytest

from qcd_ml.nn.pt_pool.get_paths import get_paths_lexicographic, get_paths_reverse_lexicographic, get_paths_one_step_lexicographic, get_paths_one_step_reverse_lexicographic

def test_get_paths_lexicographic():
    block_size = [2, 2, 2, 2]
    paths = get_paths_lexicographic(block_size)

    assert len(paths) == 16

    for path in paths:
        for i, (mu, nhops) in enumerate(path):
            try:
                assert mu < path[i+1][0] 
            except IndexError:
                pass
    

def test_get_paths_reverse_lexicographic():
    block_size = [2, 2, 2, 2]
    paths = get_paths_reverse_lexicographic(block_size)

    assert len(paths) == 16

    for path in paths:
        for i, (mu, nhops) in enumerate(path):
            try:
                assert mu > path[i+1][0] 
            except IndexError:
                pass


def test_get_paths_one_step_lexicographic_reference():
    block_size = [4, 4, 4, 4]
    paths = get_paths_one_step_lexicographic(block_size)

    assert [(0, -1), (1, -1), (2, -1), (3, -1), (0, -1), (1, -1)] in paths


def test_get_paths_one_step_reverse_lexicographic_reference():
    block_size = [4, 4, 4, 4]
    paths = get_paths_one_step_reverse_lexicographic(block_size)

    assert [(3, -1), (2, -1), (1, -1), (0, -1), (1, -1), (0, -1)] in paths


try:
    import gpt as g
    from qcd_ml.nn.pt_pool import v_ProjectLayer
    import qcd_ml
    from qcd_ml.compat.gpt import lattice2ndarray, ndarray2lattice

    @pytest.mark.slow
    def test_paths_in_project_layer_against_gpt():
        U_trch = torch.ones(4, 8,8,8,16, 3,3, dtype=torch.cdouble)
        U_trch = torch.randn_like(U_trch)
        U_gpt = [ndarray2lattice(u.numpy(), g.grid([8,8,8,16], g.double), g.mcolor) for u in U_trch]

        grid = g.grid([8,8,8,16], g.double)
        psi = g.vspincolor(grid)
        coarse_grid = g.grid([2, 2, 2, 4], g.double)
        rng = g.random("test_ProjectLayer_against_gpt")

        ot_ci = g.ot_vector_spin_color(4, 3)
        ot_cw = g.ot_matrix_spin(4)

        L_fine = [8, 8, 8, 16]
        L_coarse = [2, 2, 2, 4]
        block_size = [4, 4, 4, 4]


        for get_paths_trch, get_path_gpt in [
                (get_paths_lexicographic, g.ml.layer.parallel_transport_pooling.path.lexicographic)
                , (get_paths_reverse_lexicographic, g.ml.layer.parallel_transport_pooling.path.reversed_lexicographic)
                , (get_paths_one_step_lexicographic, g.ml.layer.parallel_transport_pooling.path.one_step_lexicographic)
                , (get_paths_one_step_reverse_lexicographic, g.ml.layer.parallel_transport_pooling.path.one_step_reversed_lexicographic)
                ]:

            tfp = v_ProjectLayer([(U_trch, get_paths_trch(block_size, _gpt_compat=True))], L_fine, L_coarse, _gpt_compat=True)

            t = g.ml.layer.parallel_transport_pooling.transfer(
                grid,
                coarse_grid,
                ot_ci,
                ot_cw,
                [
                    (U_gpt, get_path_gpt)
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
    def test_paths_in_project_layer_against_gpt():
        pass
