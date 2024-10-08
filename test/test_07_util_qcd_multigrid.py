import pytest
import torch 

from qcd_ml.util.qcd.multigrid import ZPP_Multigrid
from qcd_ml.util.solver import GMRES
from qcd_ml.qcd.dirac import dirac_wilson_clover

@pytest.fixture(scope="session")
def test_mm_setup(config_1500_sess):
    config_1500 = config_1500_sess
    psi = torch.complex(
            torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double)
            , torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double))

    n_basis = 4
    bv = [torch.randn_like(psi) for _ in range(n_basis)] 

    block_size = [4, 4, 4, 4]

    w = dirac_wilson_clover(config_1500, -0.58, 1.0)

    mm = ZPP_Multigrid.gen_from_fine_vectors(
             bv
             , block_size
             , lambda b,x0: GMRES(w, b, x0, eps=1e-3, maxiter=300, inner_iter=30)
             , verbose=False)
    return mm


@pytest.fixture
def rand_fine_vec():
    psi = torch.complex(
            torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double)
            , torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double))
    return psi


@pytest.mark.slow
def test_MM_is_Id_on_coarse(test_mm_setup, rand_fine_vec):
    coarse_vec = test_mm_setup.v_project(rand_fine_vec)

    coarse_vec2 = test_mm_setup.v_project(test_mm_setup.v_prolong(coarse_vec))

    assert torch.allclose(coarse_vec, coarse_vec2)


@pytest.mark.slow
def test_MM_is_Id_on_fine(test_mm_setup):
    basis_vecs = test_mm_setup.get_basis_vectors()

    for fine_vec in basis_vecs:
        fine_vec2 = test_mm_setup.v_prolong(test_mm_setup.v_project(fine_vec))

        assert torch.allclose(fine_vec, fine_vec2)


@pytest.mark.slow
def test_MM_save_load(test_mm_setup, tmpdir):
    test_mm_setup.save(tmpdir / "test_mm_setup.pt")
    mm2 = ZPP_Multigrid.load(tmpdir / "test_mm_setup.pt")

    assert test_mm_setup.block_size == mm2.block_size
    assert test_mm_setup.n_basis == mm2.n_basis
    assert test_mm_setup.L_coarse == mm2.L_coarse
    assert test_mm_setup.L_fine == mm2.L_fine
    #assert test_mm_setup.ui_blocked == mm2.ui_blocked
