import pytest
import torch 

from qcd_ml.util.qcd.multigrid import ZPP_Multigrid
from qcd_ml.util.solver import GMRES_restarted
from qcd_ml.qcd.dirac import dirac_wilson_clover

@pytest.fixture
def test_mm_setup(config_1500):
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
             , lambda b,x0: GMRES_restarted(w, b, x0, eps=1e-3, maxiter_inner=20, max_restart=5)
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
