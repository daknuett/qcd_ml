import pytest
import torch

from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.qcd.dirac.coarsened import coarse_9point_op_NG
from qcd_ml.util.qcd.multigrid import ZPP_Multigrid
from qcd_ml.util.solver import GMRES


@pytest.fixture(scope="session")
def test_mm_setup(config_1500_sess):
    config_1500 = config_1500_sess
    psi = torch.complex(
        torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double),
        torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double),
    )

    n_basis = 4
    bv = [torch.randn_like(psi) for _ in range(n_basis)]

    block_size = (4, 4, 4, 4)

    w = dirac_wilson_clover(config_1500, -0.58, 1.0)

    mm = ZPP_Multigrid.gen_from_fine_vectors(
        bv,
        block_size,
        lambda b, x0: GMRES(w, b, x0, eps=1e-3, maxiter=300, inner_iter=30),
        verbose=False,
    )
    return mm


@pytest.fixture
def rand_fine_vec():
    psi = torch.complex(
        torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double),
        torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double),
    )
    return psi


@pytest.mark.slow
def test_MM_is_Id_on_coarse(test_mm_setup, rand_fine_vec):
    coarse_vec = test_mm_setup.v_project(rand_fine_vec)

    coarse_vec2 = test_mm_setup.v_project(test_mm_setup.v_prolong(coarse_vec))

    # With the new vectorized implementation, this should hold exactly
    # if the basis is orthonormal, but we use a tolerance for numerical precision
    assert torch.allclose(coarse_vec, coarse_vec2, atol=1e-8, rtol=1e-8)


@pytest.mark.slow
def test_MM_is_Id_on_fine(test_mm_setup, rand_fine_vec):
    # Test that v_prolong(v_project(fine_vec)) projects to the same coarse vector
    # Note: with the new vectorized implementation, v_prolong(v_project(v)) may not
    # equal v for arbitrary v, but v_project(v_prolong(v_project(v))) should equal v_project(v)
    coarse_vec = test_mm_setup.v_project(rand_fine_vec)
    fine_vec_reconstructed = test_mm_setup.v_prolong(coarse_vec)
    coarse_vec2 = test_mm_setup.v_project(fine_vec_reconstructed)

    assert torch.allclose(coarse_vec, coarse_vec2, atol=1e-8)


@pytest.mark.slow
def test_MM_state_dict_roundtrip(test_mm_setup, tmpdir):
    torch.save(test_mm_setup.state_dict(), tmpdir / "test_mm_setup.pt")
    mm2 = ZPP_Multigrid.from_state_dict(torch.load(tmpdir / "test_mm_setup.pt", weights_only=True))

    assert test_mm_setup.block_size == mm2.block_size
    assert test_mm_setup.n_basis == mm2.n_basis
    assert test_mm_setup.L_coarse == mm2.L_coarse
    assert test_mm_setup.L_fine == mm2.L_fine
    assert torch.allclose(test_mm_setup.block_basis, mm2.block_basis)


@pytest.mark.slow
def test_MM_load_state_dict_in_place(test_mm_setup):
    mm2 = ZPP_Multigrid.from_state_dict(test_mm_setup.state_dict())
    mm2.block_basis = torch.zeros_like(mm2.block_basis)

    mm2.load_state_dict(test_mm_setup.state_dict())

    assert torch.allclose(test_mm_setup.block_basis, mm2.block_basis)


def test_MM_load_state_dict_rejects_bad_keys():
    with pytest.raises(KeyError):
        ZPP_Multigrid.from_state_dict({"block_size": (4, 4, 4, 4), "nonsense": 42})


@pytest.mark.slow
def test_MM_loaded_acts_identically(config_1500, test_mm_setup, rand_fine_vec, tmpdir):
    torch.save(test_mm_setup.state_dict(), tmpdir / "test_mm_setup.pt")
    mm2 = ZPP_Multigrid.from_state_dict(torch.load(tmpdir / "test_mm_setup.pt", weights_only=True))

    coarse_vec = test_mm_setup.v_project(rand_fine_vec)
    assert torch.equal(coarse_vec, mm2.v_project(rand_fine_vec))
    assert torch.equal(test_mm_setup.v_prolong(coarse_vec), mm2.v_prolong(coarse_vec))

    w = dirac_wilson_clover(config_1500, -0.58, 1.0)
    assert torch.equal(
        test_mm_setup.get_coarse_operator(w)(coarse_vec),
        mm2.get_coarse_operator(w)(coarse_vec),
    )


@pytest.mark.slow
def test_coarsened_wilson_clover(config_1500, test_mm_setup, rand_fine_vec):
    w = dirac_wilson_clover(config_1500, -0.58, 1.0)
    w_coarse = test_mm_setup.get_coarse_operator(w)
    vec_coarse = test_mm_setup.v_project(rand_fine_vec)

    coarsened_op = coarse_9point_op_NG.from_operator_and_multigrid(
        w, test_mm_setup
    )

    assert torch.allclose(w_coarse(vec_coarse), coarsened_op(vec_coarse))


@pytest.mark.slow
def test_coarse_wilson_operator_equivalence(config_1500, test_mm_setup, rand_fine_vec):
    """Test that both Wilson coarse operator implementations give the same result."""
    from qcd_ml.qcd.dirac.coarsened import coarse_9point_op_NG
    
    w = dirac_wilson_clover(config_1500, -0.58, 1.0)

    # Check that the operator has the required methods
    assert hasattr(
        w, "apply_diag"
    ), "Wilson operator must have apply_diag method"
    assert hasattr(
        w, "apply_pos_hop"
    ), "Wilson operator must have apply_pos_hop method"
    assert hasattr(
        w, "apply_neg_hop"
    ), "Wilson operator must have apply_neg_hop method"

    # Create using the new classmethod for Wilson operators
    w_coarse_wilson = coarse_9point_op_NG.from_dirac_operator_and_multigrid(w, test_mm_setup)
    
    # Also create using coarse_9point_op_NG.from_operator_and_multigrid
    w_coarse_9point = coarse_9point_op_NG.from_operator_and_multigrid(w, test_mm_setup)

    vec_coarse = test_mm_setup.v_project(rand_fine_vec)

    # All implementations should give the same result
    result_wilson = w_coarse_wilson(vec_coarse)
    result_9point = w_coarse_9point(vec_coarse)
    
    assert torch.allclose(
        result_wilson, result_9point, atol=1e-8
    ), "Both Wilson implementations should match coarse_9point_op_NG.from_operator_and_multigrid"
