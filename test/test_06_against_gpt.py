import numpy as np
import torch
from qcd_ml.qcd.dirac import dirac_wilson_clover, dirac_wilson
from qcd_ml.util.solver import GMRES
import pytest

try:
    import gpt as g
    from qcd_ml.compat.gpt import lattice2ndarray, ndarray2lattice


    def test_wilson(config_1500):
        grid = g.grid([8,8,8,16], g.double)
        psi = g.vspincolor(grid)
        U = [ndarray2lattice(Ui.numpy(), grid, g.mcolor) for Ui in config_1500]
        
        w_gpt = g.qcd.fermion.wilson_clover(U, {"mass": -0.5,
            "csw_r": 0.0,
            "csw_t": 0.0,
            "xi_0": 1.0,
            "nu": 1.0,
            "isAnisotropic": False,
            "boundary_phases": [1,1,1,1]})

        w_torch = dirac_wilson(config_1500, -0.5)
        w = lambda x: torch.tensor(lattice2ndarray(w_gpt(ndarray2lattice(x.numpy(), U[0].grid, g.vspincolor))))

        rng = g.random("test_wilson")
        rng.cnormal(psi)

        psi_torch = torch.tensor(lattice2ndarray(psi))

        assert torch.allclose(w_torch(psi_torch), w(psi_torch))


    def test_wilson_clover(config_1500):
        grid = g.grid([8,8,8,16], g.double)
        psi = g.vspincolor(grid)
        U = [ndarray2lattice(Ui.numpy(), grid, g.mcolor) for Ui in config_1500]
        
        w_gpt = g.qcd.fermion.wilson_clover(U, {"mass": -0.5,
            "csw_r": 1.0,
            "csw_t": 1.0,
            "xi_0": 1.0,
            "nu": 1.0,
            "isAnisotropic": False,
            "boundary_phases": [1,1,1,1]})

        w_torch = dirac_wilson_clover(config_1500, -0.5, 1.0)
        w = lambda x: torch.tensor(lattice2ndarray(w_gpt(ndarray2lattice(x.numpy(), U[0].grid, g.vspincolor))))

        rng = g.random("test_wilson_clover")
        rng.cnormal(psi)

        psi_torch = torch.tensor(lattice2ndarray(psi))

        assert torch.allclose(w_torch(psi_torch), w(psi_torch))


    def test_wilson_clover2(config_1500):
        grid = g.grid([8,8,8,16], g.double)
        psi = g.vspincolor(grid)
        U = [ndarray2lattice(Ui.numpy(), grid, g.mcolor) for Ui in config_1500]
        
        w_gpt = g.qcd.fermion.wilson_clover(U, {"mass": -0.5,
            "csw_r": 1.0,
            "csw_t": 1.0,
            "xi_0": 1.0,
            "nu": 1.0,
            "isAnisotropic": False,
            "boundary_phases": [1,1,1,1]})

        w_torch = dirac_wilson_clover(config_1500, -0.5, 1.0)
        w_for_gpt = lambda x: ndarray2lattice(w_torch(torch.tensor(lattice2ndarray(x))).numpy(), U[0].grid, g.vspincolor)

        rng = g.random("test_wilson_clover")
        rng.cnormal(psi)

        Dpsi_gpt = w_gpt(g.copy(psi))
        Dpsi_passthrough = w_for_gpt(g.copy(psi))

        assert g.norm2(Dpsi_gpt - Dpsi_passthrough) / g.norm2(Dpsi_gpt) < 1e-15


    def test_gmres(config_1500):
        grid = g.grid([8,8,8,16], g.double)
        psi = g.vspincolor(grid)
        U = [ndarray2lattice(Ui.numpy(), grid, g.mcolor) for Ui in config_1500]
        
        w_gpt = g.qcd.fermion.wilson_clover(U, {"mass": 0.55,
            "csw_r": 1.0,
            "csw_t": 1.0,
            "xi_0": 1.0,
            "nu": 1.0,
            "isAnisotropic": False,
            "boundary_phases": [1,1,1,1]})

        w_torch = dirac_wilson_clover(config_1500, 0.58, 1.0)
        w = lambda x: torch.tensor(lattice2ndarray(w_gpt(ndarray2lattice(x.numpy(), U[0].grid, g.vspincolor))))

        rng = g.random("test_gmres")
        rng.cnormal(psi)
        
        psi_torch = torch.tensor(lattice2ndarray(psi))
        x_my, ret = GMRES(w_gpt, g.copy(psi), g.copy(psi), maxiter=300, eps=1e-7, inner_iter=30, innerproduct=lambda x,y: g.inner_product(x,y))
        slv = g.algorithms.inverter.fgmres(eps=1e-7, maxiter=300, restartlen=30)
        x_gpt = slv(w_gpt)(g.copy(psi), g.copy(psi))

        assert g.norm2(x_my - x_gpt) < 1e-8
        assert g.norm2(w_gpt(x_my) - psi) < 1e-8
        gpt_hist = np.array(slv.history)
        assert np.allclose(ret["history"][:gpt_hist.shape[0]], gpt_hist**0.5)


    def test_gmres2(config_1500):
        grid = g.grid([8,8,8,16], g.double)
        psi = g.vspincolor(grid)
        U = [ndarray2lattice(Ui.numpy(), grid, g.mcolor) for Ui in config_1500]
        
        w_gpt = g.qcd.fermion.wilson_clover(U, {"mass": -0.55,
            "csw_r": 1.0,
            "csw_t": 1.0,
            "xi_0": 1.0,
            "nu": 1.0,
            "isAnisotropic": False,
            "boundary_phases": [1,1,1,1]})

        w_torch = dirac_wilson_clover(config_1500, -0.55, 1.0)
        w = lambda x: torch.tensor(lattice2ndarray(w_gpt(ndarray2lattice(x.numpy(), U[0].grid, g.vspincolor))))

        rng = g.random("test_gmres")
        rng.cnormal(psi)
        
        psi_torch = torch.tensor(lattice2ndarray(psi))
        x_my, ret = GMRES(w, torch.clone(psi_torch), torch.clone(psi_torch), maxiter=900, eps=1e-7, inner_iter=30)
        slv = g.algorithms.inverter.fgmres(eps=1e-7, maxiter=900, restartlen=30)
        x_gpt = slv(w_gpt)(g.copy(psi), g.copy(psi))

        assert torch.sum(torch.abs(x_my - torch.tensor(lattice2ndarray(x_gpt)))) < 1e-8

        gpt_hist = np.array(slv.history)
        assert np.allclose(ret["history"][:gpt_hist.shape[0]], gpt_hist**0.5)


    def test_gmres_approx_solving(config_1500):
        grid = g.grid([8,8,8,16], g.double)
        psi = g.vspincolor(grid)
        U = [ndarray2lattice(Ui.numpy(), grid, g.mcolor) for Ui in config_1500]
        
        w_gpt = g.qcd.fermion.wilson_clover(U, {"mass": -0.5,
            "csw_r": 0.0,
            "csw_t": 0.0,
            "xi_0": 1.0,
            "nu": 1.0,
            "isAnisotropic": False,
            "boundary_phases": [1,1,1,1]})

        w_torch = dirac_wilson(config_1500, -0.5)
        w = lambda x: torch.tensor(lattice2ndarray(w_gpt(ndarray2lattice(x.numpy(), U[0].grid, g.vspincolor))))

        rng = g.random("test_gmres_approx_solving")
        rng.cnormal(psi)
        
        psi_torch = torch.tensor(lattice2ndarray(psi))
        x_torch, _ret = GMRES(w_torch, psi_torch, psi_torch, maxiter=300, eps=1e-3, inner_iter=30)

        slv = g.algorithms.inverter.fgmres(eps=1e-3, maxiter=300, restartlen=30)
        x_gpt = slv(w_gpt)(psi, psi)

        assert torch.allclose(x_torch, torch.tensor(lattice2ndarray(x_gpt)))


    @pytest.mark.slow
    def test_gmres_large_test_matrix(config_1500):
        grid = g.grid([8,8,8,16], g.double)
        psi = g.vspincolor(grid)
        coarse_grid = g.grid([2, 2, 2, 4], g.double)
        rng = g.random("test")
        U = [ndarray2lattice(Ui.numpy(), grid, g.mcolor) for Ui in config_1500]

        w_gpt = g.qcd.fermion.wilson_clover(    
            U,
            {    
                "mass": -0.55,
                "csw_r": 1,    
                "csw_t": 1,    
                "xi_0": 1,    
                "nu": 1,    
                "isAnisotropic": True,    
                "boundary_phases": [1.0, 1.0, 1.0, 1.0], # for now only periodic allowed since PT uses periodic!    
            },
        )


        print("==" * 30)
        print("TEST MATRIX:")

        print( 
        '''
        GMRES from GPT
              Dirac operator from >  |   GPT    |   TORCH
              -----------------------+----------+--------
              Inner product from \\/  |          |        
              GPT                    |    0     |    1   
              TORCH                  |    -     |    -    

        GMRES from TORCH
              Dirac operator from >  |   GPT    |   TORCH
              -----------------------+----------+--------
              Inner product from \\/  |          |        
              GPT                    |    A     |    B   
              TORCH                  |    C     |    D    

        '''
        )

        torch.manual_seed(0xdeadbeef)

        rng.cnormal(psi)

        U_torch = config_1500

        psi_torch = torch.tensor(lattice2ndarray(psi))
        print("==" * 30)


        w_passthrough = lambda x: w_gpt(ndarray2lattice(lattice2ndarray(x), U[0].grid, g.vspincolor))
        w = lambda x: torch.tensor(lattice2ndarray(w_gpt(ndarray2lattice(x.numpy(), U[0].grid, g.vspincolor))))

        w_torch = dirac_wilson_clover(U_torch, -0.55, 1.0)
        w_for_gpt = lambda x: ndarray2lattice(w_torch(torch.tensor(lattice2ndarray(x))).numpy(), U[0].grid, g.vspincolor)

        def mat_for_gpt(dst, src):
            dst @= w_for_gpt(src)

        solver = g.algorithms.inverter.fgmres({"eps": 1e-7, "maxiter": 900, "restartlen": 30}) 

        print("      EXPERIMENT     <<0>>")
        psi_0 = solver(w_gpt)(g.copy(psi), g.copy(psi))
        print("(0) Ax - b (rel):", g.norm2(w_gpt(psi_0) - psi) / g.norm2(psi))

        print("      EXPERIMENT     <<1>>")
        psi_1 = solver(mat_for_gpt)(g.copy(psi), g.copy(psi))
        print("(1) Ax - b (rel):", g.norm2(w_gpt(psi_1) - psi) / g.norm2(psi))

        print("==" * 30)

        print("      EXPERIMENT     <<A>>")
        psi_A, ret_A = GMRES(w_gpt, g.copy(psi), g.copy(psi), maxiter=900, inner_iter=30, eps=1e-7, innerproduct=lambda x,y: g.inner_product(x, y))
        print("converged:", ret_A["converged"], "residual:", ret_A["res"], "iterations:", ret_A["k"], "target_residual:", ret_A["target_residual"])
        print("(A) Ax - b (rel):", g.norm2(w_gpt(psi_A) - psi) / g.norm2(psi))


        print("==" * 30)
        print("      EXPERIMENT     <<C>>")
        # Inner product from TORCH; dirac operator from GPT
        psi_C, ret_C = GMRES(w, torch.clone(psi_torch), torch.clone(psi_torch), maxiter=900, eps=1e-7, inner_iter=30)
        print("converged:", ret_C["converged"], "residual:", ret_C["res"], "iterations:", ret_C["k"], "target_residual:", ret_C["target_residual"])
        print("(C) Ax - b (rel):", g.norm2(w_gpt(ndarray2lattice(psi_C.numpy(), U[0].grid, g.vspincolor)) - psi) / g.norm2(psi))



        print("==" * 30)
        print("      EXPERIMENT     <<D>>")
        # Inner product from TORCH; dirac operator from TORCH
        psi_D, ret_D = GMRES(w_torch, torch.clone(psi_torch), torch.clone(psi_torch), maxiter=900, eps=1e-7, inner_iter=30)
        print("converged:", ret_D["converged"], "residual:", ret_D["res"], "iterations:", ret_D["k"], "target_residual:", ret_D["target_residual"])
        print("(D) Ax - b (rel):", g.norm2(w_gpt(ndarray2lattice(psi_D.numpy(), U[0].grid, g.vspincolor)) - psi))

        print("==" * 30)
        print("      EXPERIMENT     <<B>>")
        # Inner product from GPT; dirac operator from TORCH
        psi_B, ret_B = GMRES(w_torch, torch.clone(psi_torch), torch.clone(psi_torch), maxiter=900, eps=1e-7, inner_iter=30
                              , innerproduct=lambda x,y: g.inner_product(ndarray2lattice(x.numpy(), U[0].grid, g.vspincolor), ndarray2lattice(y.numpy(), U[0].grid, g.vspincolor)))
        print("converged:", ret_B["converged"], "residual:", ret_B["res"], "iterations:", ret_B["k"], "target_residual:", ret_B["target_residual"])

        print("(B) Ax - b (rel):", g.norm2(w_gpt(ndarray2lattice(psi_B.numpy(), U[0].grid, g.vspincolor)) - psi))

        def complex_mse_loss(output, target):
            err = (output - target)
            return (err * err.conj()).real.sum()

        def l2norm(v):
            return (v * v.conj()).real.sum()

        print("==" * 30)

        diff_01 = g.norm2(psi_0 - psi_1) / g.norm2(psi_0)
        diff_0A = g.norm2(psi_0 - psi_A) / g.norm2(psi_0)
        diff_0B = g.norm2(psi_0 - ndarray2lattice(psi_B.numpy(), U[0].grid, g.vspincolor)) / g.norm2(psi_0)
        diff_0C = g.norm2(psi_0 - ndarray2lattice(psi_C.numpy(), U[0].grid, g.vspincolor)) / g.norm2(psi_0)
        diff_0D = g.norm2(psi_0 - ndarray2lattice(psi_D.numpy(), U[0].grid, g.vspincolor)) / g.norm2(psi_0)


        print(f"difference in result (0 - 1) (rel): {diff_01:.4e}")
        print()

        print(f"difference in result (0 - A) (rel): {diff_0A:.4e}")
        print("difference in result (0 - B) (rel): "
              , f"{diff_0B:.4e}")
        print("difference in result (0 - C) (rel): "
              , f"{diff_0C:.4e}")
        print("difference in result (0 - D) (rel): "
              , f"{diff_0D:.4e}")

        print()

        diff_1A = g.norm2(psi_1 - psi_A) / g.norm2(psi_1)
        diff_1B = g.norm2(psi_1 - ndarray2lattice(psi_B.numpy(), U[1].grid, g.vspincolor)) / g.norm2(psi_1)
        diff_1C = g.norm2(psi_1 - ndarray2lattice(psi_C.numpy(), U[1].grid, g.vspincolor)) / g.norm2(psi_1)
        diff_1D = g.norm2(psi_1 - ndarray2lattice(psi_D.numpy(), U[1].grid, g.vspincolor)) / g.norm2(psi_1)

        print(f"difference in result (1 - A) (rel): {diff_1A:.4e}")
        print("difference in result (1 - B) (rel): "
              , f"{diff_1B:.4e}")
        print("difference in result (1 - C) (rel): "
              , f"{diff_1C:.4e}")
        print("difference in result (1 - D) (rel): "
              , f"{diff_1D:.4e}")

        print()

        diff_AB = g.norm2(psi_A - ndarray2lattice(psi_B.numpy(), U[0].grid, g.vspincolor)) / g.norm2(psi_A)
        diff_AC = g.norm2(psi_A - ndarray2lattice(psi_C.numpy(), U[0].grid, g.vspincolor)) / g.norm2(psi_A)
        diff_AD = g.norm2(psi_A - ndarray2lattice(psi_D.numpy(), U[0].grid, g.vspincolor)) / g.norm2(psi_A)

        diff_CD = complex_mse_loss(psi_C, psi_D) / l2norm(psi_C)
        diff_BC = complex_mse_loss(psi_B, psi_C) / l2norm(psi_C)
        diff_BD = complex_mse_loss(psi_B, psi_D) / l2norm(psi_B)

        print("difference in result:")
        print(
        f'''
                   B             C             D
        A  {diff_AB: .4e}      {diff_AC: .4e}    {diff_AD: .4e} 
        B                   {diff_BC: .4e}    {diff_BD: .4e} 
        C                                  {diff_CD: .4e} 
        D
        ''')

        for diff in (
                diff_01, diff_0A, diff_0B, diff_0C, diff_0D
                , diff_1A, diff_1B, diff_1C, diff_1D
                , diff_AB, diff_AC, diff_AD
                , diff_BC, diff_BD, diff_CD
                ):
            assert diff < 1e-15

except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_wilson(config_1500):
        pass

    @pytest.mark.skip("missing gpt")
    def test_wilson_clover(config_1500):
        pass
    
    @pytest.mark.skip("missing gpt")
    def test_wilson_clover2(config_1500):
        pass
    
    @pytest.mark.skip("missing gpt")
    def test_gmres(config_1500):
        pass

    @pytest.mark.skip("missing gpt")
    def test_gmres_approx_solving(config_1500):
        pass

    @pytest.mark.skip("missing gpt")
    def test_gmres_large_test_matrix(config_1500):
        pass
