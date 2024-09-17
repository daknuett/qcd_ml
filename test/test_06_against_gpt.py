import numpy as np
import torch
from qcd_ml.qcd.dirac import dirac_wilson_clover, dirac_wilson
from qcd_ml.util.solver import GMRES_torch
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


    def test_gmres(config_1500):
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

        rng = g.random("test_gmres")
        rng.cnormal(psi)
        
        psi_torch = torch.tensor(lattice2ndarray(psi))
        x_torch, _ret = GMRES_torch(w_torch, psi_torch, psi_torch, maxiter=1000, eps=1e-9)

        slv = g.algorithms.inverter.fgmres(eps=1e-9, maxiter=1000, restartlen=3000)
        x_gpt = slv(w_gpt)(psi, psi)

        assert torch.allclose(x_torch, torch.tensor(lattice2ndarray(x_gpt)))


except ImportError:

    @pytest.mark.skip("missing gpt")
    def test_wilson(config_1500):
        pass

    @pytest.mark.skip("missing gpt")
    def test_wilson_clover(config_1500):
        pass
    
    @pytest.mark.skip("missing gpt")
    def test_gmres(config_1500):
        pass
