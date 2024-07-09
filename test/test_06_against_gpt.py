import numpy as np
import torch
from qcd_ml.dirac import dirac_wilson_clover, dirac_wilson

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

        print(w_torch(psi_torch)[0,0,0,0])
        print(w(psi_torch)[0,0,0,0])

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

        print(w_torch(psi_torch)[0,0,0,0])
        print(w(psi_torch)[0,0,0,0])

        assert torch.allclose(w_torch(psi_torch), w(psi_torch))
except ImportError:
    pass

