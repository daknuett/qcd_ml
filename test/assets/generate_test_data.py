import gpt as g 
import numpy as np 

def lattice2ndarray(lattice):
    """ 
    Converts a gpt (https://github.com/lehner/gpt) lattice to a numpy ndarray 
    keeping the ordering of axes as one would expect.
    Example::
        q_top = g.qcd.gauge.topological_charge_5LI(U_smeared, field=True)
        plot_scalar_field(lattice2ndarray(q_top))
    """
    shape = lattice.grid.fdimensions
    shape = list(reversed(shape))
    if lattice[:].shape[1:] != (1,):
        shape.extend(lattice[:].shape[1:])
   
    result = lattice[:].reshape(shape)
    result = np.swapaxes(result, 0, 3)
    result = np.swapaxes(result, 1, 2)
    return result

U1 = g.load("/home/knd35666/data/ensembles/ens_001/1500.config")
U2 = g.load("/home/knd35666/data/ensembles/ens_001/1200.config")
V = g(U1[0] * U1[2])
V_np = lattice2ndarray(V)
np.save("V_1500mu0_1500mu2.npy", V_np)

psi = g.vspincolor(U1[0].grid)
psi[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
psibar = g(U1[0] * psi)
np.save("psi_test.npy", lattice2ndarray(psi))
np.save("psi_1500mu0_psitest.npy", lattice2ndarray(psibar))



U1500_adj = g(g.adj(U1))
np.save("1500_adj.npy", [lattice2ndarray(U1500_adjmu) for U1500_adjmu in U1500_adj])

U1500_gtrans_1200_mu0 = g.qcd.gauge.transformed(U1, U2[0])
np.save("1500_gtrans_1200mu0.npy", [lattice2ndarray(Umu) for Umu in U1500_gtrans_1200_mu0])

w = g.qcd.fermion.wilson_clover(U1, {"mass": -0.5,
    "csw_r": 0.0,
    "csw_t": 0.0,
    "xi_0": 1.0,
    "nu": 1.0,
    "isAnisotropic": False,
    "boundary_phases": [1,1,1,1]})

psi_Dw_m0p5_psitest = w(psi)
np.save("psi_Dw1500_m0p5_psitest.npy", lattice2ndarray(psi_Dw_m0p5_psitest))

w = g.qcd.fermion.wilson_clover(U1, {"mass": -0.5,
    "csw_r": 1.0,
    "csw_t": 1.0,
    "xi_0": 1.0,
    "nu": 1.0,
    "isAnisotropic": False,
    "boundary_phases": [1,1,1,1]})

psi_Dwc_m0p5_psitest = w(psi)
np.save("psi_Dwc1500_m0p5_psitest.npy", lattice2ndarray(psi_Dwc_m0p5_psitest))
