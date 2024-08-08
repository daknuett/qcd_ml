import os
import sys

import torch
import numpy as np

sys.path.append("src/")
import qcd_ml
import qcd_ml.compat.gpt

import gpt

# global variable needed later
calls = 0

# parameters
fermion_mass = -0.58
nbasisvectors = 12
blocksize = [4,4,4,4]

# load the gauge field (both gpt as well as native versions)
U = torch.tensor(np.load(os.path.join("test", "assets", "1500.config.npy")))
U_gpt = gpt.load("/glurch/scratch/knd35666/ensembles/ens001/1500.config/")

# Wilson-clover Dirac operator
w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, fermion_mass, 1)
w_gpt = gpt.qcd.fermion.wilson_clover(U_gpt,
        {"mass": fermion_mass,
         "csw_r": 1,
         "csw_t": 1,
         "xi_0": 1,
         "nu": 1,
         "isAnisotropic": False,
         "boundary_phases": [1,1,1,1]})

# we use the gpt version for now
def w(x):
    global calls
    calls += 1
    return torch.tensor(qcd_ml.compat.gpt.lattice2ndarray(w_gpt(qcd_ml.compat.gpt.ndarray2lattice(x.numpy(), U_gpt[0].grid, gpt.vspincolor))))

# multigrid setup
def solve(b, x0):
    return qcd_ml.util.solver.GMRES_restarted(w, b, x0, eps=1e-3, maxiter_inner=25, max_restart=40)
rand_vecs = [torch.randn(8,8,8,16,4,3, dtype=torch.cdouble) for _ in range(nbasisvectors)]
zpp = qcd_ml.util.qcd.multigrid.ZPP_Multigrid.gen_from_fine_vectors(rand_vecs, blocksize, lambda b, x0: solve(b, x0), verbose=True)

torch.save(zpp.ui_blocked, "mg_setup.pt")


# in the following, we test the effectiveness of classical multigrid

# test vector
test_v = torch.ones(8,8,8,16,4,3, dtype=torch.cdouble)
test_v /= (test_v * test_v.conj()).real.sum()**0.5

# unpreconditioned iteration count
calls = 0
x, ret = qcd_ml.util.solver.GMRES_restarted(w, test_v, test_v, eps=1e-4, maxiter_inner=25, max_restart=400)
it_ref = ret["k"]
calls_ref = calls
print(f"Unpreconditioned iteration count: {it_ref}")
print(f"Unpreconditioned solver applications of D: {calls_ref}")

# coarse Wilson-clover Dirac operator
def w_coarse(x):
    global calls
    calls += 1
    return zpp.v_project(w(zpp.v_prolong(x)))

# classical multigrid
def mg(v):
    v_, ret = qcd_ml.util.solver.GMRES_torch(w, v, v, eps=1e-1, maxiter=25)

    res_coarse = zpp.v_project(w(v_) - v)
    zero_coarse = torch.zeros_like(res_coarse)

    v_coarse, ret_coarse = qcd_ml.util.solver.GMRES_torch(w_coarse, zero_coarse, res_coarse, eps=1e-3, maxiter=25)
    v_ = v_ + zpp.v_prolong(v_coarse)

    v_, ret_smoother = qcd_ml.util.solver.GMRES_torch(w, v, v_, eps=1e-3, maxiter=25)

    return v_

# preconditioned iteration count
calls = 0
x_p, ret_p = qcd_ml.util.solver.GMRES_restarted(w, test_v, test_v, prec=lambda v: mg(v), eps=1e-4, maxiter_inner=25, max_restart=400)
calls += ret_p["k"]
print(f"Preconditioned iteration count: {ret_p['k']}")
print(f"Preconditioned solver applications of D: {calls}")
print(f"Outer iteration count gain: {it_ref/ret_p['k']} ({it_ref}/{ret_p['k']})")
print(f"D-Application count gain: {calls_ref/calls} ({calls_ref}/{calls})")

