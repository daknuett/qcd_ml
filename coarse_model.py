import os
import sys

import torch
import numpy as np

sys.path.append("src/")
import qcd_ml
import qcd_ml.compat.gpt

import gpt

# parameters
fermion_mass = -0.58
nbasisvectors = 12
blocksize = [4,4,4,4]

# load the gauge field (both gpt as well as native versions)
U = torch.tensor(np.load(os.path.join("test", "assets", "1500.config.npy")))
U_gpt = gpt.load("/glurch/scratch/knd35666/ensembles/ens001/1500.config/")

# coarse grid 1 hop paths
paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(3, -1)]]

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
w = lambda x: torch.tensor(qcd_ml.compat.gpt.lattice2ndarray(w_gpt(qcd_ml.compat.gpt.ndarray2lattice(x.numpy(), U_gpt[0].grid, gpt.vspincolor))))

# coarse Wilson-clover Dirac operator
w_coarse = lambda x: zpp.v_project(w(zpp.v_prolong(x)))

# load multigrid setup
ui_blocked = torch.load("mg_setup.pt")
grid = list(U[0].shape[:-2])
coarse_grid = [a//b for a, b in zip(grid, blocksize)]
zpp = qcd_ml.util.qcd.multigrid.ZPP_Multigrid(blocksize, ui_blocked, nbasisvectors, coarse_grid, grid)

# create coarse LPTC layer
layer = qcd_ml.nn.lptc.v_LPTC_NG(1, 1, paths, zpp.L_coarse, nbasisvectors)

# initialize weights
layer.weights.data = 0.01 * torch.randn_like(layer.weights.data, dtype=torch.cdouble)
layer.weights.data[:,:,0] += torch.eye(nbasisvectors)

# function to calculate mse
def complex_mse(output, target):
    err = (output - target)
    return (err * err.conj()).real.sum()

# function to calculate l2norm
def l2norm(v):
    return (v * v.conj()).real.sum()

# define optimizer and training parameters
training_epochs = 1000
check_every = 100
adam_lr = 1e-2
optimizer = torch.optim.Adam(layer.parameters(), lr=adam_lr)

# logging
cost = np.zeros(training_epochs)
its = np.zeros(training_epochs // check_every + 1)

# test vector
test_v = torch.ones(*zpp.L_coarse, nbasisvectors, dtype=torch.cdouble)
test_v /= l2norm(test_v)**0.5

# unpreconditioned iteration count
x, ret = qcd_ml.util.solver.GMRES_restarted(w_coarse, test_v, test_v, eps=1e-4, maxiter_inner=25, max_restart=400)
it_ref = ret["k"]
print(f"Unpreconditioned iteration count: {it_ref}")

# training
for t in range(training_epochs):
    source = torch.randn(*zpp.L_coarse, nbasisvectors, dtype=torch.cdouble)
    Dsource = w_coarse(source)

    source_norm = l2norm(Dsource)
    in_vec = torch.stack([Dsource / source_norm])
    out_vec = torch.stack([source / source_norm])

    curr_cost = complex_mse(layer.forward(in_vec), out_vec)
    cost[t] = curr_cost.item()
    optimizer.zero_grad()
    curr_cost.backward()
    optimizer.step()
    print(f"{t} - {cost[t]}")

    if t % check_every == 0:
        with torch.no_grad():
            x_p, ret_p = qcd_ml.util.solver.GMRES_restarted(w_coarse, test_v, test_v, prec=lambda v: layer.forward(torch.stack([v]))[0], eps=1e-4, maxiter_inner=25, max_restart=400)
            its[t // check_every] = ret_p["k"]
            print(f"\tICG: {it_ref / its[t//check_every]} ({it_ref}/{its[t//check_every]})")

# calculate final iteration count
with torch.no_grad():
    x_p, ret_p = qcd_ml.util.solver.GMRES_restarted(w_coarse, test_v, test_v, prec=lambda v: layer.forward(torch.stack([v]))[0], eps=1e-4, maxiter_inner=25, max_restart=400)
    its[-1] = ret_p["k"]
    print(f"\tICG: {it_ref / its[-1]} ({it_ref}/{its[-1]})")

torch.save(list(layer.parameters()), "coarse_model.pt")

