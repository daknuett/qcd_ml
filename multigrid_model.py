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

# 1 hop fine paths
paths_fine = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]

# 1 hop coarse paths
paths_coarse = [[]] + [[(mu, 1)] for mu in range(4)] + [[(3, -1)]]

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

# Smoother model
class Smoother(torch.nn.Module):
    def __init__(self, U, paths):
        super(Smoother, self).__init__()

        self.U = U
        self.paths = paths
        self.l0 = qcd_ml.nn.ptc.v_PTC(2,2,self.paths,self.U)
        self.l1 = qcd_ml.nn.ptc.v_PTC(2,2,self.paths,self.U)
        self.l2 = qcd_ml.nn.ptc.v_PTC(2,2,self.paths,self.U)
        self.l3 = qcd_ml.nn.ptc.v_PTC(2,1,self.paths,self.U)

    def forward(self, v):
        v = self.l0(v)
        v = self.l1(v)
        v = self.l2(v)
        v = self.l3(v)
        return v

# Multigrid model
class Multigrid(torch.nn.Module):
    def __init__(self, U, paths_fine, paths_coarse, zpp):
        super(Multigrid, self).__init__()

        self.U = U 
        self.paths_fine = paths_fine
        self.paths_coarse = paths_coarse
        self.zpp = zpp 
        self.coarse_model = qcd_ml.nn.lptc.v_LPTC_NG(1,1,self.paths_coarse,zpp.L_coarse, nbasisvectors)
        self.smoother = Smoother(U, paths_fine)

    def forward(self, v):
        v_c = self.zpp.v_project(v)
        v_c = self.coarse_model.forward(torch.stack([v_c]))[0]
        v2 = self.zpp.v_prolong(v_c)
        return self.smoother.forward(torch.stack([v, v2]))[0]

# load the smoother weights
weights_smoother = [torch.nn.Parameter(e, requires_grad=False) for e in torch.load("smoother.pt", weights_only=True)]

# load the coarse grid weights
weights_coarse = torch.nn.Parameter(torch.load("coarse_model.pt", weights_only=True)[0], requires_grad=False)

# create the multigrid model
mg_model = Multigrid(U, paths_fine, paths_coarse, zpp)

# set the correct weights
for li, wi in zip([mg_model.smoother.l0,
                   mg_model.smoother.l1,
                   mg_model.smoother.l2,
                   mg_model.smoother.l3],
                  weights_smoother):
    li.weights.data = wi
mg_model.coarse_model.weights.data = weights_coarse

# we do not want to change the coarse model
for wi in mg_model.coarse_model.parameters():
    wi.requires_grad = False

# function to calculate mse
def complex_mse(output, target):
    err = (output - target)
    return (err * err.conj()).real.sum()

# function for l2norm
def l2norm(v):
    return (v * v.conj()).real.sum()

# define optimizer and training parameters
training_epochs = 100
check_every = 10
adam_lr = 1e-2
optimizer = torch.optim.Adam(mg_model.parameters(), lr=adam_lr)

# logging
cost = np.zeros(training_epochs)
its = np.zeros(training_epochs // check_every + 1)

# test vector
test_v = torch.ones(8,8,8,16,4,3, dtype=torch.cdouble)
test_v /= l2norm(test_v)**0.5

# unpreconditioned iteration count
x, ret = qcd_ml.util.solver.GMRES_restarted(w, test_v, test_v, eps=1e-4, maxiter_inner=25, max_restart=400)
it_ref = ret["k"]
print(f"Unpreconditioned iteration count: {it_ref}")

# training
for t in range(training_epochs):
    v1 = torch.randn_like(test_v, dtype=torch.cdouble)
    v2 = torch.randn_like(test_v, dtype=torch.cdouble)

    Dinvv2, ret = qcd_ml.util.solver.GMRES_restarted(w, v2, v2, eps=1e-3, maxiter_inner=25, max_restart=40)
    wv1 = w(v1)

    ins = [wv1, v2]
    outs = [v1, Dinvv2]

    # normalize
    norms = [l2norm(e) for e in ins]
    ins = [e/n for e, n in zip(ins, norms)]
    outs = [e/n for e, n in zip(outs, norms)]

    curr_cost = complex_mse(mg_model.forward(ins[0]), outs[0])
    curr_cost += complex_mse(mg_model.forward(ins[1]), outs[1])

    cost[t] = curr_cost.item()
    optimizer.zero_grad()
    curr_cost.backward()
    optimizer.step()
    print(f"{t} - {cost[t]}")

    if t % check_every == 0:
        with torch.no_grad():
            x_p, ret_p = qcd_ml.util.solver.GMRES_restarted(w, test_v, test_v, prec=lambda v: mg_model.forward(v), eps=1e-4, maxiter_inner=25, max_restart=40)
            its[t // check_every] = ret_p["k"]
            print(f"\tICG: {it_ref / its[t//check_every]} ({it_ref}/{its[t//check_every]})")

# calculate final iteration count
with torch.no_grad():
    x_p, ret_p = qcd_ml.util.solver.GMRES_restarted(w, test_v, test_v, prec=lambda v: mg_model.forward(v), eps=1e-4, maxiter_inner=25, max_restart=400)
    its[-1] = ret_p["k"]
    print(f"\tICG: {it_ref / its[-1]} ({it_ref}/{its[-1]})")

torch.save(list(mg_model.parameters()), "multigrid_model.pt")

