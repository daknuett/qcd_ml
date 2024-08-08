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

# 1 hop paths
paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]

# create the high-mode PTC layer
fine_ptc = qcd_ml.nn.ptc.v_PTC(1,1,paths,U)

# function to apply high-mode PTC layer
mh = lambda x: fine_ptc.forward(torch.stack([x]))[0]

# load weights
fine_ptc.weights = torch.nn.Parameter(torch.load("1h1l_ptc.pt", weights_only=True)[0], requires_grad=False)

# Wilson-clover Dirac operator (both gpt as well as native versions)
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

# defect correction
def ukn(b, mh, w, n):
    uk = torch.zeros_like(b)

    for k in range(n):
        result = b
        result -= w(uk)
        result = mh(result)
        uk += result
    return uk

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

# create the smoother model
model = Smoother(U, paths)

# initialize weights
for li in [model.l0, model.l1, model.l2, model.l3]:
    li.weights.data = 0.001 * torch.randn_like(li.weights.data, dtype=torch.cdouble)
    li.weights.data[:,:,0] += torch.eye(4)

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
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

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

# helping zero
null = torch.zeros_like(test_v, dtype=torch.cdouble)

# training
for t in range(training_epochs):
    source = torch.randn(8,8,8,16,4,3, dtype=torch.cdouble)
    source /= l2norm(source)

    in_vec = torch.stack([source, null])
    out_vec = torch.stack([ukn(source, mh, w, 2)])

    curr_cost = complex_mse(model.forward(in_vec), out_vec)
    cost[t] = curr_cost.item()
    optimizer.zero_grad()
    curr_cost.backward()
    optimizer.step()
    print(f"{t} - {cost[t]}")

    if t % check_every == 0:
        with torch.no_grad():
            x_p, ret_p = qcd_ml.util.solver.GMRES_restarted(w, test_v, test_v, prec=lambda v: model.forward(torch.stack([v, null]))[0], eps=1e-4, maxiter_inner=25, max_restart=40)
            its[t // check_every] = ret_p["k"]
            print(f"\tICG: {it_ref / its[t//check_every]} ({it_ref}/{its[t//check_every]})")

# calculate final iteration count
with torch.no_grad():
    x_p, ret_p = qcd_ml.util.solver.GMRES_restarted(w, test_v, test_v, prec=lambda v: model.forward(torch.stack([v, null]))[0], eps=1e-4, maxiter_inner=25, max_restart=400)
    its[-1] = ret_p["k"]
    print(f"\tICG: {it_ref / its[-1]} ({it_ref}/{its[-1]})")

torch.save(list(model.parameters()), "smoother.pt")
