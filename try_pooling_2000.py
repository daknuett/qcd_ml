import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime

import sys
sys.path.append("src/")

from qcd_ml.nn.lptc import v_LPTC
from qcd_ml.qcd.dirac import dirac_wilson_clover
#from qcd_ml.compat.gpt import lattice2ndarray, ndarray2lattice
from qcd_ml.util.solver import GMRES_torch
from qcd_ml.util.qcd.multigrid import ZPP_Multigrid
from qcd_ml.base.paths import v_evaluate_path, v_ng_evaluate_path, v_reverse_evaluate_path, PathBuffer
from qcd_ml.base.operations import v_spin_transform, v_ng_spin_transform, v_gauge_transform

from qcd_ml_accel.pool4d import v_pool4d, v_unpool4d



import itertools
def get_paths_lexicographic(block_size):
    paths = []
    for position in itertools.product(*(range(bs) for bs in block_size)):
        path = sum([[(mu, -1)] for mu, n in enumerate(position) for _ in range(n)], start=[])
        paths.append(path)
    return paths

def get_paths_reverse_lexicographic(block_size):
    return [list(reversed(pth)) for pth in get_paths_lexicographic(block_size)]

def get_paths_one_step_lexicographic(block_size):
    paths = []
    for position in itertools.product(*(range(bs) for bs in block_size)):
        path = []
        pos = np.array(position)
        while pos.any():
            for mu in range(pos.shape[0]):
                if pos[mu] > 0:
                    path.append((mu, -1))
                    pos[mu] -= 1
        paths.append(path)
    return paths
def get_paths_one_step_reverse_lexicographic(block_size):
    return [list(reversed(pth)) for pth in get_paths_one_step_lexicographic(block_size)]


def path_get_orig_point(path):
    point = [0] * 4
    for mu, nhops in path:
        point[mu] -= nhops
    return point

class v_ProjectLayer(torch.nn.Module):
    def __init__(self, gauges_and_paths, L_fine, L_coarse):
        super().__init__()
        self.path_buffers = [[PathBuffer(Ui, pij) for pij in pi] for Ui, pi in gauges_and_paths]
        self.weights = torch.nn.Parameter(
                torch.randn(len(gauges_and_paths), *tuple(U.shape[1:-2]), 4, 4, dtype=torch.cdouble)
                )
        self.L_fine = L_fine
        self.L_coarse = L_coarse
        self.block_size = torch.tensor([lf // lc for lf, lc in zip(L_fine, L_coarse)], dtype=torch.int64)

        self.base_points = np.array([[path_get_orig_point(pb.path) for pb in path_buffers] for path_buffers in self.path_buffers])
        # This keeps a gauge field for every point on the lattice
        # such that we can use this exact field to transform the
        # base_points before summing them up.
        self.gauge_fields = torch.zeros(len(gauges_and_paths), *tuple(U.shape[1:]), dtype=torch.cdouble)

        identity = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.cdouble)
        for i, gpi in enumerate(self.path_buffers):
            for j, pb in enumerate(gpi):
                base_point = self.base_points[i,j]
                if pb._is_identity:
                    self.gauge_fields[i
                            , base_point[0]::self.block_size[0]
                            , base_point[1]::self.block_size[1]
                            , base_point[2]::self.block_size[2]
                            , base_point[3]::self.block_size[3]] = identity
                else:
                    self.gauge_fields[i
                            , base_point[0]::self.block_size[0]
                            , base_point[1]::self.block_size[1]
                            , base_point[2]::self.block_size[2]
                            , base_point[3]::self.block_size[3]] = pb.accumulated_U[base_point[0]::self.block_size[0]
                                                                        , base_point[1]::self.block_size[1]
                                                                        , base_point[2]::self.block_size[2]
                                                                        , base_point[3]::self.block_size[3]]

            
    def v_project(self, features_in):
        if features_in.shape[0] != 1:
            raise NotImplementedError()
        before_pool = torch.zeros(features_in.shape[0], self.gauge_fields.shape[0], *features_in.shape[1:]
                                 , dtype=torch.cdouble)
        for i, fea_i in enumerate(features_in):
            for j, (gfj, wj) in enumerate(zip(self.gauge_fields, self.weights)):
                before_pool[i,j] = v_spin_transform(wj, v_gauge_transform(gfj, fea_i))

        return torch.stack([v_pool4d(torch.sum(before_pool, axis=1)[0], self.block_size)])

    def v_prolong(self, features_in):
        if features_in.shape[0] != 1:
            raise NotImplementedError()
        before_weights = torch.zeros(features_in.shape[0], self.gauge_fields.shape[0]
                                     , *(self.L_fine), *(features_in.shape[5:])
                                     , dtype=torch.cdouble)
        

        for i in range(self.gauge_fields.shape[0]):
            before_weights[0,i] = v_unpool4d(features_in[0], self.block_size)

        before_accumulate = torch.zeros_like(before_weights)

        for i, fea_i in enumerate(before_weights):
            for j, (gfj, wj) in enumerate(zip(self.gauge_fields, self.weights)):
                before_accumulate[i,j] = v_spin_transform(wj.adjoint(), v_gauge_transform(gfj.adjoint(), fea_i[j]))

        return torch.sum(before_accumulate, axis=1)

def log(*args, **kwargs):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ", end="")
    print(*args, **kwargs, flush=True)

vec = torch.complex(
        torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double)
        , torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double))

log("Load data")
U = torch.tensor(np.load(os.path.join("test", "assets","2000.config.npy")))

U_smeared = torch.load("U_smeared_2000.pt")
U_smeared = [torch.tensor(np.array(Us)) for Us in U_smeared]

w = dirac_wilson_clover(U, -0.56, 1.0)

mg = ZPP_Multigrid.load("mg_setup_2000.pt")

log("Setup model")


tfp = v_ProjectLayer([
                (U_smeared[0], get_paths_lexicographic(mg.block_size))
                , (U_smeared[1], get_paths_reverse_lexicographic(mg.block_size))
                , (U_smeared[2], get_paths_one_step_lexicographic(mg.block_size))
                , (U_smeared[3], get_paths_one_step_reverse_lexicographic(mg.block_size))
                , (U_smeared[4], get_paths_lexicographic(mg.block_size))
                , (U_smeared[5], get_paths_reverse_lexicographic(mg.block_size))
                , (U_smeared[6], get_paths_one_step_lexicographic(mg.block_size))
                , (U_smeared[7], get_paths_one_step_reverse_lexicographic(mg.block_size))
                , (U_smeared[8], get_paths_lexicographic(mg.block_size))
            ]
            , mg.L_fine, mg.L_coarse)

log("Rescale weights")

with torch.no_grad():
    for wi in tfp.parameters():
        wi *= 1e-2


optimizer = torch.optim.Adam(tfp.parameters(), lr=1e-3)

n_training = 10000
log_every = 100

def complex_mse_loss(output, target):
    err = (output - target)
    return (err * err.conj()).real.sum()

def l2norm(v):
    return (v * v.conj()).real.sum()


t_vec_gen = 0
t_cost_compute = 0
t_train = 0

vec = torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)
vec_coarse = tfp.v_project(torch.stack([vec]))

loss = np.zeros(n_training)
for t in range(1, n_training+1):
    t_start = time.perf_counter_ns()
    vec = torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)
    
    norm = l2norm(vec)
    vec = vec / norm

    prvec = mg.v_prolong(mg.v_project(vec))

    vec2 = torch.rand_like(vec)
    vec2 = mg.v_prolong(mg.v_project(vec2))
    vec2 = vec2 / l2norm(vec2)
    
    vec3 = torch.rand_like(vec_coarse)
    vec3 = vec3 / l2norm(vec3)
    t_stop = time.perf_counter_ns()
    t_vec_gen = (t_stop - t_start) / 1000**2
    #print("TRAINING VECTOR GEN OK")

    t_start = time.perf_counter_ns()
    score1 = complex_mse_loss(tfp.v_prolong(tfp.v_project(torch.stack([vec]))), torch.stack([prvec]))
    score2 = complex_mse_loss(tfp.v_prolong(tfp.v_project(torch.stack([vec2]))), torch.stack([vec2]))
    score3 = complex_mse_loss(tfp.v_project(tfp.v_prolong(vec3)), vec3)
    score = 2 * score1 + 2 * score2 + score3
    t_stop = time.perf_counter_ns()
    #print("SCORE F GEN OK")

    t_cost_compute = (t_stop - t_start) / 1000**2
    loss[t-1] = score.item()

    t_start = time.perf_counter_ns()
    optimizer.zero_grad()
    score.backward()
    optimizer.step()
    t_stop = time.perf_counter_ns()
    t_train = (t_stop - t_start) / 1000**2
    
    if t % log_every == 0:
        log(f"[{t:5d}|{t / n_training * 100:6.2f}%] {score.item():.3e} ({t_vec_gen:.3f} + {t_cost_compute:.3f} + {t_train:.3f})", end="\r")

log("Training done")

torch.save(tfp.state_dict(), "tfp_2000_01.pt")
np.save("loss_2000_01.npy", loss)
