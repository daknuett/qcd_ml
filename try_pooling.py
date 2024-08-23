import torch
import numpy as np
import os

import time

import sys
sys.path.append("src/")

from qcd_ml.nn.lptc import v_LPTC
from qcd_ml.qcd.dirac import dirac_wilson_clover
#from qcd_ml.compat.gpt import lattice2ndarray, ndarray2lattice
from qcd_ml.util.solver import GMRES_torch
from qcd_ml.util.qcd.multigrid import ZPP_Multigrid
from qcd_ml.base.paths import v_evaluate_path, v_ng_evaluate_path, v_reverse_evaluate_path, PathBuffer
from qcd_ml.base.operations import v_spin_transform, v_ng_spin_transform

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

vec = torch.complex(
        torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double)
        , torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double)).cuda()

U = torch.tensor(np.load(os.path.join("test", "assets","1500.config.npy")))

U_smeared = torch.load("U_smeared.pt")
U_smeared = [torch.tensor(np.array(Us)) for Us in U_smeared]

import itertools
def get_paths_lexicographic(block_size):
    paths = []
    for position in itertools.product(*(range(bs) for bs in block_size)):
        path = sum([[(mu, 1)] for mu, n in enumerate(position) for _ in range(n)], start=[])
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
                    path.append((mu, 1))
                    pos[mu] -= 1
        paths.append(path)
    return paths
def get_paths_one_step_reverse_lexicographic(block_size):
    return [list(reversed(pth)) for pth in get_paths_one_step_lexicographic(block_size)]

class v_ParallelTransportPool(torch.nn.Module):
    def __init__(self, paths, U):
        super().__init__()
        self.weights = torch.nn.Parameter(
                torch.complex(torch.randn(1, 1, *tuple(U.shape[1:-2]), 4, 4, dtype=torch.double)
                              , torch.randn(1, 1, *tuple(U.shape[1:-2]), 4, 4, dtype=torch.double))
                )

        self.path_buffers = [PathBuffer(U, pi) for pi in paths]

    
    def forward(self, features_in):
        if features_in.shape[0] != 1:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {1}")
        
        features_out = torch.zeros_like(features_in)

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi in self.path_buffers:
                    features_out[io] = features_out[io] + pi.v_transport(v_spin_transform(wfo, fi))

        return features_out

    
    def reverse(self, features_in):
        if features_in.shape[0] != 1:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {1}")
        
        features_out = torch.zeros_like(features_in)

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi in self.path_buffers:
                    features_out[io] = features_out[io] + v_spin_transform(wfo.adjoint(), pi.v_reverse_transport(fi))

        return features_out 


class v_PoolingLayer(torch.nn.Module):
    def __init__(self, gauges_and_paths):
        super().__init__()
        self.layers = torch.nn.ModuleList([v_ParallelTransportPool(Pi, Ui) for Ui, Pi in gauges_and_paths])


    def pool(self, features_in):
        result = torch.zeros_like(features_in)
        for li in self.layers:
            result = result + li.forward(features_in)
        return result

    def de_pool(self, features_in):
        result = torch.zeros_like(features_in)
        for li in self.layers:
            result = result + li.reverse(features_in)
        return result
class v_SubSampling(torch.nn.Module):
    def __init__(self, L_fine, L_coarse):
        super().__init__()
        self.L_fine = L_fine
        self.L_coarse = L_coarse
        self.block_size = [lf // lc for lf, lc in zip(L_fine, L_coarse)]

    def v_project(self, features):
        res = torch.complex(
            torch.zeros(features.shape[0], *self.L_coarse, *features.shape[-2:], dtype=torch.double)
            , torch.zeros(features.shape[0], *self.L_coarse, *features.shape[-2:], dtype=torch.double)
        )
        res = res + features[:,::self.block_size[0], ::self.block_size[1], ::self.block_size[2], ::self.block_size[3]]
        return res

    def v_prolong(self, features):
        res = torch.complex(
            torch.zeros(features.shape[0], *self.L_fine, *features.shape[-2:], dtype=torch.double)
            , torch.zeros(features.shape[0], *self.L_fine, *features.shape[-2:], dtype=torch.double)
        )
        res[:,::self.block_size[0], ::self.block_size[1], ::self.block_size[2], ::self.block_size[3]] = res[:,::self.block_size[0], ::self.block_size[1], ::self.block_size[2], ::self.block_size[3]] + features
        return res 

class v_ProjectLayer(torch.nn.Module):
    def __init__(self, gauges_and_paths, L_fine, L_coarse):
        super().__init__()
        self.pooling = v_PoolingLayer(gauges_and_paths)
        self.subsampling = v_SubSampling(L_fine, L_coarse)

    def parameters(self):
        yield from self.pooling.parameters()

    def project(self, features_in):
        return self.subsampling.v_project(self.pooling.pool(features_in))

    def prolong(self, features_in):
        return self.pooling.de_pool(self.subsampling.v_prolong(features_in))



mg = ZPP_Multigrid.load("mg_setup.pt").cuda()


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

tfp.cuda()

num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')


vec = torch.complex(
        torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double)
        , torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.double))

x = torch.stack([vec]).cuda()

import torch.utils.benchmark as benchmark


t0 = benchmark.Timer(
    stmt='tfp.project(x)',
    #setup='from __main__ import tfp',
    globals={'x': x, 'tfp': tfp},
    num_threads=num_threads,
    label='Multithreaded v_ProjectLayer',
    sub_label='Implemented using path buffer')

m0 = t0.blocked_autorange()

print(m0)


optimizer = torch.optim.Adam(tfp.parameters(), lr=1e-2)

n_training = 100
check_every = 1
plot_every = 100


def complex_mse_loss(output, target):
    err = (output - target)
    return (err * err.conj()).real.sum()

def l2norm(v):
    return (v * v.conj()).real.sum()

t_vec_gen = 0
t_cost_compute = 0
t_train = 0

loss = np.zeros(n_training)
for t in range(1, n_training+1):
    t_start = time.perf_counter_ns()
    vec = torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble).cuda()
    
    norm = l2norm(vec)
    vec = vec / norm

    prvec = mg.v_prolong(mg.v_project(vec))

    vec2 = torch.rand_like(vec)
    vec2 = mg.v_prolong(mg.v_project(vec2))
    vec2 = vec2 / l2norm(vec2)

    vec3 = torch.rand_like(tfp.project(torch.stack([vec])))
    vec3 = vec3 / l2norm(vec3)
    t_stop = time.perf_counter_ns()
    t_vec_gen = (t_stop - t_start) / 1000**2
    #print("TRAINING VECTOR GEN OK")

    t_start = time.perf_counter_ns()
    score1 = complex_mse_loss(tfp.prolong(tfp.project(torch.stack([vec]))), torch.stack([prvec]))
    score2 = complex_mse_loss(tfp.prolong(tfp.project(torch.stack([vec2]))), torch.stack([vec2]))
    score3 = complex_mse_loss(tfp.project(tfp.prolong(torch.stack([vec3]))), torch.stack([vec3]))
    score = score1 + score2 + score3
    t_stop = time.perf_counter_ns()
    #print("SCORE F GEN OK")

    t_cost_compute = (t_stop - t_start) / 1000**2
    print(f"T [{t:5d}|{t / n_training * 100:6.2f}%] {score.item():.3e} ({t_vec_gen:.3f} + {t_cost_compute:.3f} + {t_train:.3f})", end="\r")
    loss[t-1] = score.item()

    t_start = time.perf_counter_ns()
    optimizer.zero_grad()
    score.backward()
    optimizer.step()
    t_stop = time.perf_counter_ns()
    t_train = (t_stop - t_start) / 1000**2
    
    print(f"  [{t:5d}|{t / n_training * 100:6.2f}%] {score.item():.3e} ({t_vec_gen:.3f} + {t_cost_compute:.3f} + {t_train:.3f})", end="\r")
