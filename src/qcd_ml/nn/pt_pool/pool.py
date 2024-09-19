import torch 
import numpy as np
from ...base.paths import PathBuffer, path_get_orig_point
from ...base.operations import v_spin_transform, v_gauge_transform

import warnings

try: 
    from qcd_ml_accel.pool4d import v_pool4d, v_unpool4d
except ImportError:
    from .pool4d import v_pool4d, v_unpool4d
    warnings.warn("Using slow python implementation of v_pool4d and v_unpool4d (install qcd_ml_accel for faster implementation)")



class v_ProjectLayer(torch.nn.Module):
    def __init__(self, gauges_and_paths, L_fine, L_coarse, _gpt_compat=False):
        super().__init__()
        self.path_buffers = [[PathBuffer(Ui, pij) for pij in pi] for Ui, pi in gauges_and_paths]

        self.weights = torch.nn.Parameter(
                torch.randn(len(gauges_and_paths), *L_fine, 4, 4, dtype=torch.cdouble)
                )
        self.L_fine = L_fine
        self.L_coarse = L_coarse
        self.block_size = torch.tensor([lf // lc for lf, lc in zip(L_fine, L_coarse)], dtype=torch.int64)

        self.base_points = np.array([[path_get_orig_point(pb.path) for pb in path_buffers] for path_buffers in self.path_buffers])
        # This keeps a gauge field for every point on the lattice
        # such that we can use this exact field to transform the
        # base_points before summing them up.
        self.gauge_fields = torch.zeros(len(gauges_and_paths), *tuple(gauges_and_paths[0][0].shape[1:]), dtype=torch.cdouble)
        self.__gpt_compat = _gpt_compat

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

            # XXX: extract this into another module
            def l2norm(v):
                return (v * v.conj()).real.sum()

            if _gpt_compat:
                # I have no idea why lehner/gpt does this, but we need to do it to match.
                self.gauge_fields[i] /= l2norm(self.gauge_fields[i])**0.5

            
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

