import torch 

from ..base.paths import v_evaluate_path, v_ng_evaluate_path
from ..base.operations import v_spin_transform, v_ng_spin_transform

class v_LPTC(torch.nn.Module):
    """
    Local Parallel Transport Convolution for objects that 
    transform vector-like.

    Weights are stored as [feature_in, feature_out, path].

    paths is a list of paths. Every path is a list [(direction, nhops)].
    An empty list is the path that does not perform any hops.
    """
    def __init__(self, n_feature_in, n_feature_out, paths, U):
        super().__init__()
        self.weights = torch.nn.Parameter(
                torch.randn(n_feature_in, n_feature_out, len(paths), *tuple(U[0].shape[0:-2]), 4, 4, dtype=torch.cdouble)
                )

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out
        self.paths = paths
        self.U = U


    def forward(self, features_in):
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}")
        
        features_out = [torch.zeros_like(features_in[0]) for _ in range(self.n_feature_out)]

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi, wi in zip(self.paths, wfo):
                    features_out[io] = features_out[io] + v_spin_transform(wi, v_evaluate_path(self.U, pi, fi))

        return torch.stack(features_out)

class v_LPTC_NG(torch.nn.Module):
    """
    Local Parallel Transport Convolution for objects that 
    transform vector-like but with no gauge degrees of freedom.

    Weights are stored as [feature_in, feature_out, path].

    paths is a list of paths. Every path is a list [(direction, nhops)].
    An empty list is the path that does not perform any hops.
    """
    def __init__(self, n_feature_in, n_feature_out, paths, grid_dims, internal_dof):
        super().__init__()
        self.weights = torch.nn.Parameter(
                torch.randn(n_feature_in, n_feature_out, len(paths), *tuple(grid_dims), internal_dof, internal_dof, dtype=torch.cdouble)
                )

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out
        self.paths = paths
        self.internal_dof = internal_dof
        self.grid_dims = grid_dims


    def forward(self, features_in):
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}")
        
        features_out = [torch.zeros_like(features_in[0]) for _ in range(self.n_feature_out)]

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi, wi in zip(self.paths, wfo):
                    features_out[io] = features_out[io] + v_ng_spin_transform(wi, v_ng_evaluate_path(pi, fi))

        return torch.stack(features_out)
