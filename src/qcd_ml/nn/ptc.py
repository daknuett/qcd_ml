import torch 

from ..base.paths import PathBuffer
from ..base.operations import v_spin_const_transform

class v_PTC(torch.nn.Module):
    """
    Parallel Transport Convolution for objects that 
    transform vector-like.

    Weights are stored as [feature_in, feature_out, path].

    paths is a list of paths. Every path is a list [(direction, nhops)].
    An empty list is the path that does not perform any hops.
    """
    def __init__(self, n_feature_in, n_feature_out, paths, path_buffer: PathBuffer):
        super().__init__()
        self.weights = torch.nn.Parameter(
                torch.randn(n_feature_in, n_feature_out, len(paths), 4, 4, dtype=torch.cdouble)
                )

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out
        # FIXME: This is more memory intensive compared to the 
        # implementation using v_evaluate_path, because instead of one 
        # copy of U, all gauge transport matrices are stored.
        # On the other hand this may not be a big deal in most cases,
        # because, for 1h, the number of gauge fields is identical.
        self.path_transporters = [path_buffer.path(pi) for pi in paths]


    def forward(self, features_in):
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}")
        
        features_out = [torch.zeros_like(features_in[0]) for _ in range(self.n_feature_out)]

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi, wi in zip(self.path_transporters, wfo):
                    features_out[io] = features_out[io] + v_spin_const_transform(wi, pi.v_transport(fi))

        return torch.stack(features_out)

    
    def gauge_transform_using_transformed(self, new_path_buffer: PathBuffer):
        """
        Update the v_PTC layer: The old gauge field U is replaced by 
        U_transformed. The weights are kept.

        NOTE: This does not create a transformed copy of the layer!
              Instead the layer is updated.

        Mostly used for testing.
        """
        for i, pi in enumerate(self.path_transporters):
            self.path_transporters[i] = new_path_buffer.path(pi.path_indicator)
