import torch 

from ..hop import v_hop

class v_PTC(torch.nn.Module):
    """
    Parallel Transport Convolution for objects that 
    transform vector-like.

    Weights are stored as [feature_in, feature_out, path].

    paths is a list of paths. Every path is a list [(direction, nhops)].
    An empty list is the path that does not perform any hops.
    """
    def __init__(self, n_feature_in, n_feature_out, paths):
        self.weights = torch.nn.Parameter(
                torch.complex(torch.tensor(n_feature_in, n_feature_out, len(paths), dtype=torch.double)
                              , torch.tensor(n_feature_in, n_feature_out, len(paths), dtype=torch.double))
                )

        self.n_feature_in = n_feature_in
        self.n_feature_paths = n_feature_paths


