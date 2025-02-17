import torch


class LGE_Exp(torch.nn.Module):
    r"""
    Provides an exponentiation layer for matrix-like fields acting on
    gauge links, i.e.,

    .. math::
        U_\mu(x) \rightarrow \exp\left(i\sum\limits_j \beta_{\mu,i} W_{i}(x)\right) U_\mu(x)
    """
    def __init__(self, n_features_in):
        super(LGE_Exp, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(4, n_features_in, dtype=torch.cdouble))
        
    def forward(self, U, W):
        hermitian = W.adjoint() - W
        trace = torch.einsum("mabcdii->mabcd", hermitian)

        identity = torch.clone(hermitian)
        identity[:,:,:,:,:] = torch.eye(3,3, dtype=torch.cdouble)
        trace_removal = torch.einsum("mabcdij,mabcd->mabcdij", identity, trace)
        traceless_hermitian = 1j/2 * (hermitian -  trace_removal / 3)

        transform_matrix = torch.matrix_exp(1j * torch.einsum("ui,i...->u...", self.weights, traceless_hermitian))
        return torch.einsum("uabcdij,uabcdjk->uabcdik", transform_matrix, U)
