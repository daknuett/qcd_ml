"""
------------

This module contains lattice gauge equvariant bilinear layers.
"""

import torch


class LGE_Bilinear(torch.nn.Module):
    r"""
    This class provides lattice gauge equivariant bilinear layers.

    .. math::
        W_{x,i}, W_{x,i}' \rightarrow \sum_{j,k} \alpha_{i,j,k} W_{x,j} W_{x,k}'

    See 10.1103/PhysRevLett.128.032003 for more details.
    """

    def __init__(self, n_input1, n_input2, n_output):
        super(LGE_Bilinear, self).__init__()
        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.n_output = n_output

        self.weights = torch.nn.Parameter(
                torch.randn(n_input1, n_input2, n_output, dtype=torch.cdouble))

    def forward(self, features_in1, features_in2):
        r"""
        This class provides lattice gauge equivariant bilinear layers.
        .. math::
            W_{x,i}, W_{x,i}' \rightarrow \sum_{j,k} \alpha_{i,j,k} W_{x,j} W_{x,k}'
        """

        return torch.einsum("jki,jabcdnr,kabcdrm->iabcdnm", self.weights, features_in1, features_in2)


class Apply_LGE_Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features_in1, features_in2, weights):
        ctx.save_for_backward(features_in1, features_in2, weights)
        return torch.einsum("jki,jabcdnr,kabcdrm->iabcdnm"
                            , weights, features_in1, features_in2)

    @staticmethod
    def backward(ctx, grad_output):
        features_in1, features_in2, weights = ctx.saved_tensors

        grad_weights = torch.einsum("iabcdnm,jabcdnr,kabcdrm->jki"
			                    	, torch.conj(grad_output), features_in1, features_in2)

        grad_f1 = torch.einsum("iabcdnm,jki,kabcdrm->jabcdnr"
                               , torch.conj(grad_output), weights, features_in2)

        grad_f2 = torch.einsum("iabcdnm,jki,jabcdnr->kabcdrm"
                               , torch.conj(grad_output), weights, features_in1)
        return torch.conj(grad_f1), torch.conj(grad_f2), torch.conj(grad_weights)


class LGE_BilinearLM(torch.nn.Module):
    r"""
    This is an implementation that provides the same functionality as
    LGE_Bilinear. The backward pass is slower by a factor of 2, but with significantly
    less memory consumption.

    This class provides lattice gauge equivariant bilinear layers.

    .. math::
        W_{x,i}, W_{x,i}' \rightarrow \sum_{j,k} \alpha_{i,j,k} W_{x,j} W_{x,k}'

    See 10.1103/PhysRevLett.128.032003 for more details.
    """

    def __init__(self, n_input1, n_input2, n_output):
        super(LGE_Bilinear2, self).__init__()
        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.n_output = n_output

        self.weights = torch.nn.Parameter(
                torch.randn(n_input1, n_input2, n_output, dtype=torch.cdouble))
        self.fn = Apply_LGE_Bilinear.apply

    def forward(self, features_in1, features_in2):
        r"""
        This class provides lattice gauge equivariant bilinear layers.
        .. math::
            W_{x,i}, W_{x,i}' \rightarrow \sum_{j,k} \alpha_{i,j,k} W_{x,j} W_{x,k}'
        """

        return self.fn(features_in1, features_in2, self.weights)
