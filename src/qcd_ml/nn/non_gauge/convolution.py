r"""
Non gauge equivariant convolutions.
"""

import torch


class C_Convolution(torch.nn.Module):
    r"""
    This class provides a :attr:`nd`-dimensional convolutional layer with circular padding.
    Originally described for 2D convolutions in the supplemental material of https://link.aps.org/doi/10.1103/PhysRevLett.128.032003.

    The convolution is defined as

    .. math::
        U_i(x) \rightarrow b_i + \sum_j \omega_{ij} \star U_j(x)

    where :math:`\star` is the :attr:`nd`-dimensional `cross-correlation` operator with periodic boundary conditions.

    Note:
        Padding and stride are set automatically to preserve lattice size.

    Attributes:
        nd (int): Number of spatial dimensions.
        n_input (int): Number of input channels.
        n_output (int): Number of output channels.
        kernel_size (tuple): Size of the convolution kernel.
        padding (list): Padding size.
        stride (list): Stride size.
        weights (torch.nn.Parameter): Learnable convolution weights.
        biases (torch.nn.Parameter or None): Learnable biases, or None if disabled.
    """

    def __init__(self, n_input: int, n_output: int, kernel_size: int, bias: bool = True, nd: int = 4) -> None:
        """Initialize the C_Convolution layer.

        Args:
            n_input: Number of input channels.
            n_output: Number of output channels.
            kernel_size: Size of the convolution kernel (can be a single int or a sequence).
            bias: Whether to include a bias term. Defaults to True.
            nd: Number of spatial dimensions. Defaults to 4.
        """
        super(C_Convolution, self).__init__()

        # number of lattice dimensions
        self.nd = nd

        # number of channels
        self.n_input = n_input
        self.n_output = n_output

        # kernel size
        self.kernel_size = torch.nn.modules.utils._ntuple(nd)(kernel_size)

        # padding size
        self.padding = self._padding()

        # stride size
        self.stride = [1] * nd

        # weights
        self.weights = torch.nn.Parameter(torch.randn(n_input
                                                      , n_output
                                                      , *self.kernel_size
                                                      , dtype=torch.cdouble))

        # biases
        if bias:
            self.biases = torch.nn.Parameter(torch.randn(n_output, dtype=torch.cdouble))
        else:
            self.biases = None

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the convolution.

        .. math::
            U_i(x) \rightarrow b_i + \sum_j \omega_{ij} \star U_j(x)

        Args:
            U: Input tensor of shape (n_input, *spatial_dims, ...).

        Returns:
            Output tensor of shape (n_output, *spatial_dims, ...).
        """
        nu = U.dim()
        assert nu > self.nd # (C, x_0, ..., x_{nd-1}, ...)

        # apply padding
        U = self._circular_pad(U)

        # unfold U
        for i, (ks, s) in enumerate(zip(self.kernel_size, self.stride)):
            dim = i + 1
            U = U.unfold(dim, ks, s)

        # compute convolution
        dim0 = [0] + list(range(2, self.nd + 2))
        dim1 = [0] + list(range(nu, self.nd + nu))
        U = torch.tensordot(self.weights, U, dims=(dim0, dim1))

        # add biases
        if self.biases is not None:
            U += self.biases.view(-1, *[1] * (nu - 1)).expand_as(U)

        return U

    def _circular_pad(self, U: torch.Tensor) -> torch.Tensor:
        """Apply circular padding.

        Args:
            U: Input tensor to pad.

        Returns:
            Padded tensor.
        """
        for i, ks in enumerate(self.kernel_size):
            if ks > 1:
                dim = i + 1

                left_i0 = U.shape[dim] - self.padding[2 * i]
                left_i1 = U.shape[dim]
                left_index = torch.arange(left_i0, left_i1, device=U.device)
                left_pad = U.index_select(dim, left_index)

                right_i0 = 0
                right_i1 = self.padding[2 * i + 1]
                right_index = torch.arange(right_i0, right_i1, device=U.device)
                right_pad = U.index_select(dim, right_index)

                U = torch.cat([left_pad, U, right_pad], dim=dim)

        return U

    def _padding(self) -> list:
        """Padding size to preserve lattice size.

        Returns:
            List of padding sizes [left_0, right_0, left_1, right_1, ...].
        """
        padding = []
        for ks in reversed(self.kernel_size):
            if ks % 2 == 0:
                # even kernel
                padding.append((ks - 1) // 2)
                padding.append(ks // 2)
            else:
                # odd kernel
                padding.append(ks // 2)
                padding.append(ks // 2)
        return padding

    def extra_repr(self) -> str:
        """Extra representation for the layer.

        Returns:
            String representation of the layer's key attributes.
        """
        return (
            f"{self.n_input}, {self.n_output}, kernel_size={self.kernel_size}"
            f", bias={self.biases is not None}"
        )
