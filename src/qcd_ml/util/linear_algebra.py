"""
qcd_ml.util.linear_algebra
==========================

Linear algebra utility functions for QCD computations.
"""

import torch


def innerproduct(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the inner product of two tensors.

    The inner product is computed as the sum of the element-wise product of x conjugated
    and y.

    Args:
        x: First tensor.
        y: Second tensor.

    Returns:
        torch.Tensor: The complex inner product value.
    """
    return (x.conj() * y).sum()


def norm(x: torch.Tensor) -> torch.Tensor:
    """Compute the L2 norm of a tensor.

    Args:
        x: Tensor to compute norm of.

    Returns:
        torch.Tensor: The real-valued norm (sqrt of inner product with self).
    """
    return torch.sqrt(innerproduct(x, x).real)
