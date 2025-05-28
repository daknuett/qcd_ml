import torch
from ..static import gamma


class dirac_dwf5_None:
    r"""
    A generic 5D domain wall operator with unspecified kernel.
    Note that this class is not supposed to be used directly but 
    use one of the derived classes.
    """
    def __init__(self, U, m, M5, b, c, Ls):
        self.m = m
        self.M5 = M5
        self.b = b
        self.c = c
        self.Ls = Ls

        self.gamma5 = gamma5.to(get_device_by_reference(U[0]))

        self.kernel = None

    def __call__(self, v):
        if v.shape[0] != self.Ls or v.shape[-1] != 3 or v.shape[-2] != 4:
            raise ValueError(f"expected v to be of shape [Ls={self.Ls}, x,y,z,t, 4,3] but got {v.shape}")

        # project to positive/negative chirality
        chiral_positive = 0.5 * v + 0.5*torch.einsum("ij,sxyztjg->sxyztig", self.gamma5, v)
        chiral_negative = 0.5 * v - 0.5*torch.einsum("ij,sxyztjg->sxyztig", self.gamma5, v)

        # 5d propagation
        result = torch.empty_like(v)
        for s in range(self.Ls):
            result[s] = self.b * self.kernel(v[s]) + v[s]
        for s in range(1, self.Ls):
            result[s] += self.c * self.kernel(chiral_positive[s-1]) - chiral_positive[s-1]
        for s in range(self.Ls - 1):
            result[s] += self.c * self.kernel(chiral_negative[s+1]) - chiral_negative[s+1]

        # mass term
        result[0] -= self.m * (self.c *self.kernel(chiral_positive[self.Ls - 1]) - chiral_positive[self.Ls - 1])
        result[self.Ls - 1] -= self.m * (self.c *self.kernel(chiral_negative[0]) - chiral_negative[0])

        return result
