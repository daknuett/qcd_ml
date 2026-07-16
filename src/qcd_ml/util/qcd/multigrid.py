#!/usr/bin/env python3

"""
Provides Multigrid with zero point projection.
"""

from typing import Callable, List, Tuple
import torch
import itertools
from qcd_ml.util.linear_algebra import innerproduct, norm

def orthonormalize(vecs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Orthonormalize a list of vectors using the Gram-Schmidt process.

    Args:
        vecs: List of input vectors (tensors) to orthonormalize.

    Returns:
        List[torch.Tensor]: List of orthonormalized vectors. The output has the same
            length as the input, and each vector is normalized to unit length and
            orthogonal to all previous vectors in the list.

    Note:
        This implementation uses the modified Gram-Schmidt process.
    """
    basis = []
    for vec in vecs:
        for b in basis:
            vec = vec - innerproduct(b, vec) * b
        vec = vec / norm(vec)
        basis.append(vec)
    return basis

class ZPP_Multigrid:
    """Multigrid with zeropoint projection.

    Use ``.v_project`` and ``.v_prolong`` to project and prolong vectors.
    Use ``.get_coarse_operator`` to construct a coarse operator.
    For Wilson(-clover) Dirac operators, use ``qcd_ml.qcd.dirac.coarsened.coarse_9point_op_NG.from_dirac_operator_and_multigrid`` 
    for a faster implementation.

    use ``ZPP_Multigrid.gen_from_fine_vectors([random vectors], [i, j, k, l], lambda b, xo: <solve Dx = b for x>)``
    to construct a ``ZPP_Multigrid``.

    Attributes:
        block_size: Tuple of 4 integers specifying the block size in each dimension.
        block_basis: Tensor of shape (Lx, Ly, Lz, Lt, ..., n_basis) containing all basis vectors
            on the fine lattice.
        n_basis: Number of basis vectors.
        L_coarse: Tuple of 4 integers specifying the coarse lattice dimensions.
        L_fine: Tuple of 4 integers specifying the fine lattice dimensions.
    """

    def __init__(self,
                 block_size: Tuple[int, ...],
                 block_basis: torch.Tensor,
                 n_basis: int,
                 L_coarse: Tuple[int, ...],
                 L_fine: Tuple[int, ...]) -> None:
        """Initialize the ZPP_Multigrid.

        Args:
            block_size: Size of blocks in each of the 4 spacetime dimensions.
            block_basis: Tensor of shape (Lx, Ly, Lz, Lt, ..., n_basis) containing all basis vectors
                on the fine lattice.
            n_basis: Number of basis vectors per block.
            L_coarse: Dimensions of the coarse lattice (length 4 tuple).
            L_fine: Dimensions of the fine lattice (length 4 tuple).
        """
        self.block_size = block_size
        self.block_basis = block_basis
        self.n_basis = n_basis
        self.L_coarse = L_coarse
        self.L_fine = L_fine

    def cuda(self) -> 'ZPP_Multigrid':
        """Move all basis vectors to CUDA device.

        Returns:
            ZPP_Multigrid: A new ZPP_Multigrid instance with all basis vectors moved
                to CUDA device. All other attributes remain the same.

        Note:
            This creates a new instance rather than modifying in place.
        """
        block_basis = self.block_basis.cuda()

        return self.__class__(self.block_size, block_basis, self.n_basis, self.L_coarse, self.L_fine)


    @classmethod
    def gen_from_fine_vectors(cls,
                              fine_vectors: List[torch.Tensor],
                              block_size: Tuple[int, ...],
                              solver: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, dict]],
                              verbose: bool = False) -> 'ZPP_Multigrid':
        """Used to generate a multigrid setup using fine vectors, a block size and a solver.

        solver should be
            ``(x, info) = solver(b, x0)``
        which solves
            :math:`D x = b`

        we will choose
            ``b = torch.zeros_like(x0)``

        Args:
            fine_vectors: List of fine lattice vectors to use as starting points
                for computing zero-point vectors.
            block_size: Size of blocks in each of the 4 spacetime dimensions.
            solver: Solver function that takes (b, x0) and returns (x, info) where
                x is the solution to Dx = b, and info is a dictionary with
                convergence information (must contain 'converged', 'k', 'res' keys
                when verbose=True).
            verbose: If True, print convergence information for each solve.

        Returns:
            ZPP_Multigrid: Initialized multigrid instance with zero-point projected
                basis vectors.
        """
        # length of basis
        n_basis = len(fine_vectors)
        # normalize
        bv = [bi / norm(bi) for bi in fine_vectors]
        # compute zero point vectors
        zero = torch.zeros_like(bv[0])
        ui = []
        for i, b in enumerate(bv):
            uk, ret = solver(zero, b)
            if verbose:
                print(f"[{i:2d}]: {ret['converged']} ({ret['k']:5d}) <{ret['res']:.4e}>")
            ui.append(uk)

        # size of fine lattice
        L_fine = tuple(ui[0].shape[:4])
        # size of coarse lattice
        L_coarse = tuple(lf // bs for lf, bs in zip(L_fine, block_size))
        
        # Get the spin-color shape from the first basis vector
        sample_shape = ui[0].shape[4:]
        
        # Create the full block_basis tensor
        block_basis = torch.zeros(
            *L_fine, *sample_shape, n_basis,
            dtype=torch.cdouble, device=ui[0].device
        )
        
        lx, ly, lz, lt = block_size
        
        for bx, by, bz, bt in itertools.product(*(range(li) for li in L_coarse)):
            # Collect basis vectors for this block
            block_vecs = []
            for u in ui:
                u_block = u[bx * lx: (bx + 1)*lx
                            , by * ly: (by + 1)*ly
                            , bz * lz: (bz + 1)*lz
                            , bt * lt: (bt + 1)*lt]
                block_vecs.append(u_block)
            
            # Orthogonalize over block
            block_vecs = orthonormalize(block_vecs)
            
            # Place each basis vector at its position on the fine lattice
            for k, uk in enumerate(block_vecs):
                slices = [
                    slice(bx * lx, (bx + 1) * lx),
                    slice(by * ly, (by + 1) * ly),
                    slice(bz * lz, (bz + 1) * lz),
                    slice(bt * lt, (bt + 1) * lt),
                ] + [slice(None) for _ in range(len(sample_shape))] + [k]
                block_basis[tuple(slices)] = uk

        return cls(block_size, block_basis, n_basis, L_coarse, L_fine)
    
    def v_project(self, v: torch.Tensor) -> torch.Tensor:
        """project fine vector ``v`` to coarse grid.

        Args:
            v: Fine grid vector with shape (Lx, Ly, Lz, Lt, ...).

        Returns:
            torch.Tensor: Coarse grid projection with shape (L_coarse[0], L_coarse[1],
                L_coarse[2], L_coarse[3], n_basis) and dtype torch.cdouble.
        """
        # Project onto block basis modes
        # block_basis has shape (*L_fine, ..., n_basis)
        # v has shape (*L_fine, ...)
        # Result of einsum will have shape (*L_fine, n_basis)
        projection = torch.einsum(
            "...sck,...sc->...k", self.block_basis.conj(), v
        )
        
        # Block-sum: reshape to group into blocks and sum over block dimensions
        new_shape = [
            self.L_coarse[0],
            self.block_size[0],
            self.L_coarse[1],
            self.block_size[1],
            self.L_coarse[2],
            self.block_size[2],
            self.L_coarse[3],
            self.block_size[3],
            projection.shape[-1],
        ]
        x = projection.reshape(new_shape)
        block_dims = [1, 3, 5, 7]
        coarse_vec = x.sum(dim=block_dims)
        
        return coarse_vec
    
    def v_prolong(self, v: torch.Tensor) -> torch.Tensor:
        """prolong coarse vector ``v`` to fine grid.

        Args:
            v: Coarse grid vector with shape (L_coarse[0], L_coarse[1], L_coarse[2],
                L_coarse[3], n_basis).

        Returns:
            torch.Tensor: Fine grid vector with shape (L_fine[0], L_fine[1], L_fine[2],
                L_fine[3], ...) and dtype torch.cdouble.
        """
        x = v
        x = x.repeat_interleave(self.block_size[0], dim=0)
        x = x.repeat_interleave(self.block_size[1], dim=1)
        x = x.repeat_interleave(self.block_size[2], dim=2)
        x = x.repeat_interleave(self.block_size[3], dim=3)
        
        # Reconstruct fine vector from coarse coefficients
        # x has shape (*L_fine, n_basis)
        # block_basis has shape (*L_fine, ..., n_basis)
        # Result will have shape (*L_fine, ...)
        fine_vec = torch.einsum("...k,...sck->...sc", x, self.block_basis)
        
        return fine_vec
    
    def get_coarse_operator(self,
                            fine_operator: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        """Given a fine operator ``fine_operator(psi)``, construct a coarse operator.

        In case of a 9-point operator, such as Wilson and Wilson-Clover Dirac operator,
        a significantly faster implementation can be achieved by using ``qcd_ml.qcd.dirac.coarsened.coarse_9point_op_NG``
        (either ``from_operator_and_multigrid`` or ``from_dirac_operator_and_multigrid``).

        Args:
            fine_operator: Operator function that takes a fine grid vector and returns
                a fine grid vector.

        Returns:
            Callable: Coarse operator function that takes a coarse grid vector and
                returns a coarse grid vector. The coarse operator is defined as:
                coarse_op(v_coarse) = v_project(fine_operator(v_prolong(v_coarse)))
        """
        def operator(source_coarse: torch.Tensor) -> torch.Tensor:
            source_fine = self.v_prolong(source_coarse)
            dst_fine = fine_operator(source_fine)
            return self.v_project(dst_fine)
        return operator

    def save(self, filename: str) -> None:
        """This is a stupid implementation. Saves all arguments as a list.

        Args:
            filename: Path to save the multigrid instance data.
        """
        torch.save([self.block_size, self.block_basis, self.n_basis, self.L_coarse, self.L_fine], filename)

    @classmethod
    def load(cls, filename: str) -> 'ZPP_Multigrid':
        """This is a stupid implementation. Loads all arguments as a list.

        Args:
            filename: Path to load the multigrid instance data from.

        Returns:
            ZPP_Multigrid: Loaded multigrid instance.
        """
        args = torch.load(filename)
        return cls(*tuple(args))

