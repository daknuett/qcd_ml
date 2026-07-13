#!/usr/bin/env python3

"""
Provides Multigrid with zero point projection.
"""

from typing import Callable, List, Tuple
import torch
import itertools
import numpy as np
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

    use ``ZPP_Multigrid.gen_from_fine_vectors([random vectors], [i, j, k, l], lambda b, xo: <solve Dx = b for x>)``
    to construct a ``ZPP_Multigrid``.

    Attributes:
        block_size: Tuple of 4 integers specifying the block size in each dimension.
        ui_blocked: Nested numpy array structure containing blocked basis vectors.
        n_basis: Number of basis vectors.
        L_coarse: Tuple of 4 integers specifying the coarse lattice dimensions.
        L_fine: Tuple of 4 integers specifying the fine lattice dimensions.
    """

    def __init__(self,
                 block_size: Tuple[int, ...],
                 ui_blocked: np.ndarray,
                 n_basis: int,
                 L_coarse: Tuple[int, ...],
                 L_fine: Tuple[int, ...]) -> None:
        """Initialize the ZPP_Multigrid.

        Args:
            block_size: Size of blocks in each of the 4 spacetime dimensions.
            ui_blocked: Nested structure containing basis vectors organized by
                coarse grid block. Shape is L_coarse[0] x L_coarse[1] x L_coarse[2] x
                L_coarse[3], with each element being a list of n_basis torch.Tensor objects.
            n_basis: Number of basis vectors per block.
            L_coarse: Dimensions of the coarse lattice (length 4 tuple).
            L_fine: Dimensions of the fine lattice (length 4 tuple).
        """
        self.block_size = block_size
        self.ui_blocked = ui_blocked
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
        ui_blocked = list(np.empty(self.L_coarse, dtype=object))
        for bx, by, bz, bt in itertools.product(*(range(li) for li in self.L_coarse)):
            ui_blocked[bx][by][bz][bt] = [uib.cuda() for uib in self.ui_blocked[bx][by][bz][bt]]

        return self.__class__(self.block_size, ui_blocked, self.n_basis, self.L_coarse, self.L_fine)

    @classmethod
    def from_basis_vectors(cls,
                           basis_vectors: List[torch.Tensor],
                           block_size: Tuple[int, ...]) -> 'ZPP_Multigrid':
        """Used to generate a multigrid setup using basis vectors and a block size.

        The basis vectors can be obtained using ``.get_basis_vectors()`` method.

        Args:
            basis_vectors: List of basis vectors on the fine lattice. Each vector
                should have shape (Lx, Ly, Lz, Lt, ...).
            block_size: Size of blocks in each of the 4 spacetime dimensions.

        Returns:
            ZPP_Multigrid: Initialized multigrid instance.
        """
        n_basis = len(basis_vectors)
        L_fine = list(basis_vectors[0].shape[:4])
        L_coarse = [lf // bs for lf, bs in zip(L_fine, block_size)]

        # Perform blocking
        lx, ly, lz, lt = block_size
        ui_blocked = list(np.empty(L_coarse, dtype=object))
        
        for bx, by, bz, bt in itertools.product(*(range(li) for li in L_coarse)):
            for uk in basis_vectors:
                u_block = uk[bx * lx: (bx + 1)*lx
                            , by * ly: (by + 1)*ly
                            , bz * lz: (bz + 1)*lz
                            , bt * lt: (bt + 1)*lt]
                if ui_blocked[bx][by][bz][bt] is None:
                    ui_blocked[bx][by][bz][bt] = []
                ui_blocked[bx][by][bz][bt].append(u_block)

            # Orthogonalize over block
            ui_blocked[bx][by][bz][bt] = orthonormalize(ui_blocked[bx][by][bz][bt])

        return cls(block_size, ui_blocked, n_basis, L_coarse, L_fine)

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
        L_fine = list(uk.shape[:4])
        # size of coarse lattice
        L_coarse = [lf // bs for lf, bs in zip(L_fine, block_size)]

        # Perform blocking
        lx, ly, lz, lt = block_size
        ui_blocked = list(np.empty(L_coarse, dtype=object))
        
        for bx, by, bz, bt in itertools.product(*(range(li) for li in L_coarse)):
            for uk in ui:
                u_block = uk[bx * lx: (bx + 1)*lx
                            , by * ly: (by + 1)*ly
                            , bz * lz: (bz + 1)*lz
                            , bt * lt: (bt + 1)*lt]
                if ui_blocked[bx][by][bz][bt] is None:
                    ui_blocked[bx][by][bz][bt] = []
                ui_blocked[bx][by][bz][bt].append(u_block)

            # Orthogonalize over block
            ui_blocked[bx][by][bz][bt] = orthonormalize(ui_blocked[bx][by][bz][bt])

        return cls(block_size, ui_blocked, n_basis, L_coarse, L_fine)
    
    def v_project(self, v: torch.Tensor) -> torch.Tensor:
        """project fine vector ``v`` to coarse grid.

        Args:
            v: Fine grid vector with shape (Lx, Ly, Lz, Lt, ...).

        Returns:
            torch.Tensor: Coarse grid projection with shape (L_coarse[0], L_coarse[1],
                L_coarse[2], L_coarse[3], n_basis) and dtype torch.cdouble.
        """
        projected = torch.zeros(self.L_coarse + [self.n_basis], dtype=torch.cdouble)
        lx, ly, lz, lt = self.block_size
        
        for bx, by, bz, bt in itertools.product(*(range(li) for li in self.L_coarse)):
            for k, uk in enumerate(self.ui_blocked[bx][by][bz][bt]):
                projected[bx, by, bz, bt, k] = innerproduct(uk, v[bx * lx: (bx + 1)*lx
                                                                , by * ly: (by + 1)*ly
                                                                , bz * lz: (bz + 1)*lz
                                                                , bt * lt: (bt + 1)*lt])
        return projected
    
    def v_prolong(self, v: torch.Tensor) -> torch.Tensor:
        """prolong coarse vector ``v`` to fine grid.

        Args:
            v: Coarse grid vector with shape (L_coarse[0], L_coarse[1], L_coarse[2],
                L_coarse[3], n_basis).

        Returns:
            torch.Tensor: Fine grid vector with shape (L_fine[0], L_fine[1], L_fine[2],
                L_fine[3], ...) and dtype torch.cdouble.
        """
        lx, ly, lz, lt = self.block_size
        prolonged = torch.zeros(self.L_fine + list(self.ui_blocked[0][0][0][0][0].shape[4:]), dtype=torch.cdouble)
        for bx, by, bz, bt in itertools.product(*(range(li) for li in self.L_coarse)):
            for k, uk in enumerate(self.ui_blocked[bx][by][bz][bt]):
                prolonged[bx * lx: (bx + 1)*lx
                        , by * ly: (by + 1)*ly
                        , bz * lz: (bz + 1)*lz
                        , bt * lt: (bt + 1)*lt] += uk * v[bx,by,bz,bt,k]
        return prolonged
    
    def get_coarse_operator(self,
                            fine_operator: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        """Given a fine operator ``fine_operator(psi)``, construct a coarse operator.

        In case of a 9-point operator, such as Wilson and Wilson-Clover Dirac operator,
        a significantly faster implementation can be achieved by using ``qcd_ml.qcd.dirac.coarsened.coarse_9point_op_NG``.

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
        torch.save([self.block_size, self.ui_blocked, self.n_basis, self.L_coarse, self.L_fine], filename)

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

    def get_basis_vectors(self) -> torch.Tensor:
        """Returns the basis vectors. This function is necessary because the basis vectors are stored
        "by-coarse-grid-index" and not on a fine grid.

        Returns:
            torch.Tensor: Tensor of shape (n_basis, L_fine[0], L_fine[1], L_fine[2],
                L_fine[3], 4, 3) containing all basis vectors reconstructed on the
                fine lattice. The dimensions 4 and 3 correspond to spin and color
                indices respectively.
        """
        result = torch.zeros(self.n_basis, *self.L_fine, 4, 3, dtype=torch.cdouble)
        for bx, by, bz, bt in itertools.product(*(range(li) for li in self.L_coarse)):
            for k, uk in enumerate(self.ui_blocked[bx][by][bz][bt]):
                result[k, bx * self.block_size[0]: (bx + 1)*self.block_size[0]
                      , by * self.block_size[1]: (by + 1)*self.block_size[1]
                      , bz * self.block_size[2]: (bz + 1)*self.block_size[2]
                      , bt * self.block_size[3]: (bt + 1)*self.block_size[3]] = uk
        return result
