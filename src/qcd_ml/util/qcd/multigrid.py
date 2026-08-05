#!/usr/bin/env python3

"""
Provides Multigrid with zero point projection.
"""

from typing import Any, Callable, Dict, List, Tuple
import torch
import itertools
from qcd_ml.util.linear_algebra import innerproduct, norm

_STATE_DICT_KEYS = frozenset(("block_size", "block_basis", "n_basis", "L_coarse", "L_fine"))

def orthonormalize(vecs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Orthonormalize a list of vectors using the Gram-Schmidt process.
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
    """

    def __init__(self,
                 block_size: Tuple[int, ...],
                 block_basis: torch.Tensor,
                 n_basis: int,
                 L_coarse: Tuple[int, ...],
                 L_fine: Tuple[int, ...]) -> None:
        self.block_size = block_size
        self.block_basis = block_basis
        self.n_basis = n_basis
        self.L_coarse = L_coarse
        self.L_fine = L_fine

    def cuda(self) -> 'ZPP_Multigrid':
        """Move all basis vectors to CUDA device.

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
        """
        def operator(source_coarse: torch.Tensor) -> torch.Tensor:
            source_fine = self.v_prolong(source_coarse)
            dst_fine = fine_operator(source_fine)
            return self.v_project(dst_fine)
        return operator

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of this multigrid setup as a dictionary.

        Use ``torch.save(mg.state_dict(), filename)`` to store it and
        ``mg.load_state_dict(torch.load(filename))`` to restore it.

        Returns:
            dict: Mapping of attribute name to value, containing the keys
                ``block_size``, ``block_basis``, ``n_basis``, ``L_coarse``
                and ``L_fine``.
        """
        return {key: getattr(self, key) for key in _STATE_DICT_KEYS}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of a multigrid setup from a dictionary, in place.
        """
        missing = _STATE_DICT_KEYS - state_dict.keys()
        unexpected = state_dict.keys() - _STATE_DICT_KEYS
        if missing or unexpected:
            raise KeyError(f"missing keys: {sorted(missing)}, unexpected keys: {sorted(unexpected)}")

        for key in _STATE_DICT_KEYS:
            setattr(self, key, state_dict[key])

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any]) -> 'ZPP_Multigrid':
        """Construct a new multigrid setup from a state dictionary.
        """
        self = cls.__new__(cls)
        self.load_state_dict(state_dict)
        return self

