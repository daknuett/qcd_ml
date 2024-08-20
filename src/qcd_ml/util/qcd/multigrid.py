#!/usr/bin/env python3

"""
Provides Multigrid with zero point projection.
"""

import torch
import itertools
import numpy as np

innerproduct = lambda x,y: (x.conj() * y).sum()
norm = lambda x: torch.sqrt(innerproduct(x, x).real)

def orthonormalize(vecs):
    basis = []
    for vec in vecs:
        for b in basis:
            vec = vec - innerproduct(b, vec) * b
        vec = vec / norm(vec)
        basis.append(vec)
    return basis


class ZPP_Multigrid:
    """
    Multigrid with zeropoint projection.

    Use .v_project and .v_prolong to project and prolong vectors.
    Use .get_coarse_operator to construct a coarse operator.

    use ZPP_Multigrid.gen_from_fine_vectors([random vectors], [i, j, k, l], lambda b, xo: <solve Dx = b for x>)
    to construct a ZPP_Multigrid.
    """
    def __init__(self, block_size, ui_blocked, n_basis, L_coarse, L_fine):
        self.block_size = block_size
        self.ui_blocked = ui_blocked
        self.n_basis = n_basis
        self.L_coarse = L_coarse
        self.L_fine = L_fine

    @classmethod
    def gen_from_fine_vectors(cls
                              , fine_vectors
                              , block_size
                              , solver
                              , verbose=False):
        """
        Used to generate a multigrid setup using fine vectors, a block size and a solver.

        solver should be 
            (x, info) = solver(b, x0)
        which solves
            D x = b
        
        we will choose
            b = torch.zeros_like(x0)

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
                    ui_blocked[bx][by][bz][bt]  = []
                ui_blocked[bx][by][bz][bt].append(u_block)

            # Orthogonalize over block
            ui_blocked[bx][by][bz][bt] = orthonormalize(ui_blocked[bx][by][bz][bt])

        return cls(block_size, ui_blocked, n_basis, L_coarse, L_fine)

    
    def v_project(self, v):
        """
        project fine vector v to coarse grid.
        """
        projected = torch.zeros(self.L_coarse + [self.n_basis], dtype=torch.cdouble)
        lx, ly, lz, lt = self.block_size
        
        for bx, by, bz, bt in itertools.product(*(range(li) for li in self.L_coarse)):
            for k, uk in enumerate(self.ui_blocked[bx][by][bz][bt]):
                projected[bx, by, bz, bt, k] = innerproduct(v[bx * lx: (bx + 1)*lx
                                                            , by * ly: (by + 1)*ly
                                                            , bz * lz: (bz + 1)*lz
                                                            , bt * lt: (bt + 1)*lt], uk)
        return projected

    
    def v_prolong(self, v):
        """
        prolong coarse vector v to fine grid.
        """
        lx, ly, lz, lt = self.block_size
        prolonged = torch.zeros(self.L_fine + list(self.ui_blocked[0][0][0][0][0].shape[4:]), dtype=torch.cdouble)
        for bx, by, bz, bt in itertools.product(*(range(li) for li in self.L_coarse)):
            for k, uk in enumerate(self.ui_blocked[bx][by][bz][bt]):
                prolonged[bx * lx: (bx + 1)*lx
                        , by * ly: (by + 1)*ly
                        , bz * lz: (bz + 1)*lz
                        , bt * lt: (bt + 1)*lt] += v[bx,by,bz,bt,k].conj() * uk
        return prolonged

    
    def get_coarse_operator(self, fine_operator):
        def operator(source_coarse):
            source_fine = self.v_prolong(source_coarse)
            dst_fine = fine_operator(source_fine)
            return self.v_project(dst_fine)
        return operator

    def save(self, filename):
        """
        This is a stupid implementation. Saves all arguments as a list.
        """
        torch.save([self.block_size, self.ui_blocked, self.n_basis, self.L_coarse, self.L_fine], filename)

    @classmethod
    def load(cls, filename):
        """
        This is a stupid implementation. Loads all arguments as a list.
        """
        args = torch.load(filename)
        return cls(*tuple(args))
