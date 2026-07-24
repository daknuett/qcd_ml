"""This module provides coarsened operators. These are projected operators from a fine 
grid onto a coarse grid.

Currently the following operators are implemented:

    - ``coarse_9point_op_NG``: Coarse 9-point operators on a Non-Gauge coarse grid.
      For 9-point operators (Wilson, Wilson Clover) using ZPP_Multigrid for coarsening.
      This class provides two methods for constructing coarse operators:
      
        - ``from_operator_and_multigrid``: Generic method that works for any operator.
        - ``from_dirac_operator_and_multigrid``: Specialized method for Wilson(-clover) Dirac operators
          that uses precomputation for better performance.
    - ``coarse_9point_op_IFG``: Coarse 9-point operators on a coarse grid that inherits 
      its gauge field from a fine gauge field. An example is the use of a ``v_ProjectLayer``.
"""
import torch
import itertools
from typing import Tuple, Type, Any, Callable

class coarse_9point_op_NG:
    """Coarse 9-point operators on a Non-Gauge coarse grid.

    Construct as such::

        mg = ZPP_Multigrid(...)
        Q = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)

        coarse_op = coarse_9point_op_NG.from_operator_and_multigrid(Q, mg)
        # or for Wilson(-clover) operators:
        coarse_op = coarse_9point_op_NG.from_dirac_operator_and_multigrid(Q, mg)

    This operator is significantly faster than the operator constructed by ``ZPP_Multigrid.get_coarse_operator(Q)``.
    The ``from_dirac_operator_and_multigrid`` method is specialized for Wilson(-clover) Dirac
    operators and uses decomposed application (apply_diag, apply_pos_hop, apply_neg_hop) for
    significantly better performance during initialization.
    """

    def __init__(self, pseudo_gauge_forward: torch.Tensor, pseudo_gauge_backward: torch.Tensor, pseudo_mass: torch.Tensor, L_coarse: Tuple[int, ...]) -> None:
        """For internal use only, use ``coarse_9point_op_NG.from_operator_and_multigrid`` 
        or ``coarse_9point_op_NG.from_dirac_operator_and_multigrid`` to obtain
        a coarsened operator.
        """
        self.pseudo_gauge_forward = pseudo_gauge_forward
        self.pseudo_gauge_backward = pseudo_gauge_backward
        self.pseudo_mass = pseudo_mass
        self.L_coarse = L_coarse

        def pseudo_gauge_apply(ps_gauge: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
            return torch.einsum("abcdij,abcdj->abcdi", ps_gauge, vec)
    
        self.pseudo_gauge_transform = pseudo_gauge_apply
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        result = self.pseudo_gauge_transform(self.pseudo_mass, x)
        for mu in range(4):
            result_mu = torch.roll(self.pseudo_gauge_transform(self.pseudo_gauge_forward[mu], x), 1, mu)
            result_mu += torch.roll(self.pseudo_gauge_transform(self.pseudo_gauge_backward[mu], x), -1, mu)
            # This is a curious edge case. We double-accounted for the
            # links.
            if self.L_coarse[mu] == 2:
                result += result_mu / 2
            else:
                result += result_mu

        return result

    @classmethod
    def from_operator_and_multigrid(cls: Type['coarse_9point_op_NG'], operator: Callable, mg: Any) -> 'coarse_9point_op_NG':
        """Constructs the pseudo-mass and pseudo-gauge for the given operator
        and a given restrict/prolong.

        Use as such::

            mg = ZPP_Multigrid(...)
            Q = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)

            coarse_op = coarse_9point_op_NG.from_operator_and_multigrid(Q, mg)

        Args:
            operator: The fine-grid operator to coarsen.
            mg: The multigrid object providing coarse grid information.

        Returns:
            A new coarse_9point_op_NG instance.
        """
        pseudo_gauge_forward = torch.zeros(4, *mg.L_coarse, mg.n_basis, mg.n_basis, dtype=torch.cdouble)
        pseudo_gauge_backward = torch.zeros(4, *mg.L_coarse, mg.n_basis, mg.n_basis, dtype=torch.cdouble)
        pseudo_mass = torch.zeros(*mg.L_coarse, mg.n_basis, mg.n_basis, dtype=torch.cdouble)
        
        coarse_op = mg.get_coarse_operator(operator)
        vec = torch.zeros(*mg.L_coarse, mg.n_basis, dtype=torch.cdouble)

        def update_idx_p(idx: list, mu: int) -> Tuple:
            idx[mu] = (idx[mu] + 1) % mg.L_coarse[mu]
            return tuple(idx)
        def update_idx_m(idx: list, mu: int) -> Tuple:
            idx[mu] = (idx[mu] + mg.L_coarse[mu] - 1) % mg.L_coarse[mu]
            return tuple(idx)
        
        for x,y,z,t in itertools.product(*(range(bi) for bi in mg.L_coarse)):
                for i in range(mg.n_basis):
                    vec *= 0
                    vec[x,y,z,t, i] = 1
                    response = coarse_op(vec)
                    pseudo_mass[x,y,z,t,:,i] = response[x,y,z,t]    
                    for mu in range(4):
                        pseudo_gauge_forward[mu, x,y,z,t, :,i] = response[update_idx_p([x,y,z,t], mu)]
                        pseudo_gauge_backward[mu, x,y,z,t, :,i] = response[update_idx_m([x,y,z,t], mu)]

        return cls(pseudo_gauge_forward, pseudo_gauge_backward, pseudo_mass, mg.L_coarse)

    @classmethod
    def from_dirac_operator_and_multigrid(cls: Type['coarse_9point_op_NG'], fine_op: Callable, mg: Any) -> 'coarse_9point_op_NG':
        """Construct a coarse operator for Wilson(-clover) Dirac operator using precomputation.
        
        This method only works for Wilson(-clover) Dirac operators that have
        apply_diag, apply_pos_hop, and apply_neg_hop methods.
        
        This implementation is significantly faster than the generic get_coarse_operator
        for Wilson-type operators as it precomputes the coarse operator structure directly.
        
        Note: This is similar to from_operator_and_multigrid but uses a different,
        more direct precomputation method specific to Wilson-type operators.
        
        Use as such::

            mg = ZPP_Multigrid(...)
            Q = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)

            coarse_op = coarse_9point_op_NG.from_dirac_operator_and_multigrid(Q, mg)
            # coarse_op is a callable that can be applied to coarse vectors

        Args:
            fine_op: The fine-grid Wilson(-clover) Dirac operator with methods:
                - apply_diag: Apply diagonal part
                - apply_pos_hop: Apply positive hopping terms
                - apply_neg_hop: Apply negative hopping terms
            mg: The multigrid object (ZPP_Multigrid) providing coarse grid information.

        Returns:
            Callable: A function that applies the coarse operator to a coarse grid vector.
        """
        # Only works for Wilson(-clover) Dirac operator
        N = mg.n_basis
        coarse_op_diag = torch.zeros(
            (*mg.L_coarse, N, N),
            dtype=torch.cdouble,
        )
        coarse_op_pos_hop = torch.zeros(
            (*mg.L_coarse, N, N, 4),
            dtype=torch.cdouble,
        )
        coarse_op_neg_hop = torch.zeros(
            (*mg.L_coarse, N, N, 4),
            dtype=torch.cdouble,
        )
        
        for idx in range(N):
            rhs_full = torch.zeros(
                (*mg.L_coarse, N),
                dtype=torch.cdouble,
            )
            rhs_full[..., idx] = 1

            rhs_prolonged_full = mg.v_prolong(rhs_full)

            checkerboard_even = (
                sum(
                    torch.meshgrid(
                        *[
                            torch.arange(d) // s
                            for (d, s) in zip(mg.L_fine, mg.block_size)
                        ],
                        indexing="ij",
                    )
                )
                % 2
            )
            checkerboard_odd = 1 - checkerboard_even

            rhs_prolonged_even = torch.einsum(
                "...,...sc->...sc", checkerboard_even, rhs_prolonged_full
            )
            rhs_prolonged_odd = torch.einsum(
                "...,...sc->...sc", checkerboard_odd, rhs_prolonged_full
            )

            diag_coarse_full = mg.v_project(fine_op.apply_diag(rhs_prolonged_full))

            pos_hop_coarse_even = torch.stack(
                [
                    mg.v_project(fine_op.apply_pos_hop(rhs_prolonged_even, mu))
                    for mu in range(4)
                ],
                dim=-1,
            )
            pos_hop_coarse_odd = torch.stack(
                [
                    mg.v_project(fine_op.apply_pos_hop(rhs_prolonged_odd, mu))
                    for mu in range(4)
                ],
                dim=-1,
            )
            neg_hop_coarse_even = torch.stack(
                [
                    mg.v_project(fine_op.apply_neg_hop(rhs_prolonged_even, mu))
                    for mu in range(4)
                ],
                dim=-1,
            )
            neg_hop_coarse_odd = torch.stack(
                [
                    mg.v_project(fine_op.apply_neg_hop(rhs_prolonged_odd, mu))
                    for mu in range(4)
                ],
                dim=-1,
            )

            checkerboard_even_coarse = (
                sum(
                    torch.meshgrid(
                        *[torch.arange(d) for d in mg.L_coarse],
                        indexing="ij",
                    )
                )
                % 2
            )
            checkerboard_odd_coarse = 1 - checkerboard_even_coarse

            coarse_op_diag[..., idx] = (
                diag_coarse_full
                + torch.einsum(
                    "...,...k->...k",
                    checkerboard_even_coarse,
                    (
                        torch.sum(pos_hop_coarse_even, dim=-1)
                        + torch.sum(neg_hop_coarse_even, dim=-1)
                    ),
                )
                + torch.einsum(
                    "...,...k->...k",
                    checkerboard_odd_coarse,
                    (
                        torch.sum(pos_hop_coarse_odd, dim=-1)
                        + torch.sum(neg_hop_coarse_odd, dim=-1)
                    ),
                )
            )
            coarse_op_pos_hop[..., idx, :] = torch.einsum(
                "...,...km->...km", checkerboard_odd_coarse, pos_hop_coarse_even
            ) + torch.einsum(
                "...,...km->...km", checkerboard_even_coarse, pos_hop_coarse_odd
            )
            coarse_op_neg_hop[..., idx, :] = torch.einsum(
                "...,...km->...km", checkerboard_odd_coarse, neg_hop_coarse_even
            ) + torch.einsum(
                "...,...km->...km", checkerboard_even_coarse, neg_hop_coarse_odd
            )

        pseudo_mass = coarse_op_diag
        
        pseudo_gauge_forward = torch.zeros(4, *mg.L_coarse, mg.n_basis, mg.n_basis, dtype=torch.cdouble)
        pseudo_gauge_backward = torch.zeros(4, *mg.L_coarse, mg.n_basis, mg.n_basis, dtype=torch.cdouble)
        for mu in range(4):
            pseudo_gauge_forward[mu] = torch.roll(coarse_op_pos_hop[...,mu], -1, dims=mu)
            pseudo_gauge_backward[mu] = torch.roll(coarse_op_neg_hop[...,mu], 1, dims=mu)
            if mg.L_coarse[mu] == 2:
                # Bodge to be compatible to unusual __call__ implementation
                pseudo_gauge_forward[mu] *= 2
                pseudo_gauge_backward[mu] *= 2

        return cls(pseudo_gauge_forward, pseudo_gauge_backward, pseudo_mass, mg.L_coarse)


class coarse_9point_op_IFG:
    """Coarse 9-point operators on a coarse grid which Inherits Fine Gauge, i.e.,
    the gauge from the fine grid is used implicitly for the operator.
    There may be a gauge field on the coarse grid, but this gauge field is ignored.

    This class can be considered as a significant acceleration of::

        tfp = v_ProjectLayer(...)

        def coarse_op_pr(Q):
            def op(x):
                with torch.no_grad():
                    return tfp.v_project(torch.stack([Q(tfp.v_prolong(torch.stack([x]))[0])]))[0]
            return op

    Construct as such::

        tfp = v_ProjectLayer(...)

        with torch.no_grad():
            coarse_op_9p = coarse_9point_op_IFG.from_operator_and_pooling(Q, tfp)

    The operator has two effective pseudo-gauge fields (forward and backward directions)
    that define how information is transferred between the coarse sites.
    """

    def __init__(self, pseudo_gauge_forward: torch.Tensor, pseudo_gauge_backward: torch.Tensor, pseudo_mass: torch.Tensor, L_coarse: Tuple[int, ...]) -> None:
        """For internal use only, to construct the operator use 
        ``coarse_9point_op_IFG.from_operator_and_pooling``.
        """
        self.pseudo_gauge_forward = pseudo_gauge_forward
        self.pseudo_gauge_backward = pseudo_gauge_backward
        self.pseudo_mass = pseudo_mass
        self.L_coarse = L_coarse

        def pseudo_gauge_apply(ps_gauge: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
            return torch.einsum("abcdijkl,abcdjl->abcdik", ps_gauge, vec)
    
        self.pseudo_gauge_transform = pseudo_gauge_apply
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        result = self.pseudo_gauge_transform(self.pseudo_mass, x)
        for mu in range(4):
            result_mu = torch.roll(self.pseudo_gauge_transform(self.pseudo_gauge_forward[mu], x), 1, mu)
            result_mu += torch.roll(self.pseudo_gauge_transform(self.pseudo_gauge_backward[mu], x), -1, mu)
            # This is a curious edge case. We double-accounted for the
            # links.
            if self.L_coarse[mu] == 2:
                result += result_mu / 2
            else:
                result += result_mu

        return result

    @classmethod
    def from_operator_and_pooling(cls: Type['coarse_9point_op_IFG'], operator: Callable, pooling: Any) -> 'coarse_9point_op_IFG':
        """Constructs the pseudo-mass and pseudo-gauge for the given operator
        and a given restrict/prolong.

        Use as such::

            tfp = v_ProjectLayer(...)
            Q = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)
            coarse_op_9p = coarse_9point_op_IFG.from_operator_and_pooling(Q, tfp)

        Args:
            operator: The fine-grid operator to coarsen.
            pooling: The pooling layer object (v_ProjectLayer) providing coarse grid information.

        Returns:
            A new coarse_9point_op_IFG instance.
        """
        pseudo_gauge_forward = torch.zeros(4, *pooling.L_coarse, 4, 4, 3, 3, dtype=torch.cdouble)
        pseudo_gauge_backward = torch.zeros(4, *pooling.L_coarse, 4, 4, 3, 3, dtype=torch.cdouble)
        pseudo_mass = torch.zeros(*pooling.L_coarse, 4, 4, 3, 3, dtype=torch.cdouble)
        
        coarse_op = lambda x: pooling.v_project(torch.stack([operator(pooling.v_prolong(torch.stack([x]))[0])]))[0]
        vec = torch.zeros(*pooling.L_coarse, 4,3, dtype=torch.cdouble)

        def update_idx_p(idx: list, mu: int) -> Tuple:
            idx[mu] = (idx[mu] + 1) % pooling.L_coarse[mu]
            return tuple(idx)
        def update_idx_m(idx: list, mu: int) -> Tuple:
            idx[mu] = (idx[mu] + pooling.L_coarse[mu] - 1) % pooling.L_coarse[mu]
            return tuple(idx)
        
        for x,y,z,t in itertools.product(*(range(bi) for bi in pooling.L_coarse)):
                for i in range(4):
                    for j in range(3):
                        vec *= 0
                        vec[x,y,z,t, i,j] = 1
                        response = coarse_op(vec)
                        pseudo_mass[x,y,z,t, :,i, :,j] = response[x,y,z,t]    
                        for mu in range(4):
                            pseudo_gauge_forward[mu, x,y,z,t, :,i, :,j] = response[update_idx_p([x,y,z,t], mu)]
                            pseudo_gauge_backward[mu, x,y,z,t, :,i, :,j] = response[update_idx_m([x,y,z,t], mu)]

        return cls(pseudo_gauge_forward, pseudo_gauge_backward, pseudo_mass, pooling.L_coarse)


