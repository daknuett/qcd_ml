"""
This module provides coarsened operators. These are projected operators from a fine 
grid onto a coarse grid.

Currently the following operators are implemented:

    - ``coarse_9point_op_NG``: Coarse 9-point operators on a Non-Gauge coarse grid.
      For 9-point operators (Wilson, Wilson Clover) using ZPP_Multigrid for coarsening.
"""
import torch
import itertools

class coarse_9point_op_NG:
    """
    Coarse 9-point operators on a Non-Gauge coarse grid.

    Construct as such::

        mg = ZPP_Multigrid(...)
        Q = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)

        coarse_op = coarse_9point_op_NG.from_operator_and_multigrid(Q, mg)

    This operator is significantly faster than the operator constructed by ``ZPP_Multigrid.get_coarse_operator(Q)``.
    """
    def __init__(self, pseudo_gauge_forward, pseudo_gauge_backward, pseudo_mass, L_coarse):
        self.pseudo_gauge_forward = pseudo_gauge_forward
        self.pseudo_gauge_backward = pseudo_gauge_backward
        self.pseudo_mass = pseudo_mass
        self.L_coarse = L_coarse

        def pseudo_gauge_apply(ps_gauge, vec):
            return torch.einsum("abcdij,abcdj->abcdi", ps_gauge, vec)
    
        self.pseudo_gauge_transform = pseudo_gauge_apply
        
    def __call__(self, x):
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
    def from_operator_and_multigrid(cls, operator, mg):
        """
        Constructs the pseudo-mass and pseudo-gauge for the given operator
        and a given restrict/prolong.

        Use as such::
            
            mg = ZPP_Multigrid(...)
            Q = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)

            coarse_op = coarse_9point_op_NG.from_operator_and_multigrid(Q, mg)

        """
        pseudo_gauge_forward = torch.zeros(4, *mg.L_coarse, mg.n_basis, mg.n_basis, dtype=torch.cdouble)
        pseudo_gauge_backward = torch.zeros(4, *mg.L_coarse, mg.n_basis, mg.n_basis, dtype=torch.cdouble)
        pseudo_mass = torch.zeros(*mg.L_coarse, mg.n_basis, mg.n_basis, dtype=torch.cdouble)
        
        coarse_op = mg.get_coarse_operator(operator)
        vec = torch.zeros(*mg.L_coarse, mg.n_basis)

        def update_idx_p(idx, mu):
            idx[mu] = (idx[mu] + 1) % mg.L_coarse[mu]
            return tuple(idx)
        def update_idx_m(idx, mu):
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

class coarse_9point_op_IFG:
    """
    Coarse 9-point operators on a coarse grid which Inherits Fine Gauge, i.e.,
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

    """
    def __init__(self, pseudo_gauge_forward, pseudo_gauge_backward, pseudo_mass, L_coarse):
        self.pseudo_gauge_forward = pseudo_gauge_forward
        self.pseudo_gauge_backward = pseudo_gauge_backward
        self.pseudo_mass = pseudo_mass
        self.L_coarse = L_coarse

        def pseudo_gauge_apply(ps_gauge, vec):
            return torch.einsum("abcdijkl,abcdjl->abcdik", ps_gauge, vec)
    
        self.pseudo_gauge_transform = pseudo_gauge_apply
        
    def __call__(self, x):
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
    def from_operator_and_pooling(cls, operator, pooling):
        """
        Constructs the pseudo-mass and pseudo-gauge for the given operator
        and a given restrict/prolong.

        Use as such::

            tfp = v_ProjectLayer(...)
            Q = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)
            coarse_op_9p = coarse_9point_op_IFG.from_operator_and_pooling(Q, tfp)
        """
        pseudo_gauge_forward = torch.zeros(4, *pooling.L_coarse, 4, 4, 3, 3, dtype=torch.cdouble)
        pseudo_gauge_backward = torch.zeros(4, *pooling.L_coarse, 4, 4, 3, 3, dtype=torch.cdouble)
        pseudo_mass = torch.zeros(*pooling.L_coarse, 4, 4, 3, 3, dtype=torch.cdouble)
        
        coarse_op = lambda x: pooling.v_project(torch.stack([operator(pooling.v_prolong(torch.stack([x]))[0])]))[0]
        vec = torch.zeros(*pooling.L_coarse, 4,3, dtype=torch.cdouble)

        def update_idx_p(idx, mu):
            idx[mu] = (idx[mu] + 1) % pooling.L_coarse[mu]
            return tuple(idx)
        def update_idx_m(idx, mu):
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
