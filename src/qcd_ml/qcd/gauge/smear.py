"""
This module provides smearing of gauge links. Currenltly the following smearing algorithms are implemented:

    - Stout smearing (``stout``) [1]_ [2]_


.. [1]: 10.1103/PhysRevD.69.054501
.. [2]: 10.1007/978-3-642-01850-3


"""
import torch

from ...base.paths  import PathBuffer
from ...base.operations import SU3_group_compose

class compiled_stout:
    """
    This class represents the "compiled" stout operation for a given ..math:`\rho` and ..math:`U` gauge field.
    This is useful because typically several smearing steps are performed and the costly computation of 
    ..math:`\exp(iQ)` can be done only once.
    """
    def __init__(self
                 , rho: torch.tensor
                 , transform_matrix: torch.tensor):
        self.rho = rho
        self.transform_matrix = transform_matrix
        
    def __call__(self, link_field):
        U_trans = torch.zeros_like(link_field)
        for mu, (expiQmu, Umu) in enumerate(zip(self.transform_matrix, link_field)):
            U_trans[mu] = SU3_group_compose(expiQmu, Umu)
        return U_trans
            
        

class stout:
    r"""
    This class is used to construct the stout smearing operation.
    See [1]_ and [2]_ for details of the smearing algorithm.

    This class is an abstract representation of the smearing operation with only 
    the parameter ..math:`\rho` being specified. The actual smearing operation is
    performed as such::

        algorithm = stout(rho)
        smearer = algorithm(U)

        for i in range(n_smearing_steps):
            U = smearer(U)

    Classmethods are provided for common ..math:`\rho` matrices:

    - ``constant_rho``: ..math:`\rho` is a constant matrix.
    - ``spatial_only``: ..math:`\rho` is a constant matrix with the temporal components set to zero.
    """
    def __init__(self, rho: torch.tensor):
        if rho.shape != (4,4):
            raise ValueError(f"expected 4x4 tensor, got {rho.shape}")
        self.rho = rho

        # rho_mumu != 0 is technically forbidden. See
        # Gattringer Lang 2012 eq. (6.52).
        for mu in range(4):
            self.rho[mu,mu] = 0

    def __call__(self, U):
        if isinstance(U, list):
            U = torch.stack(U)
        
        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        staple_paths1 = [[Hm(nu, Hp(mu, Hp(nu, [])))
                         for nu in range(4)] for mu in range(4)]
        staple_paths2 = [[Hp(nu, Hp(mu, Hm(nu, [])))
                         for nu in range(4)] for mu in range(4)]
        staple1 = [[PathBuffer(U, pnu).accumulated_U.adjoint() for pnu in pmu] for pmu in staple_paths1]
        staple2 = [[PathBuffer(U, pnu).accumulated_U.adjoint() for pnu in pmu] for pmu in staple_paths2]
        
        staples = torch.stack([torch.stack(stapmu) for stapmu in staple1]) + torch.stack([torch.stack(stapmu) for stapmu in staple2])
       
        U_tensor_adj = U.adjoint()
        Omega_mu = torch.einsum("mn,mnabcdij,mabcdjk->mabcdik", self.rho, staples, U_tensor_adj)

        hermitian = Omega_mu.adjoint() - Omega_mu
        trace = torch.einsum("mabcdii->mabcd", hermitian)

        identity = torch.clone(hermitian)
        identity[:,:,:,:,:] = torch.eye(3,3, dtype=torch.cdouble)
        trace_removal = torch.einsum("mabcdij,mabcd->mabcdij", identity, trace)
        traceless_hermitian = 1j/2 * (hermitian -  trace_removal / 3)

        transform_matrix = torch.matrix_exp(1j * traceless_hermitian)

        return compiled_stout(self.rho, transform_matrix)

    @classmethod
    def constant_rho(cls, rho):
        rho_tensor = rho * torch.ones(4,4, dtype=torch.cdouble)
        return cls(rho_tensor)
        
    @classmethod
    def spatial_only(cls, rho):
        rho_tensor = rho * torch.ones(4,4, dtype=torch.cdouble)
        rho_tensor[:,-1] *= 0
        rho_tensor[-1,:] *= 0
        return cls(rho_tensor)
