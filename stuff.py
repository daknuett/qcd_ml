import gpt as g
import numpy as np


def lattice2ndarray(lattice):
    """ 
    Converts a gpt (https://github.com/lehner/gpt) lattice to a numpy ndarray 
    keeping the ordering of axes as one would expect.
    Example::
        q_top = g.qcd.gauge.topological_charge_5LI(U_smeared, field=True)
        plot_scalar_field(lattice2ndarray(q_top))
    """
    shape = lattice.grid.fdimensions
    shape = list(reversed(shape))
    if lattice[:].shape[1:] != (1,):
        shape.extend(lattice[:].shape[1:])
   
    result = lattice[:].reshape(shape)
    result = np.swapaxes(result, 0, 3)
    result = np.swapaxes(result, 1, 2)
    return result

def ndarray2lattice(ndarray, grid, lat_constructor):
    """
    Converts an ndarray to a gpt lattice, it is the inverse 
    of lattice2ndarray.

    Example::
        lat = ndarray2lattice(arr, g.grid([4,4,4,8], g.double), g.vspincolor)
    """
    lat = lat_constructor(grid)
    data = np.swapaxes(ndarray, 0, 3)
    data = np.swapaxes(data, 1, 2)
    lat[:] = data.reshape([data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3]] + list(data.shape[4:]))
    return lat


def GMRES_torch(A, b, x0, maxiter=1000, eps=1e-4
              , regulate_b_norm=1e-4
              , innerproduct=None
              , prec=None):
    """
    GMRES solver.
    
    innerproduct is a function (vec,vec)->scalar which is a product.
    prec is a function vec->vec.

    Literature:
    - https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
    - https://www-users.cse.umn.edu/~saad/Calais/PREC.pdf

    Authors:
    - Daniel Knüttel 2024
    """
    if hasattr(A, "__call__"):
        apply_A = lambda x: A(x)
    else:
        apply_A = lambda x: A @ x

    if innerproduct is None:
        innerproduct = lambda x,y: (x.conj() * y).sum()

    
    rk = b - apply_A(x0)

    b_norm = np.sqrt(innerproduct(b, b).real) + regulate_b_norm
    
    rk_norm = np.sqrt(innerproduct(rk, rk).real)
    res = rk_norm / b_norm
    if rk_norm / b_norm <= eps:
        return x0, {"converged": True, "k": 0}

    vk = rk / rk_norm

    v = [None, vk]
    
    cs = np.zeros(maxiter + 2, np.complex128)
    sn = np.zeros(maxiter + 2, np.complex128)
    gamma = np.zeros(maxiter + 2, np.complex128)
    gamma[1] = rk_norm
    H = [None]
    
    converged = False
    for k in range(1, maxiter + 1):
        if prec is not None:
            z = prec(v[k])
        else:
            z = v[k]
        qk = apply_A(z)
        
        Hk = np.zeros(k + 2, np.complex128)
        for i in range(1, k + 1):
            Hk[i] = innerproduct(v[i], qk)
        for i in range(1, k + 1):
            qk -= Hk[i] * v[i]
            
        Hk[k+1] = np.sqrt(innerproduct(qk, qk).real)
        v.append(qk / Hk[k+1])

        for i in range(1, k):
            # (c   s ) [a]   [a']
            # (-s* c*) [b] = [b']
            tmp = cs[i+1] * Hk[i] + sn[i+1] * Hk[i+1]
            Hk[i+1] = -np.conj(sn[i+1]) * Hk[i] + np.conj(cs[i+1]) * Hk[i+1]
            Hk[i] = tmp
            

        beta = np.sqrt(np.abs(Hk[k])**2 + np.abs(Hk[k + 1])**2)

        # ( c    s )[a]   [X]
        # (-s*   c*)[b] = [0]
        # is solved by 
        # s* = b; c* = a
        sn[k+1] = np.conj(Hk[k+1]) / beta
        cs[k+1] = np.conj(Hk[k]) / beta
        Hk[k] = cs[k+1] * Hk[k] + sn[k+1] * Hk[k+1]
        Hk[k+1] = 0
        
        
        gamma[k+1] = -np.conj(sn[k+1]) * gamma[k]
        gamma[k] = cs[k+1] * gamma[k]
        
        H.append(Hk)
        res = np.abs(gamma[k+1]) / b_norm

        if np.abs(gamma[k+1]) / b_norm <= eps:
            converged = True
            break

    y = np.zeros(k+1, np.complex128)
    for i in reversed(range(1, k + 1)):
        overlap = 0
        for j in range(i+1, k+1):
            overlap += H[j][i] * y[j]
        y[i] = (gamma[i] - overlap) / H[i][i]
    if prec is None:
        x = x0 + sum(yi * vi for yi, vi in zip(y[1:], v[1:]))
    else:
        x = x0 + sum(yi * prec(vi) for yi, vi in zip(y[1:], v[1:]))
    return x, {"converged": converged, "k": k, "res": res}

def GMRES_restarted(A, b, x0, max_restart=10, maxiter_inner=100, eps=1e-4
              , regulate_b_norm=1e-3
              , innerproduct=None
              , prec=None):
    x = x0
    total_iterations = 0
    for rs in range(max_restart):
        x, ret = GMRES_torch(A, b, x, maxiter=maxiter_inner, eps=1e-4
              , regulate_b_norm=regulate_b_norm
              , innerproduct=None
              , prec=None)
        total_iterations += ret["k"]
        print("restarting with res:", ret["res"])
        if ret["converged"]:
            break
    ret["k"] = total_iterations
    return x, ret

def v_project(block_size, ui_blocked, n_basis, L_coarse, v):
    projected = torch.complex(torch.zeros(L_coarse + [n_basis], dtype=torch.double)
                              , torch.zeros(L_coarse + [n_basis], dtype=torch.double))
    lx, ly, lz, lt = block_size
    
    for bx, by, bz, bt in itertools.product(*(range(li) for li in L_coarse)):
        for k, uk in enumerate(ui_blocked[bx][by][bz][bt]):
            projected[bx, by, bz, bt, k] = innerproduct(v[bx * lx: (bx + 1)*lx
                                                        , by * ly: (by + 1)*ly
                                                        , bz * lz: (bz + 1)*lz
                                                        , bt * lt: (bt + 1)*lt], uk)
    return projected


def v_prolong(block_size, ui_blocked, n_basis, L_coarse, v):
    L_fine = [bi*li for bi,li in zip(block_size, L_coarse)]
    prolonged = torch.complex(torch.zeros(L_fine + list(ui_blocked[0][0][0][0][0].shape[4:]), dtype=torch.double)
                              , torch.zeros(L_fine + list(ui_blocked[0][0][0][0][0].shape[4:]), dtype=torch.double))
    for bx, by, bz, bt in itertools.product(*(range(li) for li in L_coarse)):
        for k, uk in enumerate(ui_blocked[bx][by][bz][bt]):
            prolonged[bx * lx: (bx + 1)*lx
                    , by * ly: (by + 1)*ly
                    , bz * lz: (bz + 1)*lz
                    , bt * lt: (bt + 1)*lt] += v[bx,by,bz,bt,k] * uk
    return prolonged


def get_coarse_operator(block_size, ui_blocked, n_basis, L_coarse, fine_operator):
    def operator(source_coarse):
        source_fine = v_prolong(block_size, ui_blocked, n_basis, L_coarse, source_coarse)
        dst_fine = fine_operator(source_fine)
        return v_project(block_size, ui_blocked, n_basis, L_coarse, dst_fine)
    return operator


def complex_mse_loss(output, target):
    err = (output - target)
    return (err * err.conj()).real.sum()


def l2norm(v):
    return (v * v.conj()).real.sum()