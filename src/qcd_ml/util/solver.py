import torch 
import numpy as np


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
              , innerproduct=innerproduct
              , prec=prec)
        total_iterations += ret["k"]
        #print("restarting with res:", ret["res"])
        if ret["converged"]:
            break
    ret["k"] = total_iterations
    return x, ret
