"""
qcd_ml.util.solver
==================

Solvers for systems of linear equations.
"""
import torch 
import numpy as np



def update_qr(H, s, c, j):
    """
    Runs and updates the QR decomposition of the matrix H.
    This function is used internally by GMRES_inner.
    """
    # Apply previous Givens rotations to the new column of H
    for i in range(j):
        tmp = -s[i] * H[i, j] + c[i] * H[i + 1, j]
        H[i, j] = np.conjugate(c[i]) * H[i, j] + np.conjugate(s[i]) * H[i + 1, j]
        H[i + 1, j] = tmp

    # Compute the new Givens rotation
    beta = np.sqrt(np.abs(H[j,j])**2 + np.abs(H[j+1,j])**2)

    s[j] = H[j+1,j] / beta
    c[j] = H[j,j] / beta

    H[j,j] = beta
    H[j+1,j] = 0.0

    
def update_result(x, Z, gamma, H, y, j):
    """
    Updates the result of GMRES_inner by going from the Krylov space 
    (spanned by Z, coefficients H and gamma) to the solution x.
    """
    for i in reversed(range(j + 1)):
        y[i] = (gamma[i] - np.dot(H[i, i+1:j+1], y[i+1:j+1])) / H[i,i]

    for i in range(j+1):
        x += y[i] * Z[i]

    return x



def GMRES_inner(A, b, x0, stopat_residual, niterations, innerproduct, preconditioner):
    """
    Inner GMRES, i.e., ``niterations`` without restart.
    """
    r0 = b - A(x0)
    v1 = r0 / innerproduct(r0, r0) ** 0.5

    x = x0

    H = np.zeros((niterations + 1, niterations), dtype=np.complex128)
    s = np.zeros(niterations + 1, dtype=np.complex128)
    c = np.zeros(niterations + 1, dtype=np.complex128)
    y = np.zeros(niterations + 1, np.complex128)
    gamma = np.zeros(niterations + 1, dtype=np.complex128)
    gamma[0] = innerproduct(r0, r0) ** 0.5
    history = np.zeros(niterations)

    V = [v1] + [None] * (niterations)
    if preconditioner is not None:
        Z = [None] * (niterations)
        Z_or_V = Z
    else:
        Z_or_V = V

    breakdown = False
    converged = False

    for j in range(niterations):
        if preconditioner is not None:
            Z_or_V[j] = preconditioner(V[j])
        Avj = A(Z_or_V[j])
        for i in range(j + 1):
            H[i, j] = innerproduct(V[i], Avj)

        vjp1_hat = Avj
        for i in range(j+1):
            vjp1_hat = vjp1_hat -  H[i, j] * V[i]

        H[j + 1, j] = np.abs(innerproduct(vjp1_hat, vjp1_hat)) ** 0.5

        if H[j + 1, j] == 0.0:
            breakdown = True
            break

        v_jp1 = vjp1_hat / H[j + 1, j]
        V[j + 1] = v_jp1

        update_qr(H, s, c, j)

        gamma[j + 1] = - s[j] * gamma[j]
        gamma[j] = np.conj(c[j]) * gamma[j]

        res = np.abs(gamma[j+1])
        history[j] = res
        
        if res < stopat_residual:
            converged = True
            break

    x = update_result(x, Z_or_V, gamma, H, y, j)

    return x, {"converged": converged, "breakdown": breakdown, "res": res, "k": j + 1, "target_residual": stopat_residual, "history": history}


def GMRES(A, b, x0
          , maxiter=1000
          , inner_iter=30
          , eps=1e-5
          , innerproduct=lambda x,y: (x.conj() * y).sum()
          , preconditioner=None
          , verbose=False
          ):
    """
    Implementation of thr GMRES algorithm for solving the linear system Ax = b.
    
    ``A``: callable or a matrix that allows ``A @ x`` to be computed.
    ``b``: right-hand side of the linear system.
    ``x0``: initial guess for the solution.
    ``maxiter``: maximum number of iterations.
    ``inner_iter``: number of iterations before restarting.
    ``eps``: tolerance for the residual. The true tolerance is ``eps * ||b||`` or ``eps * ||r0||``.
    ``innerproduct``: inner product function.
    ``preconditioner``: preconditioner function. Should be a function that takes a vector and returns a vector.
    """

    if hasattr(A, "__call__"):
        apply_A = A
    else:
        apply_A = lambda x: A @ x

    norm_b = np.abs(innerproduct(b, b)) ** 0.5
    stopat_residual = None
    if norm_b > 1e-10:
        stopat_residual = eps * norm_b

    r0 = b - apply_A(x0)

    norm_r0 = np.abs(innerproduct(r0, r0)) ** 0.5

    if norm_r0 < 1e-10 and stopat_residual is None:
        raise ValueError("b and A@x0 are zero (<1e-10)")
    if stopat_residual is None:
        stopat_residual = eps * norm_r0

    hist = np.zeros(maxiter)
    iters = 0
    x = x0
    # This is a curious edge case found in a single workflow.
    # Under exotic circumstances info may be uninitialized.
    info = {"warning": "no application of GMRES_inner yet"}

    while iters < maxiter:
        niters_this = min((inner_iter, maxiter - iters))
        x, info = GMRES_inner(apply_A, b, x, stopat_residual, niters_this, innerproduct, preconditioner)
        hist[iters: iters+niters_this] = info["history"]
        iters += info["k"]
        if verbose:
            print(f"GMRES: iter {iters}, res {info['res']}, target {info['target_residual']}")
        if info["converged"] or info["breakdown"]:
            break
        if iters >= maxiter:
            break

    info["k"] = iters
    info["history"] = hist[:iters]

    return x, info
