import numpy as np

def analyze_hessian(
    H,
    sparsity_tol=1e-3,
    verbose=True,
    save_Hess = False,
    ev_nonzero_tol = 1e-10
):
    """
    Comprehensive Hessian diagnostic analysis.
    """

    n = H.shape[0]
    assert H.shape[0] == H.shape[1], "Hessian must be square"

    # ============================================================
    # Eigenvalue analysis
    # ============================================================
    eigvals, eigvecs = np.linalg.eigh(H)

    eig_min = eigvals.min()
    eig_max = eigvals.max()

    # Eigenvector that correspond to nonzero eigenvalues
    non_ev_indx = np.where(np.abs(eigvals) > ev_nonzero_tol)[0]

    nonzero_ev_indx = non_ev_indx + 1

    nonzero_evecs = eigvecs[:, non_ev_indx]

    # ============================================================
    # Sparsity / second-order sensitivity structure
    # ============================================================
    threshold = sparsity_tol * np.max(absH)
    large_mask = absH > threshold

    density = np.sum(large_mask) / H.size

    # ============================================================
    # Collect results
    # ============================================================
    if save_Hess:
        results = {
            "H": H,
            "eigenvalues": eigvals,
            "hessian_density": density,
            "nonzero_ev_indx": nonzero_ev_indx,
            "nonzero_evecs": nonzero_evecs,
        }
    else:

        results = {
            "eigenvalues": eigvals,
            "hessian_density": density,
            "nonzero_ev_indx": nonzero_ev_indx,
            "nonzero_evecs": nonzero_evecs,
        }

    # ============================================================
    # Optional summary printout
    # ============================================================
    if verbose:
        print("===================================================")
        print(" Hessian Diagnostic Summary")
        print("===================================================")
        print(f"Matrix size                 : {n} x {n}")
        print(f"Min eigenvalue              : {eig_min:.2e}")
        print(f"Max eigenvalue              : {eig_max:.2e}")
        print(f"Hessian density             : {density:.3e}")

    return results
