import numpy as np

def make_L_D_S_Y(s, y):
    k = len(s)
    S = np.column_stack(s[:k])
    Y = np.column_stack(y[:k])

    L = np.zeros([k,k])
    d = np.zeros(k)

    for j in range(k):
        d[j] = np.dot(s[j], y[j])
        for i in range(j + 1, k):
            L[i, j] = np.dot(s[i], y[j])

    D = np.diag(d)

    return L, D, S, Y


def SR1_approx_Full(Bk, yk, sk, tol=1e-8):

    v = yk - Bk @ sk
    denom = np.inner(v, sk)

    if abs(denom) > tol * np.linalg.norm(v) * np.linalg.norm(sk):
        Bk = Bk + np.outer(v, v) / denom

    return Bk
    
def SR1_approx_Limited(B0, y, s):
    
    if len(s) == 1:
        B = np.outer((y[0] - B0@s[0]), (y[0] - B0@s[0]) )/ ( np.inner(y[0], s[0]) - np.inner(B0 @ s[0], s[0]) ) 
    else:
        L, D, S, Y = make_L_D_S_Y(s, y)
        Inv = np.linalg.inv(D + L + L.T - S.T @ B0 @ S)
        B = B0 + (Y - B0@S) @ Inv @ (Y - B0@S).T

    return B

def SR1_spar_Sparse(B0, y, s):
    """
    Sparse block SR1 Hessian approximation.
    Mel is currently set to sqrt(n) but will later be
    exposed as a tunable sparsity parameter.
    """

    # Build compact SR1 quantities
    L, D, S, Y = make_L_D_S_Y(s, y)

    # Compact SR1 form:  B = B0 + N M^{-1} N^T
    N = Y - B0 @ S
    M = D + L + L.T - S.T @ B0 @ S

    # Reduce eigenproblem via thin QR
    Q, R = np.linalg.qr(N, mode='reduced')

    # Compute projected matrix
    Z = np.linalg.solve(M, R.T)
    T = R @ Z

    # Eigen-decomposition in reduced space
    w, UT = np.linalg.eigh(T)
    U = Q @ UT

    # Sparsity level (placeholder: sqrt(n))
    Mel = int(np.floor(np.sqrt(10 * U.shape[0])))

    # Select rows with largest 2-norm
    row_norms_sq = np.sum(U**2, axis=1)
    top_idx = np.argpartition(row_norms_sq, -Mel)[-Mel:]

    # Re-orthonormalize selected rows
    U_sub = U[top_idx, :]
    Q_sub, _ = np.linalg.qr(U_sub, mode='reduced')

    # Embed sparse basis
    Q_til = np.zeros_like(U)
    Q_til[top_idx, :] = Q_sub

    # Sparse low-rank SR1 update
    B_til = B0 + Q_til @ np.diag(w) @ Q_til.T

    return B_til

def hess_approx(B, S, Y, approx_type="Sparse"):

    if approx_type == "Full":
        return SR1_approx_Full(B, Y, S)

    elif approx_type == "Limited":
        return SR1_approx_Limited(B, S, Y)

    elif approx_type == "Sparse":
        return SR1_spar_Sparse(B, S, Y)

    else:
        return SR1_spar_Sparse(B, S, Y)

