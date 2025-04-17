"""Computes 2nd derivatives of TSI w.r.t. X.
"""
import numpy as np
from numpy import conj, arange, diag, zeros, asmatrix, asarray
from scipy.sparse import issparse, csr_matrix as sparse
import torch
from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix as sparse
import time

def d2TSI_dV2dP2(GPmodel, Pg, Qg, Pl, Ql, nb, ng, muTSI, st_args):
    model = GPmodel['model']
    X_max = GPmodel['X_max']
    X_min = GPmodel['X_min']
    y_mean = GPmodel['y_mean']
    y_std = GPmodel['y_std']
    gen_idx = st_args['gen_idx']
    ng0 = len(gen_idx)

    model.eval()

    disp_load = st_args['disp_load']
    disp_load = np.array(disp_load)
    pgen_ls = st_args['pgen_ls']
    pgen_ls = np.array(pgen_ls)

    X = np.hstack([Pg, Pl, Ql])
    X = torch.autograd.Variable(torch.tensor(X).float(), requires_grad=True)

    if torch.cuda.is_available():
        model.cuda()
        X, X_max, X_min, y_mean, y_std = X.cuda(), X_max.cuda(), X_min.cuda(), y_mean.cuda(), y_std.cuda()

    num_J_H = st_args['num_J_H']
    Mul_confi = st_args['Mul_confi']
    if num_J_H == 0:
        X = X - X_min
        X = 2.0 * (X / X_max) - 1.0
        X = torch.clamp(X, -1, 1)

        def mean_f(X):
            return (model.quad_weights.unsqueeze(-1).exp() * model.likelihood(model(X)).mean).sum()

        def std_f(X):
            return (model.quad_weights.unsqueeze(-1).exp() * model.likelihood(model(X)).stddev).sum()

        def mean_df(X):
            return torch.autograd.functional.jacobian(mean_f, X, create_graph=True).sum(0)

        def std_df(X):
            return torch.autograd.functional.jacobian(std_f, X, create_graph=True).sum(0)

        # start_time = time.time()
        Hessian_mean = torch.autograd.functional.jacobian(mean_df, X)
        Hessian_mean = Hessian_mean.permute(1, 0, 2)
        Hessian_mean = Hessian_mean * y_std / torch.mm(X_max.reshape(-1, 1), X_max.reshape(1, -1)) * 2 * 2
        # print(time.time() - start_time)

        Hessian_std = torch.autograd.functional.jacobian(std_df, X)
        Hessian_std = Hessian_std.permute(1, 0, 2)
        Hessian_std = Hessian_std * y_std / torch.mm(X_max.reshape(-1, 1), X_max.reshape(1, -1)) * 2 * 2

        Hessian_mean_np = Hessian_mean.cpu().detach().numpy()
        Hessian_std_np = Hessian_std.cpu().detach().numpy()

        Hessian_np = Hessian_mean_np.copy()

        Hessian_np[0, :, :] = muTSI * (Mul_confi * Hessian_std_np[0, :, :] - Hessian_mean_np[0, :, :])
    else:

        def mean_f(X):
            return (model.quad_weights.unsqueeze(-1).exp() * model.likelihood(model(X)).mean).sum()

        def std_f(X):
            return (model.quad_weights.unsqueeze(-1).exp() * model.likelihood(model(X)).stddev).sum()

        # numerical Hessian
        nx = 2 * ng
        step = 1e-3

        Hessian_mean = torch.zeros((nx, nx))
        for i in range(nx):  # Second-order central difference
            xa = X[0, :].reshape(1, -1).reshape(1, -1).clone().detach()
            xb = X[0, :].reshape(1, -1).reshape(1, -1).clone().detach()

            xa[0, i] = xa[0, i] + step
            xa = torch.autograd.Variable(torch.tensor(xa).float(), requires_grad=True)
            xa = xa - X_min
            xa = 2.0 * (xa / X_max) - 1.0
            Jacobian_mean_xa = torch.autograd.functional.jacobian(mean_f, xa)
            Jacobian_mean_xa = Jacobian_mean_xa * (y_std / (X_max / 2.0))

            xb[0, i] = xb[0, i]
            xb = torch.autograd.Variable(torch.tensor(xb).float(), requires_grad=True)
            xb = xb - X_min
            xb = 2.0 * (xb / X_max) - 1.0
            Jacobian_mean_xb = torch.autograd.functional.jacobian(mean_f, xb)
            Jacobian_mean_xb = Jacobian_mean_xb * (y_std / (X_max / 2.0))

            Hessian_mean[:, i] = -1.0 * muTSI * (Jacobian_mean_xa[0, 0:nx] - Jacobian_mean_xb[0, 0:nx]) / step

        Hessian_mean_np = np.zeros((1, nx, nx))
        Hessian_mean_np[0, :, :] = Hessian_mean.cpu().detach().numpy()

    HT = lil_matrix((2*nb+2*ng, 2*nb+2*ng))

    GP_to_IPM = np.hstack([2*nb+pgen_ls[gen_idx], 2*nb+disp_load, 2*nb+ng+disp_load])
    Hessian_np[0, ng0:, 0:ng0] = -Hessian_np[0, ng0:, 0:ng0]  # [+ -; - +]
    Hessian_np[0, 0:ng0, ng0:] = -Hessian_np[0, 0:ng0, ng0:]  # [+ -; - +]
    for i, m in enumerate(GP_to_IPM):
        for j, n in enumerate(GP_to_IPM):
            HT[m, n] = Hessian_np[0, i, j]

    return HT
