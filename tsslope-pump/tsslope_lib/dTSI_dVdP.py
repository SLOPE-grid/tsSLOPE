"""Computes partial derivatives of TSI w.r.t. X.
"""
import numpy as np
from numpy import conj, arange, diag, zeros, asmatrix, asarray
from scipy.sparse import issparse, csr_matrix as sparse
import torch
from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix as sparse
import time

def dTSI_dVdP(GPmodel, Pg, Qg, Pl, Ql, nb, ng, st_args):
    num_J_H, Mul_confi, gen_idx = st_args['num_J_H'], st_args['Mul_confi'], st_args['gen_idx']
    ng0 = len(gen_idx)

    model = GPmodel['model']
    X_max = GPmodel['X_max']
    X_min = GPmodel['X_min']
    y_mean = GPmodel['y_mean']
    y_std = GPmodel['y_std']

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

        # full Jacobian
        # start_time = time.time()
        Jacobian_mean = torch.autograd.functional.jacobian(mean_f, X)
        Jacobian_mean = Jacobian_mean * (y_std / (X_max/2.0))  # Converting from dy/dx to dY/dX
        # print(time.time()-start_time)

        Jacobian_std = torch.autograd.functional.jacobian(std_f, X)
        Jacobian_std = Jacobian_std * (y_std / (X_max/2.0))

        Jacobian_mean_np = Jacobian_mean.cpu().detach().numpy()
        Jacobian_std_np = Jacobian_std.cpu().detach().numpy()

        Jacobian_np = Mul_confi * Jacobian_std_np - Jacobian_mean_np
    else:
        # numerical Jacobian
        nx = 2 * ng
        step = 1e-3
        Jacobian_mean = torch.zeros((1, nx))
        for i in range(nx): # First-order central difference
            xp = X[0, :].reshape(1, -1).clone().detach()
            xm = X[0, :].reshape(1, -1).clone().detach()

            xp[0, i] = X[0, i] + step / 2
            xp = torch.autograd.Variable(torch.tensor(xp).float(), requires_grad=True)
            xp = xp - X_min
            xp = 2.0 * (xp / X_max) - 1.0
            fxp = model(xp)
            weights_xp = model.quad_weights.unsqueeze(-1).exp()
            fxp = (weights_xp * fxp.mean).sum(0) * y_std + y_mean

            xm[0, i] = X[0, i] - step / 2
            xm = torch.autograd.Variable(torch.tensor(xm).float(), requires_grad=True)
            xm = xm - X_min
            xm = 2.0 * (xm / X_max) - 1.0
            fxm = model(xm)
            weights_xm = model.quad_weights.unsqueeze(-1).exp()
            fxm = (weights_xm * fxm.mean).sum(0) * y_std + y_mean

            Jacobian_mean[:, i] = -1.0 * (fxp - fxm) / step

        Jacobian_mean_np = Jacobian_mean.cpu().detach().numpy()

    dTSI = lil_matrix((1, 2*nb+2*ng))

    GP_to_IPM = np.hstack([2*nb+pgen_ls[gen_idx], 2*nb+disp_load, 2*nb+ng+disp_load])
    Jacobian_np[0, ng0:] = -Jacobian_np[0, ng0:]
    dTSI[0, GP_to_IPM] = Jacobian_np[0, :]

    return dTSI
