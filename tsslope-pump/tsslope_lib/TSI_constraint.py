"""Computes partial derivatives of TSI w.r.t. X.
"""
import numpy as np
from numpy import conj, arange, diag, zeros, asmatrix, asarray
from scipy.sparse import issparse, csr_matrix as sparse
import torch
from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix as sparse
import time

#def TSI_constraint(GPmodel, Pg, Pl, Ql, st_args):

def TSI_constraint(GPmodel, Pg, Qg, st_args):
    Mul_confi = st_args['Mul_confi']
    gen_idx = st_args['gen_idx']
    model = GPmodel['model']
    X_max = GPmodel['X_max']
    X_min = GPmodel['X_min']
    y_mean = GPmodel['y_mean']
    y_std = GPmodel['y_std']

    model.eval()

    disp_load = st_args['disp_load']
    pgen_ls = st_args['pgen_ls']
    Pg_GP = Pg.reshape(1, -1)[:, pgen_ls]
    Qg_GP = Qg.reshape(1, -1)[:, pgen_ls]
    Pg_GP = Pg_GP[:, gen_idx]
    Qg_GP = Qg_GP[:, gen_idx]
    Pl_GP = -Pg.reshape(1, -1)[:, disp_load]
    Ql_GP = -Qg.reshape(1, -1)[:, disp_load]
    X = np.hstack([Pg_GP, Pl_GP, Ql_GP])
    X = X - X_min.numpy()
    X = 2.0 * (X / X_max.numpy()) - 1.0
    X = np.clip(X, -1, 1)

    X = torch.autograd.Variable(torch.tensor(X).float(), requires_grad=True)

    if torch.cuda.is_available():
        model.cuda()
        X, y_mean, y_std = X.cuda(), y_mean.cuda(), y_std.cuda()

    GPpre = model(X)
    weights = model.quad_weights.unsqueeze(-1).exp()
    TSI_mean = (weights * GPpre.mean).sum(0) * y_std + y_mean
    TSI_mean = TSI_mean.cpu().detach().numpy()
    TSI_std = (weights * GPpre.stddev).sum(0) * y_std
    TSI_std = TSI_std.cpu().detach().numpy()
    TSI_interval_half = Mul_confi * TSI_std

    return TSI_interval_half - TSI_mean
