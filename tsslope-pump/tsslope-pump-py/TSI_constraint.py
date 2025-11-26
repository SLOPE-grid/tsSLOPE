"""Computes partial derivatives of TSI w.r.t. X.
"""
import numpy as np
from numpy import conj, arange, diag, zeros, asmatrix, asarray
from scipy.sparse import issparse, csr_matrix as sparse
import torch
from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix as sparse
import time

def TSI_constraint(Surrogate, Pg, Qg, st_args):
    if Surrogate['model_type'] == "CNN":
        return TSI_constraint_CNN(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "UQ_CNN":
        return TSI_constraint_UQ_CNN(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "DSPP":
        return TSI_constraint_GP(Surrogate, Pg, Qg, st_args)

def TSI_constraint_CNN(Surrogate, Pg, Qg, st_args):
    model = Surrogate['model']

    model.eval()

    PL = st_args['PL']
    QL = st_args['QL']

    Pg_input = Pg.reshape(1, -1)
    Qg_input = Qg.reshape(1, -1)
    Pl_input = -PL.reshape(1, -1)
    Ql_input = -QL.reshape(1, -1)

    X_np = np.hstack([Pg_input, Pl_input, Ql_input])

    X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0)

    pred = model(X).item()

    return pred

# f = μ − β sqrt(var)
def TSI_constraint_UQ_CNN(Surrogate, Pg, Qg, st_args):
    Mul_confi = st_args['Mul_confi']
    model = Surrogate['model']

    model.eval()

    PL = st_args['PL']
    QL = st_args['QL']

    Pg_input = Pg.reshape(1, -1)
    Qg_input = Qg.reshape(1, -1)
    Pl_input = -PL.reshape(1, -1)
    Ql_input = -QL.reshape(1, -1)

    X_np = np.hstack([Pg_input, Pl_input, Ql_input])

    X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0)

    mean_pred, var_pred = model(X)

    f = mean_pred - Mul_confi * torch.sqrt(var_pred + 1e-8)

    return f.squeeze().item()

def TSI_constraint_GP(GPmodel, Pg, Qg, st_args):
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
