"""Computes 2nd derivatives of TSI w.r.t. X.
"""
import numpy as np
from numpy import conj, arange, diag, zeros, asmatrix, asarray
from scipy.sparse import issparse, csr_matrix as sparse
import torch
from torch.autograd.functional import hessian
from torch.func import hessian as hessian_GP
from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix as sparse
import time


def d2TSI_dV2dP2(Surrogate, Pg, Qg, Pl, Ql, muTSI, st_args):
    if Surrogate['model_type'] == "CNF":
        return d2TSI_dV2dP2_CNF(Surrogate, Pg, Qg, Pl, Ql, muTSI, st_args)
    elif Surrogate['model_type'] == "CNN":
        return d2TSI_dV2dP2_CNN(Surrogate, Pg, Qg, Pl, Ql, muTSI, st_args)
    elif  Surrogate['model_type'] == "DKL":
        return d2TSI_dV2dP2_DKL(Surrogate, Pg, Qg, Pl, Ql, muTSI, st_args)

def x_to_std(x: torch.Tensor, scaler, x_space: str) -> torch.Tensor:
    """
    Convert x from raw to standardized space if needed.
    x_space: "raw" or "std".
    """
    x = x.view(-1)
    if x_space == "std":
        return x
    mean = scaler.mean_.view(-1).to(x.device, x.dtype)
    std = scaler.std_.view(-1).to(x.device, x.dtype)
    return (x - mean) / std

# --- Constraint value: c(x) = (1 - alpha) - F_Y(u0 | x) ---
def constraint_value(
    x_param: torch.Tensor,
    model,
    scaler,
    u0: float,
    alpha: float,
    x_space: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Chance constraint:
        c(x) = (1 - alpha) - P(Y <= u0 | x)
             = (1 - alpha) - F_Y(u0 | x) 

    We enforce c(x) >= 0.
    """
    # Map to standardized features for the model
    x_std = x_to_std(x_param, scaler, x_space).view(1, -1)
    y0 = torch.tensor([u0], device=device, dtype=dtype)
    F_u0 = model.cdf(y0, x_std).view(())       # scalar
    c = (1.0 - alpha) - F_u0
    return c

def constraint_hessian(
    x_param: torch.Tensor,
    model,
    scaler,
    u0: float,
    alpha: float,
    x_space: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Full Hessian of the constraint c(x) with respect to x_param.
    """
    def f(z):
        return constraint_value(z, model, scaler, u0, alpha, x_space, device, dtype)

    z = x_param.detach().clone().requires_grad_(True)
    H = torch.autograd.functional.hessian(f, z, create_graph=False)
    return H

def d2TSI_dV2dP2_CNF(CNFmodel, Pg, Qg, Pl, Ql, muTSI, st_args):
    model = CNFmodel['model']
    scaler = CNFmodel['scaler'] 
    ckpt = CNFmodel['ckpt'] 
    dtype = CNFmodel['dtype'] 
    u0 = CNFmodel['u0'] 
    alpha = CNFmodel['alpha'] 
    device = CNFmodel['device'] 
    x_space = CNFmodel['x_space'] 

    gen_idx = st_args['gen_idx']

    # Concatenate generators first, then loads (same as training)
    P_concat = np.concatenate([Pg[gen_idx], Pl], axis=0)  # (Ngen+Nload,)
    Q_concat = np.concatenate([Qg[gen_idx], Ql], axis=0)  # (Ngen+Nload,)

    # Per-sample layout: (2, Nunits)
    x_test_np = np.stack([P_concat, Q_concat], axis=0)  # (2, Nunits)
    x_test_np = x_test_np.reshape(-1)
    X = torch.tensor(x_test_np, dtype=dtype)

    H = constraint_hessian(X, model, scaler, u0, alpha, x_space, device, dtype)

    H = H.detach().cpu().numpy()

    return H

def d2TSI_dV2dP2_CNN(CNNmodel, Pg, Qg, Pl, Ql, muTSI, st_args):

    model = CNNmodel["model"]
    dtype = CNNmodel['dtype'] 
    active_gen_only = CNNmodel['active_gen_only']
    gen_idx = st_args['gen_idx']

    def model_scalar(pg_vector):
        pl_t = torch.tensor(Pl, dtype=dtype)
        ql_t = torch.tensor(Ql, dtype=dtype)

        X = torch.cat([pg_vector.clone(), pl_t, ql_t], dim=0)
        X = X.unsqueeze(0).unsqueeze(0)

        y = model(X)
        return y.sum()    # must be scalar

    if active_gen_only:
        pg = torch.tensor(Pg[gen_idx], dtype=dtype, requires_grad=True)
    else:
        pg = torch.tensor(Pg, dtype=dtype, requires_grad=True)

    H = hessian(model_scalar, pg)

    H = H.detach().cpu().numpy()

    return H

# derivative of f(s) > tau
def d2TSI_dV2dP2_CNN_Grad_UQ(CNNmodel, Pg, Qg, Pl, Ql, muTSI, st_args):

    model = CNNmodel["model"]
    dtype = CNNmodel['dtype'] 
    active_gen_only = CNNmodel['active_gen_only']
    gen_idx = st_args['gen_idx']
    beta = st_args['beta'] 

    def f_scalar(pg_vector, beta):
        pl_t = torch.tensor(Pl, dtype=dtype, requires_grad=False)
        ql_t = torch.tensor(Ql, dtype=dtype, requires_grad=False)

        # reconstruct input
        X = torch.cat([pg_vector, pl_t, ql_t], dim=0)
        X = X.unsqueeze(0).unsqueeze(0)  # (1, 1, N)

        model.eval()
        y = model(X).sum()

        # gradient of y w.r.t. pg_vector
        (grad_pg,) = torch.autograd.grad(
            y,
            pg_vector,
            create_graph=True   # IMPORTANT: needed for second derivative
        )

        pred = y - beta * torch.dot(grad_pg, grad_pg)
        return pred
    
    if active_gen_only:
        pg = torch.tensor(Pg[gen_idx], dtype=dtype, requires_grad=True)
    else:
        pg = torch.tensor(Pg, dtype=dtype, requires_grad=True)

    H = torch.autograd.functional.hessian(
    lambda p: f_scalar(p, beta=beta),
    pg)

    return H.detach().cpu().numpy()

def d2TSI_dV2dP2_GP(GPmodel, Pg, Qg, Pl, Ql, muTSI, st_args):
    nb, ng = st_args['numb_buses'], st_args['total_numb_gens']

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



# from line_profiler_pycharm import profile
# @profile
def d2TSI_dV2dP2_DKL(GPmodel, Pg, Qg, Pl, Ql, muTSI, st_args, ppc):
    nb, ng = st_args['numb_buses'], st_args['total_numb_gens']
    num_J_H, Mul_confi, gen_idx, syn_idx, rew_idx = st_args['num_J_H'], st_args['Mul_confi'], st_args['gen_idx'], st_args['syn_idx'], st_args['rew_idx']
    active_gen_only = Surrogate['active_gen_only']
    ng0 = len(gen_idx)

    model = GPmodel['model']
    X_max = GPmodel['X_max']
    X_min = GPmodel['X_min']
    y_mean = GPmodel['y_mean']
    y_std = GPmodel['y_std']
    ng0 = len(gen_rewsyn_genidx)

    model.eval()

    X_full_np = np.hstack([Pg, Pl]) * 100.0
    X_full = torch.tensor(X_full_np, dtype=torch.float32)

    X_part = X_full[:, :ng0].clone().detach().requires_grad_(True)   # (1, ng0) or (ns, ng0)
    X_const = X_full[:, ng0:].clone().detach()                       # (1, nPl)

    if torch.cuda.is_available():
        model = model.cuda()
        X_part = X_part.cuda()
        X_const = X_const.cuda()
        X_max_t = torch.tensor(X_max, dtype=torch.float32).cuda()
        X_min_t = torch.tensor(X_min, dtype=torch.float32).cuda()
    else:
        X_max_t = torch.tensor(X_max, dtype=torch.float32)
        X_min_t = torch.tensor(X_min, dtype=torch.float32)

    num_J_H = st_args['num_J_H']
    Mul_confi = st_args['Mul_confi']
    X_max_part = X_max_t[:ng0].reshape(-1)
    X_min_part = X_min_t[:ng0].reshape(-1)

    def _normalize_full(x_part: torch.Tensor) -> torch.Tensor:
        x_full = torch.cat([x_part, X_const], dim=1)
        x_full = x_full - X_min_t
        x_full = 2.0 * (x_full / X_max_t) - 1.0
        x_full = torch.clamp(x_full, -1.0, 1.0)
        return x_full

    HT = lil_matrix((2*nb+2*ng, 2*nb+2*ng))
    
    def mean_f(x_part):
        Xn = _normalize_full(x_part)
        return model.likelihood(model(Xn)).mean[0]

    def std_f(x_part):
        Xn = _normalize_full(x_part)
        return model.likelihood(model(Xn)).stddev[0]

    def mean_df(x_part):
        return torch.autograd.functional.jacobian(mean_f, x_part, create_graph=True).sum(0)

    def std_df(x_part):
        return torch.autograd.functional.jacobian(std_f, x_part, create_graph=True).sum(0)

    # X_max[2*ng0-1:] = -X_max[2*ng0-1:]  # dP和Pl方向相反

    Hessian_mean = hessian_GP(mean_f)(X_part)
    Hessian_mean = Hessian_mean.permute(1, 0, 2)
    Hessian_mean = Hessian_mean * y_std / torch.mm(X_max_part.reshape(-1, 1), X_max_part.reshape(1, -1)) * 2 * 2
                
    Hessian_std = hessian_GP(std_f)(X_part)
    Hessian_std = Hessian_std.permute(1, 0, 2)
    Hessian_std = Hessian_std * y_std / torch.mm(X_max_part.reshape(-1, 1), X_max_part.reshape(1, -1)) * 2 * 2
                
    Hessian_mean = torch.autograd.functional.jacobian(mean_df, X_part)
    Hessian_mean = Hessian_mean.permute(1, 0, 2)
    Hessian_mean = Hessian_mean * y_std / torch.mm(X_max_part.reshape(-1, 1), X_max_part.reshape(1, -1)) * 2 * 2
    
    Hessian_std = torch.autograd.functional.jacobian(std_df, X_part)
    Hessian_std = Hessian_std.permute(1, 0, 2)
    Hessian_std = Hessian_std * y_std / torch.mm(X_max_part.reshape(-1, 1), X_max_part.reshape(1, -1)) * 2 * 2
    
    Hessian_mean_np = Hessian_mean.cpu().detach().numpy()
    Hessian_std_np = Hessian_std.cpu().detach().numpy()

    Hessian_np = Hessian_mean_np.copy()

    Hessian_np[0, :, :] = muTSI * (Mul_confi * Hessian_std_np[0, :, :] - Hessian_mean_np[0, :, :])

    return HT


def d2TSI_dV2dP2_DKL(GPmodel, Pg, Qg, Pl, Ql, muTSI, st_args, ppc):
    nb, ng = st_args['numb_buses'], st_args['total_numb_gens']
    num_J_H, Mul_confi, gen_idx, syn_idx, rew_idx = st_args['num_J_H'], st_args['Mul_confi'], st_args['gen_idx'], st_args['syn_idx'], st_args['rew_idx']
    active_gen_only = Surrogate['active_gen_only']
    ng0 = len(gen_idx)

    model = GPmodel['model']
    likelihood = GPmodel['likelihood']
    X_max = GPmodel['X_max']
    X_min = GPmodel['X_min']
    y_mean = GPmodel['y_mean']
    y_std = GPmodel['y_std']
    ng0 = len(gen_rewsyn_genidx)

    model.eval()
    likelihood.eval()

    Pl = st_args['PL']
    Ql = st_args['QL']

    if active_gen_only:
        active_syn_idx = list(set(syn_idx) & set(gen_idx))
        active_rew_idx = list(set(rew_idx) & set(gen_idx))
        Pg_active_syn_idx = Pg[active_syn_idx].reshape(1, -1)
        Pg_active_rew_idx = Pg[active_rew_idx].reshape(1, -1)
        # Qg_active_syn_idx = Qg[active_syn_idx].reshape(1, -1)
        # Qg_active_rew_idx = Qg[active_rew_idx].reshape(1, -1)

        X_part_np = np.hstack([Pg_active_rew_idx, Pg_active_syn_idx])
    else:
        Pg_syn_idx = Pg[syn_idx].reshape(1, -1)
        Pg_rew_idx = Pg[rew_idx].reshape(1, -1)
        Qg_syn_idx = Qg[syn_idx].reshape(1, -1)
        Qg_rew_idx = Qg[rew_idx].reshape(1, -1)

        X_part_np = np.hstack([Pg_rew_idx, Pg_syn_idx])

    X_const_np = Pl.reshape(1, -1)    

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        X_max_t = torch.tensor(X_max, dtype=torch.float64).cuda()
        X_min_t = torch.tensor(X_min, dtype=torch.float64).cuda()
        X_part = torch.tensor(X_part_np, dtype=torch.float64, requires_grad=True).cuda()
        X_const = torch.tensor(X_const_np, dtype=torch.float64).cuda()
    else:
        X_max_t = torch.tensor(X_max, dtype=torch.float64)
        X_min_t = torch.tensor(X_min, dtype=torch.float64)
        X_part = torch.tensor(X_part_np, dtype=torch.float64, requires_grad=True)
        X_const = torch.tensor(X_const_np, dtype=torch.float64)

    X_max_part = X_max_t[:ng0]
    X_min_part = X_min_t[:ng0]

    def constraint_f_part(Xp):
        X_full = torch.cat([Xp, X_const], dim=1)

        X_norm = X_full - X_min_t
        X_norm = 2.0 * (X_norm / X_max_t) - 1.0
        X_norm = torch.clamp(X_norm, -1.0, 1.0)

        GPpre = likelihood(model(X_norm))
        TSI_mean = GPpre.mean[0] * y_std + y_mean
        TSI_std = GPpre.stddev[0] * y_std

        return Mul_confi * TSI_std - TSI_mean  # means TSI_interval_half - TSI_mean < 0 

    Hessian_part = torch.autograd.functional.hessian(constraint_f_part, X_part)

    scale = (2.0 / X_max_part).reshape(-1)
    Hessian_part = Hessian_part * torch.outer(scale, scale)

    Hessian_part_np = Hessian_part.detach().cpu().numpy().squeeze()

    return Hessian_part_np

