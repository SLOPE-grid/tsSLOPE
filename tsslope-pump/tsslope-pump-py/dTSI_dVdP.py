"""Computes partial derivatives of TSI w.r.t. X.
"""
import numpy as np
from numpy import conj, arange, diag, zeros, asmatrix, asarray
from scipy.sparse import issparse, csr_matrix as sparse
import torch
from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix as sparse
import time

def dTSI_dVdP(Surrogate, Pg, Qg, Pl, Ql, st_args):

    if Surrogate['model_type'] == "CNF":
        return dTSI_dVdP_CNF(Surrogate, Pg, Qg, Pl, Ql, st_args)
    elif Surrogate['model_type'] == "CNN":
        return dTSI_dVdP_CNN(Surrogate, Pg, Qg, Pl, Ql, st_args)
    else:
        return dTSI_dVdP_CNN(Surrogate, Pg, Qg, Pl, Ql, st_args)

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
    t0 = time.time()
    F_u0 = model.cdf(y0, x_std).view(())       # scalar
    total_time = time.time() - t0
    # print(f"Total time to calculate model.cdf in gradient: {total_time}")
    c = (1.0 - alpha) - F_u0
    return c

def constraint_grad(
    x_param: torch.Tensor,
    model,
    scaler,
    u0: float,
    alpha: float,
    x_space: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Return constraint value and its gradient with respect to x_param.
    """
    
    x = x_param.detach().clone().requires_grad_(True)
    c = constraint_value(x, model, scaler, u0, alpha, x_space, device, dtype)
    (g,) = torch.autograd.grad(c, x, create_graph=False, retain_graph=False)
    return g

def dTSI_dVdP_CNF(CNFmodel, Pg, Qg, Pl, Ql, st_args):
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

    g = constraint_grad(X, model, scaler, u0, alpha, x_space, device, dtype)
    
    return g.detach().cpu().numpy()

# derivative of f(s) > tau
def dTSI_dVdP_CNN(CNNmodel, Pg, Qg, Pl, Ql, st_args):

    model = CNNmodel["model"]
    dtype = CNNmodel['dtype'] 
    active_gen_only = CNNmodel['active_gen_only']
    gen_idx = st_args['gen_idx']

    # build full input vector as one differentiable tensor
    if active_gen_only:
        pg = torch.tensor(Pg[gen_idx], dtype=dtype)
    else:
        pg = torch.tensor(Pg, dtype=dtype)

    pl = torch.tensor(Pl, dtype=dtype)
    ql = torch.tensor(Ql, dtype=dtype)    

    X = torch.cat([pg, pl, ql], dim=0).requires_grad_(True)

    X_in = X.view(1, 1, -1)

    model.eval()
    y = model(X_in).sum()
    y.backward()

    # gradient wrt pg are the first len(pg) components
    grad_pg = X.grad[:len(pg)].clone()

    return grad_pg.detach().cpu().numpy()

# derivative of f(s) > tau
def dTSI_dVdP_CNN_Grad_UQ(CNNmodel, Pg, Qg, Pl, Ql, st_args):

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

    (grad_pg,) = torch.autograd.grad(
        f_scalar(pg, beta=beta),
        pg,
        create_graph=False,
        retain_graph=False)

    return grad_pg.detach().cpu().numpy()

def dTSI_dVdP_GP(GPmodel, Pg, Qg, Pl, Ql, st_args):
    nb, ng = st_args['numb_buses'], st_args['total_numb_gens']
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

    if active_gen_only:
        active_syn_idx = list(set(syn_idx) & set(gen_idx))
        active_rew_idx = list(set(rew_idx) & set(gen_idx))
        Pg_active_syn_idx = Pg[active_syn_idx].reshape(1, -1)
        Pg_active_rew_idx = Pg[active_rew_idx].reshape(1, -1)
        Qg_active_syn_idx = Qg[active_syn_idx].reshape(1, -1)
        Qg_active_rew_idx = Qg[active_rew_idx].reshape(1, -1)

        Pg_input = np.hstack([Pg_active_rew_idx, Pg_active_syn_idx])
        Qg_input = np.hstack([Qg_active_rew_idx, Qg_active_syn_idx])
    else:
        Pg_syn_idx = Pg[syn_idx].reshape(1, -1)
        Pg_rew_idx = Pg[rew_idx].reshape(1, -1)
        Qg_syn_idx = Qg[syn_idx].reshape(1, -1)
        Qg_rew_idx = Qg[rew_idx].reshape(1, -1)

        Pg_input = np.hstack([Pg_rew_idx, Pg_syn_idx])
        Qg_input = np.hstack([Qg_rew_idx, Qg_syn_idx])

    X = np.hstack([Pg_input, Pl, Ql])
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
        Jacobian_mean = torch.autograd.functional.jacobian(mean_f, X)
        Jacobian_mean = Jacobian_mean * (y_std / (X_max/2.0))  # Converting from dy/dx to dY/dX

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


# from line_profiler_pycharm import profile
# @profile
# def dTSI_dVdP(GPmodel, Pg, Qg, Pl, Ql, st_args):

#     nb, ng = st_args['numb_buses'], st_args['total_numb_gens']
#     num_J_H, Mul_confi, gen_rewsyn_genidx = st_args['num_J_H'], st_args['Mul_confi'], st_args['gen_rewsyn_genidx']
#     ng0 = len(gen_rewsyn_genidx)

#     model = GPmodel['model']
#     likelihood = GPmodel['model']
#     X_max = GPmodel['X_max']
#     X_min = GPmodel['X_min']
#     y_mean = GPmodel['y_mean']
#     y_std = GPmodel['y_std']

#     model.eval()

#     X = np.hstack([Pg, Pl]) 
#     X = torch.autograd.Variable(torch.tensor(X).float(), requires_grad=True)

#     if torch.cuda.is_available():
#         model.cuda()
#         X, X_max, X_min = X.cuda(), torch.tensor(X_max).cuda(), torch.tensor(X_min).cuda()

#     X = X - X_min
#     X = 2.0 * (X / X_max) - 1.0
#     X = torch.clamp(X, -1, 1)  # 限制极端值导致的零梯度

#     def mean_f(X):
#         return model.likelihood(model(X)).mean[0] # +y_mean和不加的梯度一致

#     def std_f(X):
#         return model.likelihood(model(X)).stddev[0]

#     def mean_df(X):
#         return torch.autograd.functional.jacobian(mean_f, X, create_graph=True).sum(0)

#     def std_df(X):
#         return torch.autograd.functional.jacobian(std_f, X, create_graph=True).sum(0)

#     # full Jacobian
#     Jacobian_mean = torch.autograd.functional.jacobian(mean_f, X)
#     Jacobian_mean = Jacobian_mean * (y_std / (X_max/2.0))  # 从dy/dx转换为dY/dX

#     Jacobian_std = torch.autograd.functional.jacobian(std_f, X)
#     Jacobian_std = Jacobian_std * (y_std / (X_max/2.0))

#     Jacobian_mean_np = Jacobian_mean.cpu().detach().numpy()
#     Jacobian_std_np = Jacobian_std.cpu().detach().numpy()

#     Jacobian_np = Mul_confi * Jacobian_std_np - Jacobian_mean_np

#     return dTSI

# def dTSI_dVdP(GPmodel, Pg, Qg, Pl, Ql, st_args):

#     nb, ng = st_args['numb_buses'], st_args['total_numb_gens']
#     num_J_H, Mul_confi, gen_rewsyn_genidx = st_args['num_J_H'], st_args['Mul_confi'], st_args['gen_rewsyn_genidx']
#     ng0 = len(gen_rewsyn_genidx)

#     model = GPmodel['model']
#     likelihood = GPmodel['likelihood']
#     X_max = GPmodel['X_max']
#     X_min = GPmodel['X_min']
#     y_mean = GPmodel['y_mean']
#     y_std = GPmodel['y_std']

#     model.eval()
#     likelihood.eval()

#     Pl = st_args['PL']
#     Ql = st_args['QL']

#     if active_gen_only:
#         active_syn_idx = list(set(syn_idx) & set(gen_idx))
#         active_rew_idx = list(set(rew_idx) & set(gen_idx))
#         Pg_active_syn_idx = Pg[active_syn_idx].reshape(1, -1)
#         Pg_active_rew_idx = Pg[active_rew_idx].reshape(1, -1)
#         # Qg_active_syn_idx = Qg[active_syn_idx].reshape(1, -1)
#         # Qg_active_rew_idx = Qg[active_rew_idx].reshape(1, -1)

#         Pg_input = np.hstack([Pg_active_rew_idx, Pg_active_syn_idx])
#         # Qg_input = np.hstack([Qg_active_rew_idx, Qg_active_syn_idx])
#     else:
#         Pg_syn_idx = Pg[syn_idx].reshape(1, -1)
#         Pg_rew_idx = Pg[rew_idx].reshape(1, -1)
#         Qg_syn_idx = Qg[syn_idx].reshape(1, -1)
#         Qg_rew_idx = Qg[rew_idx].reshape(1, -1)

#         Pg_input = np.hstack([Pg_rew_idx, Pg_syn_idx])
#         # Qg_input = np.hstack([Qg_rew_idx, Qg_syn_idx])

#     X = np.hstack([Pg_input, Pl])
#     X = torch.autograd.Variable(torch.tensor(X).float(), requires_grad=True)

#     if torch.cuda.is_available():
#         model = model.cuda()
#         likelihood = likelihood.cuda()
#         X = X.cuda()
#         X_max = torch.tensor(X_max, dtype=torch.float64).cuda()
#         X_min = torch.tensor(X_min, dtype=torch.float64).cuda()
#     else:
#         X_max = torch.tensor(X_max, dtype=torch.float64)
#         X_min = torch.tensor(X_min, dtype=torch.float64)

#     X_norm = X - X_min
#     X_norm = 2.0 * (X_norm / X_max) - 1.0
#     X_norm = torch.clamp(X_norm, -1.0, 1.0)

#     def constraint_f(X_in):
#         GPpre = likelihood(model(X_in))
#         TSI_mean = GPpre.mean[0] * y_std + y_mean
#         TSI_std = GPpre.stddev[0] * y_std
#         return Mul_confi * TSI_std - TSI_mean  # means TSI_interval_half - TSI_mean < 0 

#     dTSI = torch.autograd.functional.jacobian(constraint_f, X_norm)
#     dTSI = dTSI * (2.0 / X_max)   # chain rule: d/dX_raw

#     return dTSI
