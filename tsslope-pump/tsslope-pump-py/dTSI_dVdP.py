"""Computes partial derivatives of TSI w.r.t. X.
"""
import numpy as np
from numpy import conj, arange, diag, zeros, asmatrix, asarray
from scipy.sparse import issparse, csr_matrix as sparse
import torch
from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix as sparse
import time

def dTSI_dVdP(Surrogate, Pg, Qg, Pl, Ql, st_args):

    if Surrogate['model_type'] == "CNN":
        return dTSI_dVdP_CNN(Surrogate, Pg, Qg, Pl, Ql, st_args)
    elif Surrogate['model_type'] == "UQ_CNN":
        return dTSI_dVdP_UQ_CNN(Surrogate, Pg, Qg, Pl, Ql, st_args)
    elif Surrogate['model_type'] == "CNF":
        return dTSI_dVdP_CNF(Surrogate, Pg, Qg, Pl, Ql, st_args)
    elif Surrogate['model_type'] == "DSPP":
        return dTSI_dVdP_GP(Surrogate, Pg, Qg, Pl, Ql, st_args)

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
    c = F_u0 - (1.0 - alpha)
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
    
    c = constraint_value(x_param, model, scaler, u0, alpha, x_space, device, dtype)
    (g,) = torch.autograd.grad(c, x_param, create_graph=False, retain_graph=False)
    # x = x_param.detach().clone().requires_grad_(True)
    # c = constraint_value(x, model, scaler, u0, alpha, x_space, device, dtype)
    # (g,) = torch.autograd.grad(c, x, create_graph=False, retain_graph=False)
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


    pg = torch.tensor(Pg, dtype=torch.float32, requires_grad=True)
    qg = torch.tensor(Qg, dtype=torch.float32, requires_grad=True)
    pl = torch.tensor(Pl, dtype=torch.float32, requires_grad=False)
    ql = torch.tensor(Ql, dtype=torch.float32, requires_grad=False)

    X = torch.cat([pg, pl, qg, ql], dim=0)   # (N,)


    # # Concatenate generators first, then loads (same as training)
    # P_concat = np.concatenate([Pg, Pl], axis=0)  # (Ngen+Nload,)
    # Q_concat = np.concatenate([Qg, Ql], axis=0)  # (Ngen+Nload,)

    # # Per-sample layout: (2, Nunits)
    # x_test_np = np.stack([P_concat, Q_concat], axis=0)  # (2, Nunits)
    # x_test_np = x_test_np.reshape(-1)
    # X = torch.tensor(x_test_np, dtype=dtype)

    g = constraint_grad(X, model, scaler, u0, alpha, x_space, device, dtype)

    return g.detach().cpu().numpy()

# derivative of f(s) > tau
def dTSI_dVdP_CNN(CNNmodel, Pg, Qg, Pl, Ql, st_args):

    Mul_confi = st_args['Mul_confi']
    model = CNNmodel['model']

    # --- convert to torch ---
    pg = torch.tensor(Pg, dtype=torch.float32, requires_grad=True)
    pl = torch.tensor(Pl, dtype=torch.float32, requires_grad=False)
    ql = torch.tensor(Qg, dtype=torch.float32, requires_grad=False)

    # --- reconstruct X in CNN input shape ---
    X = torch.cat([pg, pl, ql], dim=0)   # (N,)
    X = X.unsqueeze(0).unsqueeze(0)      # (1, 1, N) adjust to your CNN

    # --- forward ---
    model.eval()
    y = model(X)
    y_scalar = y.sum()                   # convert the tensor into a scalar

    # --- backward ---
    y_scalar.backward()

    # gradient wrt pg only
    grad_pg = pg.grad

    # --- convert gradient to numpy array ---
    dTSI = grad_pg.detach().cpu().numpy()

    return dTSI

# f = μ − β sqrt(var)
def dTSI_dVdP_UQ_CNN(UQCNNmodel, Pg, Qg, Pl, Ql, st_args):

    Mul_confi = st_args['Mul_confi']
    model = UQCNNmodel['model']

    def f_scalar(pg_vector):
        pl_t = torch.tensor(Pl, dtype=torch.float32, requires_grad=False)
        ql_t = torch.tensor(Ql, dtype=torch.float32, requires_grad=False)
        
        # --- reconstruct X in CNN input shape ---
        X = torch.cat([pg, pl_t, ql_t], dim=0)   # (N,)
        X = X.unsqueeze(0).unsqueeze(0)      # (1, 1, N) adjust to your CNN

        mean_pred, var_pred = model(X)

        # f = μ − β sqrt(var)
        f = mean_pred - Mul_confi * torch.sqrt(var_pred + 1e-8)
        return f.squeeze()       # MUST BE SCALAR

    pg = torch.tensor(Pg, dtype=torch.float32, requires_grad=True)

    grad_f_pg = torch.autograd.grad(f_scalar(pg), pg, create_graph=True)[0]
    dTSI = grad_f_pg.detach().cpu().numpy()

    return dTSI

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
