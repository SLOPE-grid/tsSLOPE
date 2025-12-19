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
    elif Surrogate['model_type'] == "CNN_silu":
        return TSI_constraint_CNN(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "CNN_Grad_UQ":
        return TSI_constraint_CNN_Grad_UQ(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "CNN_silu_no_sig":
        return TSI_constraint_CNN(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "UQ_CNN":
        return TSI_constraint_UQ_CNN(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "UQ_CNN_STD":
        return TSI_constraint_UQ_CNN_STD(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "UQ_CNN_SiLU":
        return TSI_constraint_UQ_CNN(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "UQ_CNN_SiLU_no_sig":
        return TSI_constraint_UQ_CNN(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "CNF":
        return TSI_constraint_CNF(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "DSPP":
        return TSI_constraint_GP(Surrogate, Pg, Qg, st_args)

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

def TSI_constraint_CNF(Surrogate, Pg, Qg, st_args):
    model = Surrogate['model']
    scaler = Surrogate['scaler'] 
    ckpt = Surrogate['ckpt'] 
    dtype = Surrogate['dtype'] 
    u0 = Surrogate['u0'] 
    alpha = Surrogate['alpha'] 
    device = Surrogate['device'] 
    x_space = Surrogate['x_space'] 

    PL = st_args['PL']
    QL = st_args['QL']

    Pg_input = Pg
    Qg_input = Qg
    Pl_input = PL
    Ql_input = QL

    # Concatenate generators first, then loads (same as training)
    P_concat = np.concatenate([Pg_input, Pl_input], axis=0)  # (Ngen+Nload,)
    Q_concat = np.concatenate([Qg_input, Ql_input], axis=0)  # (Ngen+Nload,)

    # Per-sample layout: (2, Nunits)
    x_test_np = np.stack([P_concat, Q_concat], axis=0)  # (2, Nunits)
    x_test_np = x_test_np.reshape(-1)
    X = torch.tensor(x_test_np, dtype=dtype)

    c_val = constraint_value(X, model, scaler, u0, alpha, x_space, device, dtype)

    return c_val.item()

def TSI_constraint_CNN(Surrogate, Pg, Qg, st_args):
    model = Surrogate['model']
    dtype = Surrogate['dtype'] 

    model.eval()

    PL = st_args['PL']
    QL = st_args['QL']

    Pg_input = Pg.reshape(1, -1)
    Qg_input = Qg.reshape(1, -1)
    Pl_input = PL.reshape(1, -1)
    Ql_input = QL.reshape(1, -1)

    X_np = np.hstack([Pg_input, Pl_input, Ql_input])

    X = torch.tensor(X_np, dtype=dtype).unsqueeze(0)

    pred = model(X).item()

    return pred


def TSI_constraint_CNN_Grad_UQ(Surrogate, Pg, Qg, st_args):
    model = Surrogate['model']
    dtype = Surrogate['dtype'] 
    beta = st_args['beta'] 

    PL = st_args['PL']
    QL = st_args['QL']

    # build full input vector as one differentiable tensor
    pg = torch.tensor(Pg, dtype=dtype)
    pl = torch.tensor(PL, dtype=dtype)
    ql = torch.tensor(QL, dtype=dtype)    

    X = torch.cat([pg, pl, ql], dim=0).requires_grad_(True)

    X_in = X.view(1, 1, -1)

    model.eval()
    y = model(X_in).sum()
    y.backward()

    # gradient wrt pg are the first len(pg) components
    grad_pg = X.grad[:len(pg)].clone()

    pred = y - beta * torch.dot(grad_pg, grad_pg)
    return pred.item()

# f = μ − β sqrt(var)
def TSI_constraint_UQ_CNN(Surrogate, Pg, Qg, st_args):
    Mul_confi = st_args['Mul_confi']
    model = Surrogate['model']
    dtype = Surrogate['dtype'] 

    model.eval()

    PL = st_args['PL']
    QL = st_args['QL']

    Pg_input = Pg.reshape(1, -1)
    Qg_input = Qg.reshape(1, -1)
    Pl_input = PL.reshape(1, -1)
    Ql_input = QL.reshape(1, -1)

    X_np = np.hstack([Pg_input, Pl_input, Ql_input])

    X = torch.tensor(X_np, dtype=dtype).unsqueeze(0)

    mean_pred, var_pred = model(X)

    f = mean_pred - Mul_confi * torch.sqrt(var_pred + 1e-8)

    return f.squeeze().item()

# f = μ − β sqrt(var)
def TSI_constraint_UQ_CNN_STD(Surrogate, Pg, Qg, st_args):
    Mul_confi = st_args['Mul_confi']
    model = Surrogate['model']
    dtype = Surrogate['dtype'] 

    model.eval()

    PL = st_args['PL']
    QL = st_args['QL']

    Pg_input = Pg.reshape(1, -1)
    Qg_input = Qg.reshape(1, -1)
    Pl_input = PL.reshape(1, -1)
    Ql_input = QL.reshape(1, -1)

    X_np = np.hstack([Pg_input, Pl_input, Ql_input])

    X = torch.tensor(X_np, dtype=dtype).unsqueeze(0)

    mean_pred, STD_pred = model(X)
    std_ref = STD_pred/(mean_pred*(1-mean_pred) + 1e-8)
    print(f"Mean: {mean_pred.item()}, STD: {std_ref.item()}")

    f = mean_pred - Mul_confi * std_ref

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
