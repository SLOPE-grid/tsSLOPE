"""Computes partial derivatives of TSI w.r.t. X.
"""
import numpy as np
from numpy import conj, arange, diag, zeros, asmatrix, asarray
from scipy.sparse import issparse, csr_matrix as sparse
import torch
from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix as sparse
import time

def TSI_constraint(Surrogate, Pg, Qg, st_args):
    # print(f"In TSI\n")
    # print(f"Length of pg: {len(Pg)}\n")
    # print(f"Computing TSI constraint for model type: {Surrogate['model_type']} ...") 
    if Surrogate['model_type'] == "CNF":
        return TSI_constraint_CNF(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "DKL":
        return TSI_constraint_DKL(Surrogate, Pg, Qg, st_args)
    elif Surrogate['model_type'] == "CNN_UQ":
        return TSI_constraint_CNN_UQ(Surrogate, Pg, Qg, st_args)
    else:        
        if st_args['reorder_pg']:
            return TSI_constraint_CNN_Reorder_pg(Surrogate, Pg, Qg, st_args)
        else:
            return TSI_constraint_CNN(Surrogate, Pg, Qg, st_args)

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
    # print(f"Total time to calculate model.cdf: {total_time}")
    alpha = .9 # Remove this later
    c = (1.0 - alpha) - F_u0
    print(f"F_u0={F_u0}, TSI = (1.0 - {alpha})  - F_u0 = {c}")
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

    gen_idx = st_args['gen_idx']

    PL = st_args['PL']
    QL = st_args['QL']

    Pg_input = Pg[gen_idx]
    Qg_input = Qg[gen_idx]
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

def TSI_constraint_CNN(surrogate, Pg, Qg, st_args):
    model = surrogate["model"]
    dtype = surrogate.get("dtype", next(model.parameters()).dtype)
    device = surrogate.get("device", next(model.parameters()).device)
    active_gen_only = surrogate.get("active_gen_only", True)
    use_half_columns = surrogate.get("use_half_columns", False)

    if active_gen_only:
        gen_idx = st_args["gen_idx"]
        pg = torch.as_tensor(Pg[gen_idx], dtype=dtype, device=device).reshape(-1)
        if not use_half_columns:
            print(f"Using full columns for Qg in TSI_constraint_CNN.")
            qg = torch.as_tensor(Qg[gen_idx], dtype=dtype, device=device).reshape(-1)
    else:
        pg = torch.as_tensor(Pg, dtype=dtype, device=device).reshape(-1)
        if not use_half_columns:
            qg = torch.as_tensor(Qg, dtype=dtype, device=device).reshape(-1)

    pl = torch.as_tensor(st_args["PL"], dtype=dtype, device=device).reshape(-1)
    if not use_half_columns:
        ql = torch.as_tensor(st_args["QL"], dtype=dtype, device=device).reshape(-1)

    model.eval()
    with torch.no_grad():
        if use_half_columns:
            x_raw = torch.cat([pg, pl], dim=0)
        else:
            x_raw = torch.cat([pg, pl, qg, ql], dim=0)

        if surrogate.get("normalize_inputs", False):
            print("Normalizing inputs for TSI_constraint_CNN.")
            x_mean = surrogate.get("X_mean")
            x_std = surrogate.get("X_std")
            if x_mean is None or x_std is None:
                raise RuntimeError(
                    "Surrogate says normalize_inputs=True, but X_mean/X_std are missing."
                )
            x_mean = x_mean.to(dtype=dtype, device=device).reshape(-1)
            x_std = x_std.to(dtype=dtype, device=device).reshape(-1)
            if x_raw.numel() != x_mean.numel() or x_raw.numel() != x_std.numel():
                raise ValueError(
                    f"Input length {x_raw.numel()} does not match normalization stats "
                    f"({x_mean.numel()} means, {x_std.numel()} stds)."
                )
            x_raw = (x_raw - x_mean) / x_std

        probability = model(x_raw.view(1, 1, -1)).sum()

    return probability.item()

def TSI_constraint_CNN_Reorder_pg(Surrogate, Pg, Qg, st_args):
    model = Surrogate['model']
    dtype = Surrogate['dtype'] 
    active_gen_only = Surrogate['active_gen_only']
    gen_idx = st_args['gen_idx']
    syn_idx = st_args['syn_idx']
    rew_idx = st_args['rew_idx']

    model.eval()

    PL = st_args['PL']
    QL = st_args['QL']

    # print(f"PL = {PL.tolist()}")

    if active_gen_only:
        syn_idx_use = gen_idx[syn_idx]
        rew_idx_use = gen_idx[rew_idx]
    else:
        rew_idx_use = rew_idx
        syn_idx_use = np.setdiff1d(np.arange(len(Pg)), rew_idx_use)

    # print(f"Pg_active_syn_idx = {Pg[syn_idx_use].tolist()}")
    # print(f"Pg_active_rew_idx = {Pg[rew_idx_use].tolist()}")

    Pg_input = np.hstack([
        Pg[rew_idx_use].reshape(1, -1),
        Pg[syn_idx_use].reshape(1, -1)
    ])

    Pl_input = PL.reshape(1, -1)
    Ql_input = QL.reshape(1, -1)

    X_np = np.hstack([Pg_input, Pl_input])

    X = torch.tensor(X_np, dtype=dtype).unsqueeze(0)

    X_scaled = X * 100

    pred = model(X_scaled).item()

    return pred

def TSI_constraint_CNN_Grad_UQ(Surrogate, Pg, Qg, st_args):
    model = Surrogate['model']
    dtype = Surrogate['dtype'] 
    active_gen_only = Surrogate['active_gen_only']
    gen_idx = st_args['gen_idx']
    beta = st_args['beta'] 

    PL = st_args['PL']
    QL = st_args['QL']

    # build full input vector as one differentiable tensor
    if active_gen_only:
        pg = torch.tensor(Pg[gen_idx], dtype=dtype)
    else:
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

def TSI_constraint_CNN_UQ(surrogate, Pg, Qg, st_args):
    model = surrogate["model"]
    dtype = surrogate.get("dtype", next(model.parameters()).dtype)
    device = surrogate.get("device", next(model.parameters()).device)
    active_gen_only = surrogate.get("active_gen_only", True)
    use_half_columns = surrogate.get("use_half_columns", False)
    Mul_confi = st_args['Mul_confi']    

    if active_gen_only:
        gen_idx = st_args["gen_idx"]
        pg = torch.as_tensor(Pg[gen_idx], dtype=dtype, device=device).reshape(-1)
        if not use_half_columns:
            # print(f"Using full columns for Qg in TSI_constraint_CNN_UQ.")
            qg = torch.as_tensor(Qg[gen_idx], dtype=dtype, device=device).reshape(-1)
    else:
        pg = torch.as_tensor(Pg, dtype=dtype, device=device).reshape(-1)
        if not use_half_columns:
            qg = torch.as_tensor(Qg, dtype=dtype, device=device).reshape(-1)

    pl = torch.as_tensor(st_args["PL"], dtype=dtype, device=device).reshape(-1)
    if not use_half_columns:
        ql = torch.as_tensor(st_args["QL"], dtype=dtype, device=device).reshape(-1)

    model.eval()
    with torch.no_grad():
        if use_half_columns:
            x_raw = torch.cat([pg, pl], dim=0)
        else:
            x_raw = torch.cat([pg, pl, qg, ql], dim=0)

        if surrogate.get("normalize_inputs", False):
            # print("Normalizing inputs for TSI_constraint_CNN_UQ.")
            x_mean = surrogate.get("X_mean")
            x_std = surrogate.get("X_std")
            if x_mean is None or x_std is None:
                raise RuntimeError(
                    "Surrogate says normalize_inputs=True, but X_mean/X_std are missing."
                )
            x_mean = x_mean.to(dtype=dtype, device=device).reshape(-1)
            x_std = x_std.to(dtype=dtype, device=device).reshape(-1)
            if x_raw.numel() != x_mean.numel() or x_raw.numel() != x_std.numel():
                raise ValueError(
                    f"Input length {x_raw.numel()} does not match normalization stats "
                    f"({x_mean.numel()} means, {x_std.numel()} stds)."
                )
            x_raw = (x_raw - x_mean) / x_std

        mean_raw, var_raw = model(x_raw.view(1, 1, -1))
        mean = mean_raw.sum()
        std = torch.sqrt(var_raw.sum())
        if surrogate.get("normalize_targets", False):
            y_mean = surrogate.get("y_mean")
            y_std = surrogate.get("y_std")
            if y_mean is None or y_std is None:
                raise RuntimeError(
                    "Surrogate says normalize_targets=True, but y_mean/y_std are missing."
                )
            mean = y_mean + y_std * mean
            std = y_std * std
        constraint = mean - Mul_confi * std 

    print("Mean: ", mean, "STD: ", std )
    print("TSI UQ: ", constraint)

    return constraint.item()

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

def TSI_constraint_DKL(GPmodel, Pg, Qg, st_args):
    Mul_confi, gen_idx, syn_idx, rew_idx = st_args['Mul_confi'], st_args['gen_idx'], st_args['syn_idx'], st_args['rew_idx']
    active_gen_only = GPmodel['active_gen_only']  
    ng0 = len(gen_idx)

    model = GPmodel['model']
    likelihood = GPmodel['likelihood']
    X_max = GPmodel['X_max']
    X_min = GPmodel['X_min']
    y_mean = GPmodel['y_mean']
    y_std = GPmodel['y_std']

    model.eval()
    likelihood.eval()

    Pl = st_args['PL']
    Ql = st_args['QL']

    if active_gen_only:
        # active_syn_idx = list(set(syn_idx) & set(gen_idx))
        # active_rew_idx = list(set(rew_idx) & set(gen_idx))
        active_syn_idx = gen_idx[syn_idx]
        active_rew_idx = gen_idx[rew_idx]
        Pg_active_syn_idx = Pg[active_syn_idx].reshape(1, -1)
        Pg_active_rew_idx = Pg[active_rew_idx].reshape(1, -1)
        # Qg_active_syn_idx = Qg[active_syn_idx].reshape(1, -1)
        # Qg_active_rew_idx = Qg[active_rew_idx].reshape(1, -1)

        # print(f"Length of active_syn_idx: {len(active_syn_idx)}")
        # print(f"Length of active_rew_idx: {len(active_rew_idx)}")

        Pg_input = np.hstack([Pg_active_rew_idx, Pg_active_syn_idx])
        # Qg_input = np.hstack([Qg_active_rew_idx, Qg_active_syn_idx])
    else:
        Pg_syn_idx = Pg[syn_idx].reshape(1, -1)
        Pg_rew_idx = Pg[rew_idx].reshape(1, -1)
        Qg_syn_idx = Qg[syn_idx].reshape(1, -1)
        Qg_rew_idx = Qg[rew_idx].reshape(1, -1)

        Pg_input = np.hstack([Pg_rew_idx, Pg_syn_idx])
        # Qg_input = np.hstack([Qg_rew_idx, Qg_syn_idx])

    X = np.hstack([Pg_input, Pl.reshape(1, -1)])
    X = torch.tensor(X, dtype=torch.float64) * 100
    # X = torch.tensor(X, dtype=torch.float64) 

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        X = X.cuda()
        X_max = torch.tensor(X_max, dtype=torch.float64).cuda()
        X_min = torch.tensor(X_min, dtype=torch.float64).cuda()
    else:
        X_max = torch.tensor(X_max, dtype=torch.float64)
        X_min = torch.tensor(X_min, dtype=torch.float64)

    X = X - X_min
    X = 2.0 * (X / X_max) - 1.0
    X = torch.clamp(X, -1.0, 1.0)

    GPpre = likelihood(model(X))
    TSI_mean = GPpre.mean * y_std + y_mean
    TSI_mean = TSI_mean.cpu().detach().numpy()
    TSI_std = GPpre.stddev * y_std
    TSI_std = TSI_std.cpu().detach().numpy()

    TSI_interval_half = Mul_confi * TSI_std
    constraint = TSI_interval_half - TSI_mean  # means TSI_interval_half - TSI_mean < 0 

    return constraint.item()


