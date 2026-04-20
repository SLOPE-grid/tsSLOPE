import glob
import re
import gpytorch
import torch
import torch.nn as nn
import os
import tqdm
import urllib.request
import warnings
import numpy as np
import time
from math import floor
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from matplotlib import pyplot as plt
from gpytorch.mlls import DeepPredictiveLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy, BatchDecoupledVariationalStrategy
from gpytorch.variational import MeanFieldVariationalDistribution
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
import gpytorch.settings as settings
import scipy.io as scio

from typing import Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


# =========================
# Model definition (DKL)
# =========================

def _num_key(p: str) -> int:
    m = re.search(r"TSI_batch_(\d+)\.mat$", os.path.basename(p))
    return int(m.group(1)) if m else 10**9

class LargeFeatureExtractor(nn.Sequential):
    def __init__(self, data_dim: int):
        super().__init__()
        self.add_module("linear1", nn.Linear(data_dim, 400))
        self.add_module("relu1", nn.ReLU())
        self.add_module("linear2", nn.Linear(400, 80))
        self.add_module("relu2", nn.ReLU())
        self.add_module("linear3", nn.Linear(80, 2))  # feature_dim=2

class GPRegressionModel(gpytorch.models.ExactGP):
    """
    DKL (feature extractor + KISS-GP / GridInterpolationKernel)
    """
    def __init__(self, train_x, train_y, likelihood, data_dim: int, grid_size: int = 100):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            num_dims=2,
            grid_size=grid_size,
        )
        self.feature_extractor = LargeFeatureExtractor(data_dim)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# =========================
# Utilities (from plot-DKL.py)
# =========================

def load_bundle(bundle_path: str, device: torch.device):
    try:
        bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    except TypeError:
        bundle = torch.load(bundle_path, map_location=device)

    if not isinstance(bundle, dict):
        raise RuntimeError("no bundle .pt")
    return bundle

def normalize_x_fullsample(x_raw: torch.Tensor, X_min: torch.Tensor, X_max: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    X_max = torch.clamp(X_max, min=eps)
    x_shift = x_raw - X_min
    return 2.0 * (x_shift / X_max) - 1.0

def rebuild_all_data_from_disk(bundle: dict, device: torch.device, dir_root: str):
    data_dir = os.path.join(dir_root, "batches")
    tsi_names = bundle["tsi_names"]

    driver_dir = os.path.join(dir_root, tsi_names[0])
    driver_files = glob.glob(os.path.join(driver_dir, "TSI_batch_*.mat"))
    driver_files.sort(key=_num_key)
    if not driver_files:
        raise RuntimeError(f"batch：{driver_dir} no TSI_batch_*.mat")

    delete_idx = bundle["delete_idx"].astype(np.int64)
    min_valid_tsi = float(bundle["min_valid_tsi"])

    X_min = torch.tensor(bundle["X_min"], dtype=torch.float32, device=device)
    X_max = torch.tensor(bundle["X_max"], dtype=torch.float32, device=device)
    y_mean = float(bundle["y_mean"])
    y_std = float(bundle["y_std"])

    data_list = []
    data_list2 = []
    y_list = []

    for driver_fp in driver_files:
        idx = _num_key(driver_fp)

        data_fp = os.path.join(data_dir, f"samples_batch_{idx:03d}.mat")
        if not os.path.exists(data_fp):
            continue

        data_res = scio.loadmat(data_fp)
        if not all(k in data_res for k in ["p_rew_sample", "p_syn_sample", "p_load_sample"]):
            continue

        Data_raw = np.hstack([data_res["p_rew_sample"], data_res["p_syn_sample"], data_res["p_load_sample"]]).astype(np.float32)
        Data_raw2 = np.hstack([data_res["p_rew_sample"], data_res["p_syn_sample"], data_res["p_load_sample"], data_res["q_load_sample"]]).astype(np.float32)

        Data_raw = np.delete(Data_raw, delete_idx, axis=1)

        tsi_cols = []
        for name in tsi_names:
            tsi_fp = os.path.join(dir_root, name, f"TSI_batch_{idx:03d}.mat")
            tsi_res = scio.loadmat(tsi_fp)
            tsi_vec = tsi_res["TSI"].min(1).reshape(-1, 1).astype(np.float32)
            tsi_cols.append(tsi_vec)

        min_n = min([Data_raw.shape[0]] + [t.shape[0] for t in tsi_cols])
        Data_raw = Data_raw[:min_n, :]
        TSI_4 = np.hstack([t[:min_n, :] for t in tsi_cols])

        y_raw = np.min(TSI_4, axis=1).astype(np.float32)
        y_raw = np.where(y_raw == -100, min_valid_tsi, y_raw)

        data_list.append(Data_raw)
        data_list2.append(Data_raw2)
        y_list.append(y_raw)

    if not data_list:
        raise RuntimeError("重建数据失败：未读取到任何 batch。请检查 dir_root 数据路径。")

    X_raw_all = np.vstack(data_list)
    X_raw_all2 = np.vstack(data_list2)
    y_raw_all = np.concatenate(y_list)

    mpc = scio.loadmat('Texas7k_20210804.mat')
    busnum_to_idx = {int(busnum): idx for idx, busnum in enumerate(mpc["bus"][:, 0])}
    data = np.load("Texas7k_20210804.npz", allow_pickle=True)
    load = data["load"]
    load[:, 0] = np.vectorize(busnum_to_idx.get)(load[:, 0]).astype(int)
    load_bus_idx = load[:, 0].astype(int)
    unique_bus, inv = np.unique(load_bus_idx, return_inverse=True)
    nbus_load = len(unique_bus)

    nw = data_res["p_rew_sample"].shape[1]
    ng = data_res["p_syn_sample"].shape[1]
    nd = data_res["p_load_sample"].shape[1]

    P_rew_all = X_raw_all[:, :nw]
    P_syn_all = X_raw_all[:, nw:nw+ng]
    P_load_all = X_raw_all[:, nw+ng:]
    Q_load_all = X_raw_all2[:, nw+ng+nd:]

    P_load_merged = np.zeros((P_load_all.shape[0], nbus_load), dtype=P_load_all.dtype)
    Q_load_merged = np.zeros((Q_load_all.shape[0], nbus_load), dtype=Q_load_all.dtype)
    for j in range(nd):
        P_load_merged[:, inv[j]] += P_load_all[:, j]
        Q_load_merged[:, inv[j]] += Q_load_all[:, j]

    X_raw_all = np.hstack([P_rew_all, P_syn_all, P_load_merged])
    X_raw_all2 = np.hstack([P_rew_all, P_syn_all, P_load_merged, Q_load_merged])

    X_raw_all_t = torch.tensor(X_raw_all, dtype=torch.float32, device=device)
    X_all_norm = normalize_x_fullsample(X_raw_all_t, X_min, X_max)

    y_all_norm = (torch.tensor(y_raw_all, dtype=torch.float32, device=device) - y_mean) / y_std

    shuffled_indices = torch.tensor(bundle["shuffled_indices"].astype(np.int64), device=device)
    X_all_norm = X_all_norm.index_select(dim=0, index=shuffled_indices)
    y_all_norm = y_all_norm.index_select(dim=0, index=shuffled_indices)

    return X_all_norm, y_all_norm, X_raw_all2, y_raw_all

# =========================
# Model definition (CNF)
# =========================

class StandardScalerTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean_", torch.tensor(0.0))
        self.register_buffer("std_", torch.tensor(1.0))
        self.fitted = False

    def fit(self, x: torch.Tensor):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False, keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.mean_ = mean.detach()
        self.std_ = std.detach()
        self.fitted = True
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            return x
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            return x
        return x * self.std_ + self.mean_

def leggauss_01(n: int, device=None, dtype: Optional[torch.dtype] = None):
    """
    Gauss-Legendre nodes and weights on [0, 1].
    """
    from numpy.polynomial.legendre import leggauss

    x, w = leggauss(n)           # nodes in [-1, 1]
    s = 0.5 * (x + 1.0)          # map to [0, 1]
    ws = 0.5 * w                 # scale weights
    if dtype is None:
        dtype = torch.get_default_dtype()
    s_t = torch.tensor(s, dtype=dtype, device=device)
    w_t = torch.tensor(ws, dtype=dtype, device=device)
    return s_t, w_t

class PhiNet(nn.Module):
    """
    Positive function Psi(s, x).
    Input is concat([s], x).
    """
    def __init__(self, x_dim: int, hidden: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        in_dim = 1 + x_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([s, x], dim=-1)
        out = self.net(z)
        out = torch.clamp(out, -20.0, 20.0)
        return torch.nn.functional.softplus(out) + 1e-6

class ConditionalCDF(nn.Module):
    """
    Normalized CDF F(y|x) on [0, 1] from integrating Psi(s, x).
    The raw variable y is assumed to lie in [-100, 100] (TSI range).
    """
    def __init__(self, x_dim: int, hidden=(128, 128), n_quad: int = 32, device=None):
        super().__init__()
        self.phi = PhiNet(x_dim, hidden)
        self.device = device if device is not None else torch.device("cpu")
        s_nodes, w = leggauss_01(n_quad, device=self.device, dtype=torch.get_default_dtype())
        self.register_buffer("quad_nodes", s_nodes)    # (K,)
        self.register_buffer("quad_weights", w)        # (K,)

    # scaling between raw [-100, 100] and unit [0, 1]
    @staticmethod
    def to_unit(y_raw: torch.Tensor) -> torch.Tensor:
        return torch.clamp((y_raw + 100.0) / 200.0, 1e-6, 1.0 - 1e-6)

    @staticmethod
    def from_unit(y01: torch.Tensor) -> torch.Tensor:
        return 200.0 * y01 - 100.0

    def _phi_int_0_1(self, x: torch.Tensor) -> torch.Tensor:
        """
        Integral of Psi(s, x) over s in [0, 1] using Gauss-Legendre quadrature.
        """
        B, D = x.shape
        K = self.quad_nodes.size(0)
        s = self.quad_nodes.view(1, K, 1).expand(B, K, 1)      # (B, K, 1)
        xrep = x.view(B, 1, D).expand(B, K, D)                 # (B, K, D)
        phi_vals = self.phi(s, xrep).squeeze(-1)               # (B, K)
        return torch.sum(self.quad_weights * phi_vals, dim=1)  # (B,)

    def _phi_int_0_y(self, y01: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Integral of Psi(s, x) over s in [0, y] in unit coordinates.
        """
        y01 = y01.view(-1, 1)
        B, D = x.shape
        K = self.quad_nodes.size(0)
        z = self.quad_nodes.view(1, K, 1).expand(B, K, 1)      # (B, K, 1)
        s = y01.view(B, 1, 1) * z                              # (B, K, 1)
        xrep = x.view(B, 1, D).expand(B, K, D)                 # (B, K, D)
        phi_vals = self.phi(s, xrep).squeeze(-1)               # (B, K)
        H = y01.view(B) * torch.sum(self.quad_weights * phi_vals, dim=1)  # (B,)
        return H

    # public APIs on [0, 1]
    def cdf01(self, y01: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Z = self._phi_int_0_1(x) + 1e-8
        H = self._phi_int_0_y(y01, x)
        return torch.clamp(H / Z, 0.0, 1.0).view(-1, 1)

    def pdf01(self, y01: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Z = self._phi_int_0_1(x) + 1e-8
        phi_y = self.phi(y01.view(-1, 1), x).view(-1) + 1e-12
        return (phi_y / Z).view(-1, 1)

    def nll01(self, y01: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Z = self._phi_int_0_1(x) + 1e-8
        phi_y = self.phi(y01.view(-1, 1), x).view(-1) + 1e-12
        return (-(torch.log(phi_y) - torch.log(Z))).view(-1, 1)

    # APIs on raw [-100, 100]
    def pdf(self, y_raw: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y01 = self.to_unit(y_raw)
        f01 = self.pdf01(y01, x)
        return f01 / 200.0  # change-of-variables

    def cdf(self, y_raw: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.cdf01(self.to_unit(y_raw), x)

    def quantile01(self, u: torch.Tensor, x: torch.Tensor, tol: float = 1e-6, max_iter: int = 64) -> torch.Tensor:
        """
        Invert CDF on the unit interval using bisection.
        """
        u = torch.clamp(u.view(-1, 1), 1e-6, 1.0 - 1e-6)
        B = x.size(0)
        lo = torch.zeros(B, 1, device=x.device)
        hi = torch.ones(B, 1, device=x.device)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            Fmid = self.cdf01(mid, x)
            go_hi = (Fmid <= u)
            lo = torch.where(go_hi, mid, lo)
            hi = torch.where(go_hi, hi, mid)
            if torch.max(hi - lo).item() < tol:
                break
        return 0.5 * (lo + hi)

    def quantile(self, u: torch.Tensor, x: torch.Tensor, tol: float = 1e-6, max_iter: int = 64) -> torch.Tensor:
        return self.from_unit(self.quantile01(u, x, tol, max_iter))

    @torch.no_grad()
    def sample(self, num: int, x: torch.Tensor) -> torch.Tensor:
        """
        Draw num samples for each x.
        """
        B = x.size(0)
        u = torch.rand(B * num, 1, device=x.device)
        x_rep = x.repeat_interleave(num, dim=0)
        y = self.quantile(u, x_rep)
        return y.view(B, num, 1)

    def quantile_with_grad(self, u: torch.Tensor, x: torch.Tensor):
        """
        Return (q(u|x) on raw scale, dq/dx) via implicit differentiation.
        """
        x = x.requires_grad_(True)
        q01 = self.quantile01(u, x).detach()                 # stop grad through root-find
        F_q = self.cdf01(q01, x)                             # (B, 1)
        f_q = self.pdf01(q01, x).detach()                    # (B, 1)
        grads = []
        for b in range(x.size(0)):                           # per-sample gradient of F wrt x at fixed y=q
            grad_b = torch.autograd.grad(F_q[b:b+1].sum(), x, retain_graph=True, create_graph=False)[0][b]
            grads.append(grad_b)
        dFdx = torch.stack(grads, dim=0)                     # (B, D)
        dQ01dx = - dFdx / (f_q + 1e-12)
        q_raw = self.from_unit(q01)
        dQ_raw_dx = 200.0 * dQ01dx
        return q_raw.detach(), dQ_raw_dx.detach()

# =========================
# Model definition (DSPP)
# =========================

class DSPPHiddenLayer(DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=300, inducing_points=None, mean_type='constant', Q=8):
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # The inducing points were passed in, but the shape doesn't match the number of GPs in this layer.
            # Let's assume we wanted to use the same inducing point initialization for each GP in the layer,
            # and expand the inducing points to match this.
            inducing_points = inducing_points.unsqueeze(0).expand((output_dims,) + inducing_points.shape)
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)
        if inducing_points is None:
            # No inducing points were specified, let's just initialize them randomly.
            if output_dims is None:
                # An output_dims of None implies there is only one GP in this layer
                # (e.g., the last layer for univariate regression).
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Get the number of inducing points from the ones passed in.
            num_inducing = inducing_points.size(-2)

        # Let's use mean field / diagonal covariance structure.
        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        # Standard variational inference.
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])

        super(DSPPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims, Q)

        if mean_type == 'constant':
            # We'll use a constant mean for the final output layer.
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            # As in Salimbeni et al. 2017, we find that using a linear mean for the hidden layer improves performance.
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)

        self.covar_module = ScaleKernel(MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)

    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class TwoLayerDSPP(DSPP):
    def __init__(self, train_x_shape, inducing_points, num_inducing, hidden_dim=3, Q=3):
        hidden_layer = DSPPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=hidden_dim,
            mean_type='linear',
            inducing_points=inducing_points,
            Q=Q,
        )
        last_layer = DSPPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
            inducing_points=None,
            num_inducing=num_inducing,
            Q=Q,
        )

        likelihood = GaussianLikelihood()

        super().__init__(Q)
        self.likelihood = likelihood
        self.last_layer = last_layer
        self.hidden_layer = hidden_layer

    def forward(self, inputs, **kwargs):
        hidden_rep1 = self.hidden_layer(inputs, **kwargs)
        output = self.last_layer(hidden_rep1, **kwargs)
        return output

    def predict(self, loader):
        with settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            mus, variances, lls = [], [], []
            for x_batch, y_batch in loader:
                preds = self.likelihood(self(x_batch, mean_input=x_batch))
                mus.append(preds.mean.cpu())
                variances.append(preds.variance.cpu())

                # Compute test log probability. The output of a DSPP is a weighted mixture of Q Gaussians,
                # with the Q weights specified by self.quad_weight_grid. The below code computes the log probability of each
                # test point under this mixture.

                # Step 1: Get log marginal for each Gaussian in the output mixture.
                base_batch_ll = self.likelihood.log_marginal(y_batch, self(x_batch))

                # Step 2: Weight each log marginal by its quadrature weight in log space.
                deep_batch_ll = self.quad_weights.unsqueeze(-1) + base_batch_ll

                # Step 3: Take logsumexp over the mixture dimension, getting test log prob for each datapoint in the batch.
                batch_log_prob = deep_batch_ll.logsumexp(dim=0)
                lls.append(batch_log_prob.cpu())

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

# =========================
# Model definition (CNNs)
# =========================

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class CNN1D_SiLU(nn.Module):
    def __init__(self):
        super(CNN1D_SiLU, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
class CNN1D_GELU(nn.Module):
    def __init__(self):
        super(CNN1D_GELU, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
class CNN1D_Sigmoid(nn.Module):
    def __init__(self):
        super(CNN1D_Sigmoid, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.AvgPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
class CNN1D_Tanh(nn.Module):
    def __init__(self):
        super(CNN1D_Tanh, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
class CNN1D_Softplus(nn.Module):
    def __init__(self):
        super(CNN1D_Softplus, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.AvgPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
   


def load_surrogate(Model_Path, data_record, model_type, active_gen_only = True):
    
    if model_type == "CNN":
        return load_CNNmodel(Model_Path, data_record, model_type, active_gen_only = active_gen_only)
    elif model_type == "CNN_SiLU":
        return load_CNNmodel_SiLU(Model_Path, data_record, model_type, active_gen_only = active_gen_only)
    elif model_type == "CNN_GELU":
        return load_CNNmodel_GELU(Model_Path, data_record, model_type, active_gen_only = active_gen_only)
    elif model_type == "CNN_Sig":
        return load_CNNmodel_Sig(Model_Path, data_record, model_type, active_gen_only = active_gen_only)
    elif model_type == "CNN_Tanh":
        return load_CNNmodel_Tanh(Model_Path, data_record, model_type, active_gen_only = active_gen_only)
    elif model_type == "CNN_Soft":
        return load_CNNmodel_Soft(Model_Path, data_record, model_type, active_gen_only = active_gen_only)
    elif model_type == "CNN_Grad_UQ":
        return load_CNNmodel(Model_Path, data_record, model_type, active_gen_only = active_gen_only)
    elif model_type == "CNF":
        return load_CNFmodel(Model_Path, data_record, model_type, active_gen_only = active_gen_only)
    elif model_type == "DSPP":
        return load_GPmodel(Model_Path, data_record, model_type, active_gen_only = active_gen_only)
    else:
        print("No model_type was chosen. Code will run without a surrogate")

        data = scio.loadmat(data_record)
        data = data['Data']
        TSI = data[:, -1].reshape(-1, 1)
        TSI = (TSI >= 0).astype(int)

        data = data[:, :-1]

        Surrogate = {"model_type": None}

        return Surrogate, data, TSI

     
def load_CNFmodel(
    ckpt_path: str,
    data_record:str,
    model_type,
    device: torch.device = torch.device("cpu"),
    override_dtype: Optional[str] = "float64",
    active_gen_only = True
):
    warnings.filterwarnings("ignore")
    data = scio.loadmat(data_record)
    data = data['Data']

    TSI = data[:, -1].reshape(-1, 1)

    data = data[:, :-1]
    
    """
    Load a trained ConditionalCDF model and its StandardScalerTorch from a checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # infer dtype from checkpoint unless overridden
    def _first_tensor_dtype(d):
        if isinstance(d, dict):
            for v in d.values():
                if torch.is_tensor(v):
                    return v.dtype
                if isinstance(v, dict):
                    dt = _first_tensor_dtype(v)
                    if dt is not None:
                        return dt
        return None

    state_like = ckpt.get("state_dict", ckpt.get("model_state", {}))
    ckpt_dtype = _first_tensor_dtype(state_like) or torch.float32

    if override_dtype is not None:
        override_dtype = override_dtype.lower()
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(override_dtype, ckpt_dtype)

    torch.set_default_dtype(dtype)

    x_dim = int(ckpt["x_dim"])
    hidden = tuple(ckpt["hidden"])
    n_quad = int(ckpt["n_quad"])

    model = ConditionalCDF(x_dim=x_dim, hidden=hidden, n_quad=n_quad, device=device).to(dtype)
    model.load_state_dict(ckpt["state_dict"])

    model = model.to(dtype).to(device).eval()

    scaler = StandardScalerTorch()
    scaler.mean_ = ckpt["scaler_mean"].to(device=device, dtype=dtype)
    scaler.std_ = ckpt["scaler_std"].to(device=device, dtype=dtype)
    scaler.fitted = True

    u0 = 0.0     # threshold in raw TSI space [-100, 100]
    alpha = 0.95
    x_space = "raw" 

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['scaler'] = scaler
    Surrogate['ckpt'] = ckpt
    Surrogate['dtype'] = dtype
    Surrogate['u0'] = u0
    Surrogate['alpha'] = alpha
    Surrogate['device'] = device
    Surrogate['x_space'] = x_space
    Surrogate['model_type'] = model_type
    Surrogate['active_gen_only'] = active_gen_only

    return Surrogate, data, TSI

def load_CNNmodel(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64", active_gen_only = True):
    warnings.filterwarnings("ignore")
    data = scio.loadmat(data_record)
    data = data['Data']

    if override_dtype is not None:
        override_dtype = override_dtype.lower()
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(override_dtype)

    # Binary target: last column >= 0 → class 1, else 0
    TSI = data[:, -1].reshape(-1, 1)
    TSI = (TSI >= 0).astype(int)

    data = data[:, :-1]

    model = CNN1D().double()
    
    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type
    Surrogate['active_gen_only'] = active_gen_only

    return Surrogate, data, TSI

def load_CNNmodel_SiLU(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64", active_gen_only = True):
    warnings.filterwarnings("ignore")
    data = scio.loadmat(data_record)
    data = data['Data']

    if override_dtype is not None:
        override_dtype = override_dtype.lower()
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(override_dtype)

    # Binary target: last column >= 0 → class 1, else 0
    TSI = data[:, -1].reshape(-1, 1)
    TSI = (TSI >= 0).astype(int)

    data = data[:, :-1]

    model = CNN1D_SiLU().double()
    # model = CNN1D_SiLU()
    
    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type
    Surrogate['active_gen_only'] = active_gen_only

    return Surrogate, data, TSI

def load_CNNmodel_GELU(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64", active_gen_only = True):
    warnings.filterwarnings("ignore")
    data = scio.loadmat(data_record)
    data = data['Data']

    if override_dtype is not None:
        override_dtype = override_dtype.lower()
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(override_dtype)

    # Binary target: last column >= 0 → class 1, else 0
    TSI = data[:, -1].reshape(-1, 1)
    TSI = (TSI >= 0).astype(int)

    data = data[:, :-1]

    model = CNN1D_GELU().double()
    
    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type
    Surrogate['active_gen_only'] = active_gen_only

    return Surrogate, data, TSI

def load_CNNmodel_Sig(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64", active_gen_only = True):
    warnings.filterwarnings("ignore")
    data = scio.loadmat(data_record)
    data = data['Data']

    if override_dtype is not None:
        override_dtype = override_dtype.lower()
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(override_dtype)

    # Binary target: last column >= 0 → class 1, else 0
    TSI = data[:, -1].reshape(-1, 1)
    TSI = (TSI >= 0).astype(int)

    data = data[:, :-1]

    model = CNN1D_Sigmoid().double()
    
    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type
    Surrogate['active_gen_only'] = active_gen_only

    return Surrogate, data, TSI

def load_CNNmodel_Tanh(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64", active_gen_only = True):
    warnings.filterwarnings("ignore")
    data = scio.loadmat(data_record)
    data = data['Data']

    if override_dtype is not None:
        override_dtype = override_dtype.lower()
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(override_dtype)

    # Binary target: last column >= 0 → class 1, else 0
    TSI = data[:, -1].reshape(-1, 1)
    TSI = (TSI >= 0).astype(int)

    data = data[:, :-1]

    model = CNN1D_Tanh().double()
    
    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type
    Surrogate['active_gen_only'] = active_gen_only

    return Surrogate, data, TSI

def load_CNNmodel_Soft(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64", active_gen_only = True):
    warnings.filterwarnings("ignore")
    data = scio.loadmat(data_record)
    data = data['Data']

    if override_dtype is not None:
        override_dtype = override_dtype.lower()
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(override_dtype)

    # Binary target: last column >= 0 → class 1, else 0
    TSI = data[:, -1].reshape(-1, 1)
    TSI = (TSI >= 0).astype(int)

    data = data[:, :-1]

    model = CNN1D_Softplus().double()
    
    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type
    Surrogate['active_gen_only'] = active_gen_only

    return Surrogate, data, TSI

def load_GPmodel(Model_Path, data_record, model_type,  active_gen_only = True):
    batch_size = 500  # Size of minibatch
    milestones = [20, 150, 300]  # Epochs at which we will lower the learning rate by a factor of 0.1
    num_inducing_pts = 300  # Number of inducing points in each hidden layer 97.8, 8.0
    num_epochs = 400  # Number of epochs to train for
    initial_lr = 0.01  # Initial learning rate
    hidden_dim = 6  # Number of GPs (i.e., the width) in the hidden layer. 6:97.7, 8.8
    num_quadrature_sites = 8  # Number of quadrature sites (see paper for a description of this. 5-10 generally works well). S

    PATHcwd = os.getcwd()
    warnings.filterwarnings("ignore")
    data = scio.loadmat(data_record)
    data = data['Data']
    TSI = data[:, -1].reshape(-1, 1)
    TSI = torch.from_numpy(TSI).float()
    data = torch.from_numpy(data[:, :-1]).float()

    X = data.detach().clone()
    X_min = X.min(0)[0]
    X = X - X_min
    X_max = X.max(0)[0]
    X = 2.0 * (X / X_max) - 1.0
    y = TSI.detach().clone()
    y = y.min(1)[0]
    y_mean = y.mean()
    y -= y.mean()
    y_std = y.std()
    y /= y.std()

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.8 * X.size(0)))

    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()
    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()

    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

   # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # Use k-means to initialize inducing points (only helpful for the first layer)
    inducing_points = (train_x[torch.randperm(min(1000 * 100, train_n))[0:num_inducing_pts], :])
    inducing_points = inducing_points.clone().data.cpu().numpy()
    inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(), inducing_points, minit='matrix')[0])

    if torch.cuda.is_available():
        inducing_points = inducing_points.cuda()

    model = TwoLayerDSPP(
        train_x.shape,
        inducing_points,
        num_inducing=num_inducing_pts,
        hidden_dim=hidden_dim,
        Q=num_quadrature_sites
    )

    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    GPmodel = {}
    GPmodel['model'] = model
    GPmodel['X_max'] = X_max / 100
    GPmodel['X_min'] = X_min / 100
    GPmodel['y_mean'] = y_mean
    GPmodel['y_std'] = y_std
    GPmodel['model_type'] = model_type
    GPmodel['active_gen_only'] = active_gen_only

    data = data.numpy()
    TSI = TSI.numpy()
    TSI = TSI.min(1)

    return GPmodel, data, TSI

# =========================
# Main API you call: load_DKLmodel
# =========================

def load_DKLmodel(Model_Path: str, data_record, model_type, active_gen_only = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load bundle
    if not os.path.exists(Model_Path):
        raise FileNotFoundError(f"Model_Path no exist：{Model_Path}")
    bundle = load_bundle(Model_Path, device=device)

    # 3) rebuild normalized dataset
    X_all_norm, y_all_norm, X_raw_all, y_raw_all = rebuild_all_data_from_disk(bundle, device=device, dir_root=dir_root)

    train_n = int(bundle["train_n"])
    train_x = X_all_norm[:train_n, :].contiguous()
    train_y = y_all_norm[:train_n].contiguous()

    # 4) build model + load weights (DKL)
    data_dim = int(bundle["data_dim_after_delete"])
    grid_size = int(bundle.get("grid_size", 100))

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPRegressionModel(train_x, train_y, likelihood, data_dim=data_dim, grid_size=grid_size).to(device)

    model.load_state_dict(bundle["model_state_dict"])
    likelihood.load_state_dict(bundle["likelihood_state_dict"])
    model.eval()
    likelihood.eval()

    # 6) assemble DKLmodel dict
    DKLmodel = {
        "model": model,
        "likelihood": likelihood,
        "device": device,
        "X_min": np.array(bundle["X_min"], dtype=np.float32),
        "X_max": np.array(bundle["X_max"], dtype=np.float32),
        "y_mean": float(bundle["y_mean"]),
        "y_std": float(bundle["y_std"]),
        "model_type": model_type
    }

    return DKLmodel, X_raw_all, y_raw_all