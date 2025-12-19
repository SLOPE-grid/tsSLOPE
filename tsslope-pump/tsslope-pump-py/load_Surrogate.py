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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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

class CNN1D_GELU_Avg_UQ(nn.Module):
    def __init__(self):
        super(CNN1D_GELU_Avg_UQ, self).__init__()

        # Shared feature extractor (same as your original but without final layers)
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # output shape: [batch, 64, 1]
            nn.Flatten(),             # shape: [batch, 64]
        )

        # Two output heads: mean and log-variance
        self.fc_mean = nn.Linear(64, 1)
        self.fc_logvar = nn.Linear(64, 1)

    def forward(self, x):
        h = self.features(x)

        # Mean should be in [0,1] (classification probability)
        mean = torch.sigmoid(self.fc_mean(h))

        # log-variance → variance > 0
        logvar = self.fc_logvar(h)
        var = torch.exp(logvar)

        return mean, var

class UQ_CNN_SiLU(nn.Module):
    def __init__(self):
        super(UQ_CNN_SiLU, self).__init__()

        # Shared feature extractor (same as your original but without final layers)
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),  # output shape: [batch, 64, 1]
            nn.Flatten(),             # shape: [batch, 64]
        )

        # Two output heads: mean and log-variance
        self.fc_mean = nn.Linear(64, 1)
        self.fc_logvar = nn.Linear(64, 1)

    def forward(self, x):
        h = self.features(x)

        # Mean should be in [0,1] (classification probability)
        mean = torch.sigmoid(self.fc_mean(h))

        # log-variance → variance > 0
        logvar = self.fc_logvar(h)
        var = torch.exp(logvar)

        return mean, var

class UQ_CNN_SiLU_No_Sig(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
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
        )

        # Logit mean and log-variance
        self.fc_mu = nn.Linear(64, 1)
        self.fc_logvar = nn.Linear(64, 1)

    def forward(self, x):
        h = self.features(x)
        mu = self.fc_mu(h)              # logit mean
        logvar = self.fc_logvar(h)
        var = torch.exp(logvar)         # logit variance
        return mu, var

class UQ_CNN_SiLU_std(nn.Module):
    """
    CNN surrogate with uncertainty.

    Outputs:
      mean(x) ∈ (0,1)  → probability of instability
      std(x)  > 0      → predictive standard deviation
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
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
        )

        self.fc_mean   = nn.Linear(64, 1)
        self.fc_logstd = nn.Linear(64, 1)

        # Good initial uncertainty scale
        # nn.init.zeros_(self.fc_logstd.weight)
        # nn.init.constant_(self.fc_logstd.bias, -1.5)  # std ≈ 0.22

    def forward(self, x):
        h = self.features(x)

        mean = torch.sigmoid(self.fc_mean(h))

        log_std = self.fc_logstd(h)
        std = torch.exp(log_std)

        return mean, std

class CNN1D_GELU_Avg(nn.Module):
    def __init__(self):
        super(CNN1D_GELU_Avg, self).__init__()
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

class CNN1D_silu_No_sig(nn.Module):
    def __init__(self):
        super().__init__()
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

            nn.Linear(64, 1)  # <-- logit output
        )

    def forward(self, x):
        return self.net(x)

class CNN1D_silu(nn.Module):
    def __init__(self):
        super(CNN1D_silu, self).__init__()
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


def load_surrogate(Model_Path, data_record, model_type):
    if model_type == "CNN":
        return load_CNNmodel(Model_Path, data_record, model_type)
    elif model_type == "UQ_CNN":
        return load_UQ_CNNmodel(Model_Path, data_record, model_type)
    elif model_type == "UQ_CNN_STD":
        return load_UQ_CNN_STDmodel(Model_Path, data_record, model_type)
    elif model_type == "UQ_CNN_SiLU":
        return load_UQ_CNN_SiLU_model(Model_Path, data_record, model_type)
    elif model_type == "UQ_CNN_SiLU_no_sig":
        return load_UQ_CNN_SiLU_no_sig_model(Model_Path, data_record, model_type)
    elif model_type == "CNN_Grad_UQ":
        return load_CNN_silu_model(Model_Path, data_record, model_type)
    elif model_type == "CNN_silu":
        return load_CNN_silu_model(Model_Path, data_record, model_type)
    elif model_type == "CNN_silu_no_sig":
        return load_CNN_silu_no_sig_model(Model_Path, data_record, model_type)
    elif model_type == "CNF":
        return load_CNFmodel(Model_Path, data_record, model_type)
    elif model_type == "DSPP":
        return load_GPmodel(Model_Path, data_record, model_type)
    else:
        raise ValueError("Incorrect model_type")

     
def load_CNFmodel(
    ckpt_path: str,
    data_record:str,
    model_type,
    device: torch.device = torch.device("cpu"),
    override_dtype: Optional[str] = "float64",
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

    return Surrogate, data, TSI

def load_CNNmodel(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64",):
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

    model = CNN1D_GELU_Avg().double()
    
    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type

    return Surrogate, data, TSI

def load_CNN_silu_model(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64",):
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

    model = CNN1D_silu().double()
    
    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type

    return Surrogate, data, TSI

def load_CNN_silu_no_sig_model(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64",):
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

    model = CNN1D_silu_No_sig().double()
    
    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type

    return Surrogate, data, TSI

def load_UQ_CNNmodel(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64",):
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

    model = CNN1D_GELU_Avg_UQ().double()

    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type

    return Surrogate, data, TSI


def load_UQ_CNN_STDmodel(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64",):
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

    model = UQ_CNN_SiLU_std().double()

    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type

    return Surrogate, data, TSI

def load_UQ_CNN_SiLU_model(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64",):
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

    model = UQ_CNN_SiLU().double()

    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type

    return Surrogate, data, TSI

def load_UQ_CNN_SiLU_no_sig_model(Model_Path, data_record, model_type,
    override_dtype: Optional[str] = "float64",):
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

    model = UQ_CNN_SiLU_No_Sig().double()

    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    Surrogate = {}
    Surrogate['model'] = model
    Surrogate['dtype'] = dtype
    Surrogate['model_type'] = model_type

    return Surrogate, data, TSI

def load_GPmodel(Model_Path, data_record, model_type):
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

    data = data.numpy()
    TSI = TSI.numpy()
    TSI = TSI.min(1)

    return GPmodel, data, TSI
