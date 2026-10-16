"""
Conditional CDF Flow module for inference and chance-constraint evaluation.

Basic functions:
    - load_model_and_scaler
    - constraint_value
    - constraint_value_grad
    - constraint_hessian

Helper functions:
    - StandardScalerTorch
    - PhiNet
    - ConditionalCDF
    - x_to_std

The model represents P(Y <= y | X = x) where Y (TSI) is defined on the raw interval [-100, 100].
Internal computations use a unit interval [0, 1] via an affine scaling.

Data structure (for X and Y used during training and inference):

    X : array_like, shape (N, 2, Nunits)
        Input scenarios with active/reactive power for generators and loads.
        First Ng units: generators with variables [p_g, q_g]
        Last  Nl units: loads with variables [p_l, q_l]

    Y : array_like, shape (N, F, Z)
        Output samples for the TSI quantity of interest.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .architecture import (
        ArchitectureSpec,
        DEFAULT_DENSITY_FLOOR,
        validate_density_floor,
        architecture_from_checkpoint,
        build_hidden_layers,
        make_architecture_spec,
    )
    from .checkpoint_variants import DEFAULT_MODEL_VARIANT, resolve_checkpoint_state_dict
except ImportError:  # pragma: no cover - direct script execution fallback
    from architecture import (
        ArchitectureSpec,
        DEFAULT_DENSITY_FLOOR,
        validate_density_floor,
        architecture_from_checkpoint,
        build_hidden_layers,
        make_architecture_spec,
    )
    from checkpoint_variants import DEFAULT_MODEL_VARIANT, resolve_checkpoint_state_dict

try:
    from torch.func import grad as torch_func_grad, jacrev as torch_func_jacrev
except ImportError:  # pragma: no cover - older PyTorch fallback
    torch_func_grad = None
    torch_func_jacrev = None

INFERENCE_DTYPE = torch.float64
TEST_EXAMPLES_DIR = Path(__file__).resolve().parent / "test_examples"
DEFAULT_CKPT_PATH = TEST_EXAMPLES_DIR / "cdf_flow_checkpoint.pt"
DEFAULT_DATA_PATH = TEST_EXAMPLES_DIR / "tsi_probml_fullinputs_500_min.npz"
HESSIAN_JACREV_CHUNK_SIZE = 32


@dataclass(slots=True)
class _PhiStaticCache:
    first_weight_s: torch.Tensor
    first_weight_x: torch.Tensor
    first_bias: torch.Tensor
    hidden_weights: tuple[torch.Tensor, ...]
    hidden_biases: tuple[torch.Tensor, ...]
    final_weight: torch.Tensor
    final_bias: torch.Tensor
    quad_nodes: torch.Tensor
    quad_weights: torch.Tensor
    density_floor: float
    state_signature: Optional[tuple] = None


@dataclass(slots=True)
class ConstraintEvalCache:
    mode: str
    x_std: torch.Tensor
    scale_vec: torch.Tensor
    y0: torch.Tensor
    y01: torch.Tensor
    alpha: float
    model_cache: Optional[_PhiStaticCache] = None
    first_layer_base: Optional[torch.Tensor] = None
    joint_nodes: Optional[torch.Tensor] = None
    x_param: Optional[torch.Tensor] = None
    model: Optional["ConditionalCDF"] = None
    scaler: Optional["StandardScalerTorch"] = None
    x_space: str = "raw"
    device: Optional[torch.device] = None
    dtype: torch.dtype = INFERENCE_DTYPE
    order: int = -1
    z: Optional[torch.Tensor] = None
    h: Optional[torch.Tensor] = None
    grad_z_std: Optional[torch.Tensor] = None
    grad_h_std: Optional[torch.Tensor] = None
    hess_z_std: Optional[torch.Tensor] = None
    hess_h_std: Optional[torch.Tensor] = None
    constraint_value_batch: Optional[torch.Tensor] = None
    constraint_grad_batch: Optional[torch.Tensor] = None
    constraint_hessian_batch: Optional[torch.Tensor] = None
    state_signature: Optional[tuple] = None


def _require_float64_tensor(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.dtype != INFERENCE_DTYPE:
        raise TypeError(f"{name} must have dtype torch.float64, got {x.dtype}.")
    return x


def _require_float64_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype != INFERENCE_DTYPE:
        raise TypeError(f"inference dtype must be torch.float64, got {dtype}.")
    return dtype


def _state_signature(model: nn.Module, scaler=None) -> Optional[tuple]:
    tensors = tuple(model.parameters()) + (model.quad_nodes, model.quad_weights)
    if scaler is not None:
        tensors += (scaler.mean_, scaler.std_)
    # Inference tensors have no mutation counter, so they cannot certify a cache.
    if any(torch.is_inference(tensor) for tensor in tensors):
        return None
    return (
        tuple(type(layer) for layer in model.phi.net),
        tuple((id(tensor), tensor._version, tensor.device, tensor.dtype) for tensor in tensors),
        model.phi.density_floor,
    )


def _try_build_phi_static_cache(model: nn.Module) -> Optional[_PhiStaticCache]:
    phi_net = getattr(getattr(model, "phi", None), "net", None)
    if not isinstance(phi_net, nn.Sequential):
        return None

    modules = list(phi_net)
    if not modules or not isinstance(modules[-1], nn.Linear):
        return None
    for idx, module in enumerate(modules[:-1]):
        if idx % 2 == 0 and not isinstance(module, nn.Linear):
            return None
        if idx % 2 == 1 and not isinstance(module, nn.ReLU):
            return None

    linears = [module for module in modules if isinstance(module, nn.Linear)]
    if len(linears) < 2:
        return None

    first = linears[0]
    final = linears[-1]
    middle = linears[1:-1]

    return _PhiStaticCache(
        first_weight_s=first.weight[:, 0].detach().contiguous(),
        first_weight_x=first.weight[:, 1:].detach().contiguous(),
        first_bias=first.bias.detach().contiguous(),
        hidden_weights=tuple(layer.weight.detach().contiguous() for layer in middle),
        hidden_biases=tuple(layer.bias.detach().contiguous() for layer in middle),
        final_weight=final.weight.detach().view(-1).contiguous(),
        final_bias=final.bias.detach().view(()),
        quad_nodes=model.quad_nodes.detach().contiguous(),
        quad_weights=model.quad_weights.detach().contiguous(),
        density_floor=validate_density_floor(model.phi.density_floor),
    )


def _get_phi_static_cache(model: nn.Module) -> Optional[_PhiStaticCache]:
    cache = getattr(model, "_hpc_static_cache", None)
    signature = _state_signature(model)
    if cache is not None and signature is not None and cache.state_signature == signature:
        return cache
    cache = _try_build_phi_static_cache(model)
    if cache is not None:
        cache.state_signature = signature
    setattr(model, "_hpc_static_cache", cache)
    return cache


def _cache_mode_for_model(model: nn.Module) -> str:
    return "relu_fast" if _get_phi_static_cache(model) is not None else "generic_autograd"


class StandardScalerTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean_", torch.tensor(0.0, dtype=INFERENCE_DTYPE))
        self.register_buffer("std_", torch.tensor(1.0, dtype=INFERENCE_DTYPE))
        self.fitted = False

    def fit(self, x: torch.Tensor):
        x = _require_float64_tensor("x", x)
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False, keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.mean_ = mean.detach()
        self.std_ = std.detach()
        self.fitted = True
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = _require_float64_tensor("x", x)
        if not self.fitted:
            return x
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        x = _require_float64_tensor("x", x)
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
        dtype = INFERENCE_DTYPE
    _require_float64_dtype(dtype)
    s_t = torch.tensor(s, dtype=dtype, device=device)
    w_t = torch.tensor(ws, dtype=dtype, device=device)
    return s_t, w_t


class PhiNet(nn.Module):
    """
    Positive function Psi(s, x).
    Input is concat([s], x).
    """
    def __init__(
        self,
        x_dim: int,
        hidden: Tuple[int, ...] = (128, 128),
        activation: str = "relu",
        activations: Sequence[str] | None = None,
        *,
        density_floor: float = DEFAULT_DENSITY_FLOOR,
    ):
        super().__init__()
        self.density_floor = validate_density_floor(density_floor)
        self.architecture = make_architecture_spec(hidden, activation=activation, activations=activations)
        self.hidden = self.architecture.hidden
        self.activations = self.architecture.activations
        layers = build_hidden_layers(1 + x_dim, self.hidden, activations=self.activations)
        in_dim = 1 + x_dim
        for h in self.hidden:
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([s, x], dim=-1)
        out = self.net(z)
        out = torch.clamp(out, -20.0, 20.0)
        return torch.nn.functional.softplus(out) + self.density_floor


class ConditionalCDF(nn.Module):
    """
    Normalized CDF F(y|x) on [0, 1] from integrating Psi(s, x).
    The raw variable y is assumed to lie in [-100, 100] (TSI range).
    """
    def __init__(
        self,
        x_dim: int,
        hidden=(128, 128),
        n_quad: int = 32,
        device=None,
        activation: str = "relu",
        activations: Sequence[str] | None = None,
        *,
        density_floor: float = DEFAULT_DENSITY_FLOOR,
    ):
        super().__init__()
        self.architecture: ArchitectureSpec = make_architecture_spec(
            hidden, activation=activation, activations=activations
        )
        self.hidden = self.architecture.hidden
        self.activations = self.architecture.activations
        self.phi = PhiNet(
            x_dim,
            self.hidden,
            activation=self.architecture.activation,
            activations=self.activations,
            density_floor=density_floor,
        ).to(dtype=INFERENCE_DTYPE)
        self.device = device if device is not None else torch.device("cpu")
        s_nodes, w = leggauss_01(n_quad, device=self.device, dtype=INFERENCE_DTYPE)
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

    def _cdf01_joint_integrals(self, y01: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the normalization and partial integral in one PhiNet pass.
        """
        y01 = y01.view(-1, 1)
        B, D = x.shape
        K = self.quad_nodes.size(0)
        z = self.quad_nodes.view(1, K, 1).expand(B, K, 1)      # (B, K, 1)
        s = torch.cat([z, y01.view(B, 1, 1) * z], dim=1)       # (B, 2K, 1)
        xrep = x.view(B, 1, D).expand(B, 2 * K, D)             # (B, 2K, D)
        phi_vals = self.phi(s, xrep).squeeze(-1)               # (B, 2K)
        phi_0_1 = phi_vals[:, :K]
        phi_0_y = phi_vals[:, K:]
        Z = torch.sum(self.quad_weights * phi_0_1, dim=1) + 1e-8
        H = y01.view(B) * torch.sum(self.quad_weights * phi_0_y, dim=1)
        return Z, H

    # public APIs on [0, 1]
    def cdf01(self, y01: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        _require_float64_tensor("y01", y01)
        _require_float64_tensor("x", x)
        Z = self._phi_int_0_1(x) + 1e-8
        H = self._phi_int_0_y(y01, x)
        return torch.clamp(H / Z, 0.0, 1.0).view(-1, 1)

    def pdf01(self, y01: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        _require_float64_tensor("y01", y01)
        _require_float64_tensor("x", x)
        Z = self._phi_int_0_1(x) + 1e-8
        phi_y = self.phi(y01.view(-1, 1), x).view(-1) + 1e-12
        return (phi_y / Z).view(-1, 1)

    def nll01(self, y01: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        _require_float64_tensor("y01", y01)
        _require_float64_tensor("x", x)
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
        _require_float64_tensor("u", u)
        _require_float64_tensor("x", x)
        u = torch.clamp(u.view(-1, 1), 1e-6, 1.0 - 1e-6)
        B = x.size(0)
        lo = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        hi = torch.ones(B, 1, device=x.device, dtype=x.dtype)
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
        _require_float64_tensor("x", x)
        B = x.size(0)
        u = torch.rand(B * num, 1, device=x.device, dtype=x.dtype)
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


# ----------------------------------------------------------------------
# Model loading and chance-constraint utilities
# ----------------------------------------------------------------------

@torch.inference_mode(False)
def load_model_and_scaler(
    ckpt_path: str,
    device: torch.device = torch.device("cpu"),
    override_dtype: Optional[str] = "float64",
    model_variant: str | None = None,
):
    """
    Load a trained ConditionalCDF model and its StandardScalerTorch from a checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    selected_state_dict, resolved_model_variant = resolve_checkpoint_state_dict(ckpt, model_variant)

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

    state_like = selected_state_dict
    ckpt_dtype = _first_tensor_dtype(state_like) or INFERENCE_DTYPE
    if ckpt_dtype != INFERENCE_DTYPE:
        raise TypeError(f"Checkpoint tensors must have dtype torch.float64, got {ckpt_dtype}.")

    if override_dtype is not None:
        override_dtype = override_dtype.lower()
    if override_dtype not in (None, "float64"):
        raise ValueError("CNF inference only supports float64.")
    dtype = INFERENCE_DTYPE

    torch.set_default_dtype(dtype)

    x_dim = int(ckpt["x_dim"])
    architecture = architecture_from_checkpoint(ckpt)
    n_quad = int(ckpt["n_quad"])

    model = ConditionalCDF(
        x_dim=x_dim,
        hidden=architecture.hidden,
        n_quad=n_quad,
        device=device,
        activation=architecture.activation,
        activations=architecture.activations,
        density_floor=ckpt.get("density_floor", DEFAULT_DENSITY_FLOOR),
    ).to(dtype)
    model.load_state_dict(selected_state_dict)
    model = model.to(dtype).to(device).eval()
    model.requires_grad_(False)
    _get_phi_static_cache(model)
    ckpt["loaded_model_variant"] = resolved_model_variant

    scaler = StandardScalerTorch()
    if ckpt["scaler_mean"].dtype != INFERENCE_DTYPE or ckpt["scaler_std"].dtype != INFERENCE_DTYPE:
        raise TypeError("Scaler tensors in the checkpoint must have dtype torch.float64.")
    scaler.mean_ = ckpt["scaler_mean"].to(device=device, dtype=dtype)
    scaler.std_ = ckpt["scaler_std"].to(device=device, dtype=dtype)
    scaler.fitted = True

    return model, scaler, ckpt, dtype


def x_to_std(x: torch.Tensor, scaler: StandardScalerTorch, x_space: str) -> torch.Tensor:
    """
    Convert x from raw to standardized space if needed.
    x_space: "raw" or "std".
    """
    x = _require_float64_tensor("x", x.view(-1))
    return _x_to_std_batch(x, scaler, x_space).view(-1)


def _x_to_std_batch(x: torch.Tensor, scaler: StandardScalerTorch, x_space: str) -> torch.Tensor:
    if x_space not in ("raw", "std"):
        raise ValueError(f"x_space must be 'raw' or 'std', got {x_space!r}.")
    x = _require_float64_tensor("x", x)
    if x.ndim == 1:
        x = x.view(1, -1)
    elif x.ndim != 2:
        raise ValueError(f"x must have shape (D,) or (B, D), got {tuple(x.shape)}.")
    if x_space == "std":
        return x
    mean = scaler.mean_.view(1, -1).to(x.device, dtype=INFERENCE_DTYPE)
    std = scaler.std_.view(1, -1).to(x.device, dtype=INFERENCE_DTYPE)
    return (x - mean) / std


def _parameter_scale_vector(
    scaler: StandardScalerTorch,
    x_space: str,
    device: torch.device,
) -> torch.Tensor:
    if x_space == "std":
        return torch.ones_like(scaler.std_.view(-1), device=device, dtype=INFERENCE_DTYPE)
    std = scaler.std_.view(-1).to(device=device, dtype=INFERENCE_DTYPE)
    return std.reciprocal()


def _constraint_value_single_generic(
    x_param: torch.Tensor,
    model: ConditionalCDF,
    scaler: StandardScalerTorch,
    u0: float,
    alpha: float,
    x_space: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_std = _x_to_std_batch(x_param, scaler, x_space)
    y0 = torch.tensor([u0], device=device, dtype=dtype)
    return model.cdf(y0, x_std).view(()) - (1.0 - alpha)


@torch.inference_mode(False)
@torch.enable_grad()
def _compute_generic_constraint_results(cache: ConstraintEvalCache, max_order: int) -> ConstraintEvalCache:
    if cache.order >= max_order:
        return cache
    if cache.x_param is None or cache.model is None or cache.scaler is None or cache.device is None:
        raise ValueError("Generic constraint cache is missing model evaluation context.")

    x_batch = _require_float64_tensor("x_param", cache.x_param)
    if x_batch.ndim == 1:
        x_batch = x_batch.view(1, -1)

    values: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    hessians: list[torch.Tensor] = []

    for row in x_batch:
        if max_order == 0:
            with torch.inference_mode():
                value = _constraint_value_single_generic(
                    row, cache.model, cache.scaler, cache.y0.item(), cache.alpha,
                    cache.x_space, cache.device, cache.dtype,
                )
            values.append(value.detach())
            continue

        def f(z: torch.Tensor) -> torch.Tensor:
            return _constraint_value_single_generic(
                z, cache.model, cache.scaler, cache.y0.item(), cache.alpha,
                cache.x_space, cache.device, cache.dtype,
            )

        z = row.detach().clone().requires_grad_(True)
        value = f(z)
        values.append(value.detach())

        if max_order >= 1:
            grad_create_graph = max_order >= 2 and torch_func_grad is None
            (grad,) = torch.autograd.grad(value, z, create_graph=grad_create_graph, retain_graph=grad_create_graph)
            grads.append(grad.detach())

        if max_order >= 2:
            z_h = row.detach().clone().requires_grad_(True)
            if torch_func_grad is not None and torch_func_jacrev is not None:
                chunk_size = min(HESSIAN_JACREV_CHUNK_SIZE, z_h.numel())
                hessian = torch_func_jacrev(torch_func_grad(f), chunk_size=chunk_size)(z_h)
            else:
                hessian = torch.autograd.functional.hessian(f, z_h, create_graph=False)
            hessians.append(hessian.detach())

    cache.constraint_value_batch = torch.stack(values, dim=0)
    if grads:
        cache.constraint_grad_batch = torch.stack(grads, dim=0)
    if hessians:
        cache.constraint_hessian_batch = torch.stack(hessians, dim=0)
    cache.order = max_order
    return cache


def _phi_from_pre_activation(a: torch.Tensor, density_floor: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.clamp(a, -20.0, 20.0)
    sig = torch.sigmoid(t)
    grad_mask = ((a >= -20.0) & (a <= 20.0)).to(a.dtype)
    hess_mask = ((a > -20.0) & (a < 20.0)).to(a.dtype)
    phi = F.softplus(t) + density_floor
    d1 = sig * grad_mask
    d2 = sig * (1.0 - sig) * hess_mask
    return phi, d1, d2


def _phi_pre_activation(
    cache: ConstraintEvalCache,
    s_batch: torch.Tensor,
    need_grad: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    model_cache = cache.model_cache
    if s_batch.ndim == 1:
        s_batch = s_batch.view(1, -1).expand(cache.x_std.shape[0], -1)
    elif s_batch.ndim != 2:
        raise ValueError(f"s_batch must have shape (K,) or (B, K), got {tuple(s_batch.shape)}.")

    batch_size, num_nodes = s_batch.shape
    pre1 = cache.first_layer_base[:, None, :] + s_batch[:, :, None] * model_cache.first_weight_s.view(1, 1, -1)
    hidden = F.relu(pre1).reshape(batch_size * num_nodes, -1)
    masks: list[torch.Tensor] = []
    if need_grad:
        masks.append((pre1 > 0.0).reshape(batch_size * num_nodes, -1))

    for weight, bias in zip(model_cache.hidden_weights, model_cache.hidden_biases):
        pre = F.linear(hidden, weight, bias)
        hidden = F.relu(pre)
        if need_grad:
            masks.append(pre > 0.0)

    pre_output = F.linear(hidden, model_cache.final_weight.view(1, -1), model_cache.final_bias.view(1)).view(batch_size, num_nodes)
    if not need_grad:
        return pre_output, None

    sensitivity = model_cache.final_weight.view(1, -1).expand(batch_size * num_nodes, -1)
    for weight, mask in zip(reversed(model_cache.hidden_weights), reversed(masks[1:])):
        sensitivity = (sensitivity * mask.to(sensitivity.dtype)) @ weight
    grad_a = (sensitivity * masks[0].to(sensitivity.dtype)) @ model_cache.first_weight_x
    return pre_output, grad_a.view(batch_size, num_nodes, -1)


def _accumulate_weighted_gram(grad_a: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    hessians = []
    for batch_idx in range(grad_a.shape[0]):
        g = grad_a[batch_idx]
        scaled = coeff[batch_idx].unsqueeze(1) * g
        hessians.append(g.transpose(0, 1) @ scaled)
    return torch.stack(hessians, dim=0)


def _ensure_cache_order(cache: ConstraintEvalCache, max_order: int) -> ConstraintEvalCache:
    if cache.mode != "relu_fast":
        return _compute_generic_constraint_results(cache, max_order)
    if cache.order >= max_order:
        return cache

    model_cache = cache.model_cache
    batch_size = cache.x_std.shape[0]
    num_quad = model_cache.quad_nodes.numel()
    joint_nodes = cache.joint_nodes.view(1, -1).expand(batch_size, -1)
    pre_output, grad_a = _phi_pre_activation(cache, joint_nodes, need_grad=max_order >= 1)
    phi, d1, d2 = _phi_from_pre_activation(pre_output, model_cache.density_floor)

    quad_weights = model_cache.quad_weights.view(1, num_quad)
    z_phi = phi[:, :num_quad]
    h_phi = phi[:, num_quad:]
    cache.z = torch.sum(quad_weights * z_phi, dim=1) + 1e-8
    cache.h = cache.y01 * torch.sum(quad_weights * h_phi, dim=1)

    if max_order >= 1:
        assert grad_a is not None
        grad_z_nodes = grad_a[:, :num_quad, :]
        grad_h_nodes = grad_a[:, num_quad:, :]
        z_grad_coeff = (quad_weights * d1[:, :num_quad]).unsqueeze(-1)
        h_grad_coeff = (cache.y01 * quad_weights * d1[:, num_quad:]).unsqueeze(-1)
        cache.grad_z_std = torch.sum(z_grad_coeff * grad_z_nodes, dim=1)
        cache.grad_h_std = torch.sum(h_grad_coeff * grad_h_nodes, dim=1)

        if max_order >= 2:
            z_hess_coeff = quad_weights * d2[:, :num_quad]
            h_hess_coeff = cache.y01 * quad_weights * d2[:, num_quad:]
            cache.hess_z_std = _accumulate_weighted_gram(grad_z_nodes, z_hess_coeff)
            cache.hess_h_std = _accumulate_weighted_gram(grad_h_nodes, h_hess_coeff)

    cache.order = max_order
    return cache


def _ensure_constraint_results(cache: ConstraintEvalCache, max_order: int) -> ConstraintEvalCache:
    signature = _state_signature(cache.model, cache.scaler)
    if signature is None or signature != cache.state_signature:
        raise ValueError("Model or scaler state changed; prepare a new constraint cache.")
    if cache.mode != "relu_fast":
        return _compute_generic_constraint_results(cache, max_order)
    cache = _ensure_cache_order(cache, max_order)
    if cache.constraint_value_batch is None:
        ratio = torch.clamp(cache.h / cache.z, 0.0, 1.0)
        cache.constraint_value_batch = ratio - (1.0 - cache.alpha)

    if max_order >= 1 and cache.constraint_grad_batch is None:
        ratio = cache.h / cache.z
        ratio_grad_mask = ((ratio >= 0.0) & (ratio <= 1.0)).to(cache.z.dtype)
        inv_z = cache.z.reciprocal()
        inv_z_sq = inv_z * inv_z
        grad_ratio_std = cache.grad_h_std * inv_z.unsqueeze(1) - (cache.h * inv_z_sq).unsqueeze(1) * cache.grad_z_std
        scale_vec = cache.scale_vec.view(1, -1)
        cache.constraint_grad_batch = grad_ratio_std * scale_vec * ratio_grad_mask.unsqueeze(1)

    if max_order >= 2 and cache.constraint_hessian_batch is None:
        ratio = cache.h / cache.z
        ratio_hess_mask = ((ratio > 0.0) & (ratio < 1.0)).to(cache.z.dtype).view(-1, 1, 1)
        inv_z = cache.z.reciprocal()
        inv_z_sq = inv_z * inv_z
        inv_z_cu = inv_z_sq * inv_z

        gh_gz = torch.einsum("bi,bj->bij", cache.grad_h_std, cache.grad_z_std)
        gz_gz = torch.einsum("bi,bj->bij", cache.grad_z_std, cache.grad_z_std)

        hess_ratio_std = (
            cache.hess_h_std * inv_z.view(-1, 1, 1)
            - (gh_gz + gh_gz.transpose(1, 2)) * inv_z_sq.view(-1, 1, 1)
            - cache.hess_z_std * (cache.h * inv_z_sq).view(-1, 1, 1)
            + gz_gz * (2.0 * cache.h * inv_z_cu).view(-1, 1, 1)
        )

        scale_outer = torch.einsum("i,j->ij", cache.scale_vec, cache.scale_vec).view(1, cache.scale_vec.numel(), cache.scale_vec.numel())
        cache.constraint_hessian_batch = hess_ratio_std * scale_outer * ratio_hess_mask

    return cache


@torch.inference_mode(False)
def prepare_constraint_cache(
    x_param: torch.Tensor,
    model: ConditionalCDF,
    scaler: StandardScalerTorch,
    u0: float,
    alpha: float,
    x_space: str,
    device: torch.device,
    dtype: torch.dtype,
    max_order: int = 0,
) -> ConstraintEvalCache:
    _require_float64_dtype(dtype)
    x_param = _require_float64_tensor("x_param", x_param).clone()
    x_std = _x_to_std_batch(x_param, scaler, x_space)
    model_cache = _get_phi_static_cache(model)
    mode = "relu_fast" if model_cache is not None else "generic_autograd"

    y0 = torch.tensor([u0], device=device, dtype=dtype)
    y01 = model.to_unit(y0).view(())
    cache = ConstraintEvalCache(
        mode=mode,
        x_std=x_std,
        scale_vec=_parameter_scale_vector(scaler, x_space, device=device),
        y0=y0,
        y01=y01,
        alpha=alpha,
        model_cache=model_cache,
        first_layer_base=(
            F.linear(x_std, model_cache.first_weight_x, model_cache.first_bias)
            if model_cache is not None
            else None
        ),
        joint_nodes=(
            torch.cat([model_cache.quad_nodes, y01 * model_cache.quad_nodes])
            if model_cache is not None
            else None
        ),
        x_param=x_param if x_param.ndim == 2 else x_param.view(1, -1),
        model=model,
        scaler=scaler,
        x_space=x_space,
        device=device,
        dtype=dtype,
        state_signature=_state_signature(model, scaler),
    )
    if max_order >= 0:
        _ensure_constraint_results(cache, max_order)
    return cache


def _validate_constraint_cache(cache, x_param, model, scaler, u0, alpha, x_space, device, dtype):
    x = _require_float64_tensor("x_param", x_param)
    if x.ndim == 1:
        x = x.view(1, -1)
    if (
        cache.model is not model
        or cache.scaler is not scaler
        or cache.x_space != x_space
        or cache.alpha != alpha
        or cache.y0.item() != u0
        or torch.device(cache.device) != torch.device(device)
        or cache.dtype != dtype
        or cache.x_param.device != x.device
        or not torch.equal(cache.x_param, x)
    ):
        raise ValueError("Constraint cache does not match the input or evaluation arguments; prepare a new cache.")


def constraint_value_from_cache(cache: ConstraintEvalCache) -> torch.Tensor:
    _ensure_constraint_results(cache, 0)
    return cache.constraint_value_batch


def constraint_value_grad_from_cache(cache: ConstraintEvalCache) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_constraint_results(cache, 1)
    return cache.constraint_value_batch, cache.constraint_grad_batch


def constraint_hessian_from_cache(cache: ConstraintEvalCache) -> torch.Tensor:
    _ensure_constraint_results(cache, 2)
    return cache.constraint_hessian_batch


# --- Constraint value: c(x) = F_Y(u0 | x) - (1 - alpha) ---

def constraint_value(
    x_param: torch.Tensor,
    model: ConditionalCDF,
    scaler: StandardScalerTorch,
    u0: float,
    alpha: float,
    x_space: str,
    device: torch.device,
    dtype: torch.dtype,
    cache: Optional[ConstraintEvalCache] = None,
) -> torch.Tensor:
    """
    Chance constraint:
        c(x) = P(Y <= u0 | x) - (1 - alpha)
             = F_Y(u0 | x) - (1 - alpha)

    We enforce c(x) <= 0.
    """
    _require_float64_dtype(dtype)
    if cache is None:
        model_cache = _get_phi_static_cache(model)
        if model_cache is None:
            x_std = x_to_std(x_param, scaler, x_space).view(1, -1)
            y0 = torch.tensor([u0], device=device, dtype=dtype)
            grad_required = torch.is_grad_enabled() and (
                x_std.requires_grad or any(p.requires_grad for p in model.parameters())
            )
            if grad_required:
                return model.cdf(y0, x_std).view(()) - (1.0 - alpha)
            with torch.inference_mode():
                y01 = model.to_unit(y0)
                Z, H = model._cdf01_joint_integrals(y01, x_std)
                return (torch.clamp(H / Z, 0.0, 1.0).view(()) - (1.0 - alpha))
        cache = prepare_constraint_cache(
            x_param, model, scaler, u0, alpha, x_space, device, dtype, max_order=0
        )
    else:
        _validate_constraint_cache(cache, x_param, model, scaler, u0, alpha, x_space, device, dtype)
        _ensure_constraint_results(cache, 0)
    return cache.constraint_value_batch[0]


@torch.inference_mode(False)
@torch.enable_grad()
def constraint_value_grad(
    x_param: torch.Tensor,
    model: ConditionalCDF,
    scaler: StandardScalerTorch,
    u0: float,
    alpha: float,
    x_space: str,
    device: torch.device,
    dtype: torch.dtype,
    cache: Optional[ConstraintEvalCache] = None,
):
    """
    Return constraint value and its gradient with respect to x_param.
    Input differentiation remains enabled inside no_grad/inference_mode callers.
    """
    _require_float64_dtype(dtype)
    if cache is None:
        model_cache = _get_phi_static_cache(model)
        if model_cache is None:
            x = _require_float64_tensor("x_param", x_param).detach().clone().requires_grad_(True)
            c = constraint_value(x, model, scaler, u0, alpha, x_space, device, dtype)
            (g,) = torch.autograd.grad(c, x, create_graph=False, retain_graph=False)
            return c.detach(), g
        cache = prepare_constraint_cache(
            x_param, model, scaler, u0, alpha, x_space, device, dtype, max_order=1
        )
    else:
        _validate_constraint_cache(cache, x_param, model, scaler, u0, alpha, x_space, device, dtype)
        _ensure_constraint_results(cache, 1)
    return cache.constraint_value_batch[0], cache.constraint_grad_batch[0]


@torch.inference_mode(False)
@torch.enable_grad()
def constraint_hessian(
    x_param: torch.Tensor,
    model: ConditionalCDF,
    scaler: StandardScalerTorch,
    u0: float,
    alpha: float,
    x_space: str,
    device: torch.device,
    dtype: torch.dtype,
    cache: Optional[ConstraintEvalCache] = None,
) -> torch.Tensor:
    """
    Full Hessian of the constraint c(x) with respect to x_param.
    Input differentiation remains enabled inside no_grad/inference_mode callers.
    """
    _require_float64_dtype(dtype)
    if cache is None:
        model_cache = _get_phi_static_cache(model)
        if model_cache is None:
            def f(z):
                return constraint_value(z, model, scaler, u0, alpha, x_space, device, dtype)

            z = _require_float64_tensor("x_param", x_param).detach().clone().requires_grad_(True)
            if torch_func_grad is not None and torch_func_jacrev is not None:
                chunk_size = min(HESSIAN_JACREV_CHUNK_SIZE, z.numel())
                return torch_func_jacrev(torch_func_grad(f), chunk_size=chunk_size)(z)
            return torch.autograd.functional.hessian(f, z, create_graph=False)
        cache = prepare_constraint_cache(
            x_param, model, scaler, u0, alpha, x_space, device, dtype, max_order=2
        )
    else:
        _validate_constraint_cache(cache, x_param, model, scaler, u0, alpha, x_space, device, dtype)
        _ensure_constraint_results(cache, 2)
    return cache.constraint_hessian_batch[0]


def constraint_numerical_diagnostics(
    x, model, scaler=None, u0=0.0, *, x_space="raw", include_gradients=False, batch_size=8,
):
    """Inspect the deployed quadrature, outside training/benchmark hot paths.

    With no scaler, inputs are already in model coordinates. Counts are
    unweighted node counts; integral/floor contributions use quadrature weights.
    Gradients are optional and use the existing FP64 deployment implementation.
    """
    if batch_size <= 0 or x_space not in ("raw", "std"):
        raise ValueError("Positive batch_size and x_space 'raw' or 'std' are required.")
    if not torch.isfinite(torch.as_tensor(u0)).all():
        raise ValueError("Diagnostic threshold must be finite.")
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    x = torch.as_tensor(x, device=device, dtype=dtype).detach()
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
        raise ValueError("Diagnostic inputs must be a nonempty finite (N, D) array.")
    if include_gradients and (dtype != torch.float64 or scaler is None):
        raise ValueError("Gradient diagnostics require FP64 and a deployment scaler.")
    report = {"status": "completed", "rows": [], "summary": {}, "warnings": [], "failures": [],
              "settings": {"threshold_raw": float(u0), "n_quad": model.quad_nodes.numel(),
                           "dtype": str(dtype), "clamp_limits": [-20., 20.], "near_limits": [-18., 18.],
                           "density_floor": model.phi.density_floor, "normalizer_regularizer": 1e-8,
                           "log_density_guard": 1e-12, "probability_log_clip": [1e-12, 1 - 1e-12],
                           "floor_warning_share": .1, "include_gradients": include_gradients,
                           "node_fractions": "unweighted", "excludes_performance_measurements": True}}
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size]
        try:
            with torch.no_grad():
                xs = (xb - scaler.mean_.to(xb)) / scaler.std_.to(xb) if scaler is not None and x_space == "raw" else xb
                nodes, weights = model.quad_nodes.to(xb), model.quad_weights.to(xb)
                y = model.to_unit(torch.full((len(xb),), float(u0), device=device, dtype=dtype))
                integrals, stats = {}, []
                for name, s in (("normalizer", nodes.expand(len(xb), -1)),
                                ("threshold", y[:, None] * nodes[None, :])):
                    a = model.phi.net(torch.cat((s[:, :, None], xs[:, None, :].expand(-1, len(nodes), -1)), -1)).squeeze(-1)
                    integrals[name] = (weights * F.softplus(a.clamp(-20, 20))).sum(1) * (y if name == "threshold" else 1)
                    for i, row in enumerate(a):
                        finite = torch.isfinite(row)
                        fields = {name + "_node_count": len(nodes),
                                  name + "_preclamp_min": float(row.min()) if finite.all() else None,
                                  name + "_preclamp_max": float(row.max()) if finite.all() else None,
                                  name + "_nonfinite_count": int((~finite).sum())}
                        for label, mask in (("below", row < -20), ("above", row > 20),
                                            ("near", (row < -18) | (row > 18))):
                            count = int((mask & finite).sum())
                            fields[name + "_" + label + "_count"] = count
                            fields[name + "_" + label + "_fraction"] = count / len(nodes)
                        if name == "normalizer":
                            stats.append(fields)
                        else:
                            stats[i].update(fields)
                z_floor = model.phi.density_floor * weights.sum()
                h_floor = y * z_floor
                z = integrals["normalizer"] + z_floor + 1e-8
                h = integrals["threshold"] + h_floor
                p = (h / z).clamp(0, 1)
                p_without = (integrals["threshold"] / (integrals["normalizer"] + 1e-8)).clamp(0, 1)
                values = {"probability": p, "density_floor_probability": h_floor / z,
                          "density_floor_threshold_share": h_floor / h,
                          "density_floor_normalizer_share": z_floor / z,
                          "normalizer_regularizer_share": 1e-8 / z,
                          "probability_without_density_floor": p_without,
                          "density_floor_probability_delta": p - p_without}
                for i, item in enumerate(stats):
                    item.update(row_position=start + i, gradient_norm_raw=None, gradient_norm_std=None, warnings=[])
                    for key, tensor in values.items():
                        item[key] = float(tensor[i]) if torch.isfinite(tensor[i]) else None
                    if any(item[name + "_nonfinite_count"] for name in ("normalizer", "threshold")) or any(item[k] is None for k in values):
                        item["warnings"].append("Nonfinite numerical diagnostic values.")
                    if any(item[name + "_below_count"] or item[name + "_above_count"] for name in ("normalizer", "threshold")):
                        item["warnings"].append("Pre-clamp outputs exceed [-20, 20].")
                    if any(item[name + "_near_count"] for name in ("normalizer", "threshold")):
                        item["warnings"].append("Pre-clamp outputs approach/exceed [-18, 18].")
                    if any(item[k] is not None and item[k] >= .1 for k in ("density_floor_threshold_share", "density_floor_normalizer_share")):
                        item["warnings"].append("Additive density floor supplies at least 10% of an integral.")
                    report["rows"].append(item)
        except Exception as exc:
            report["failures"].append({"row_positions": list(range(start, start + len(xb))), "stage": "values", "error": f"{type(exc).__name__}: {exc}"})
            continue
        if include_gradients:
            for i, item in enumerate(stats):
                try:
                    _, g = constraint_value_grad(xb[i], model, scaler, u0, .95, x_space, device, dtype)
                    std = scaler.std_.to(g)
                    raw, standardized = (g, g * std) if x_space == "raw" else (g / std, g)
                    if not torch.isfinite(raw).all() or not torch.isfinite(standardized).all():
                        raise ValueError("Nonfinite input gradient.")
                    for key, gradient in (("gradient_norm_raw", raw), ("gradient_norm_std", standardized)):
                        norm = torch.linalg.vector_norm(gradient)
                        if not torch.isfinite(norm):
                            raise ValueError("Nonfinite input-gradient norm.")
                        item[key] = float(norm)
                except Exception as exc:
                    report["failures"].append({"row_position": start + i, "stage": "gradient", "error": f"{type(exc).__name__}: {exc}"})
    rows = report["rows"]
    summary = report["summary"]
    summary.update(requested_rows=len(x), evaluated_rows=len(rows),
                   gradient_rows=sum(r["gradient_norm_raw"] is not None for r in rows))
    for name in ("normalizer", "threshold"):
        count = sum(r[name + "_node_count"] for r in rows)
        summary[name + "_node_count"] = count
        for label in ("nonfinite", "below", "above", "near"):
            total = sum(r[name + "_" + label + "_count"] for r in rows)
            summary[name + "_" + label + "_count"] = total
            summary[name + "_" + label + "_fraction"] = total / count if count else None
        for label, op in (("min", min), ("max", max)):
            vals = [r[name + "_preclamp_" + label] for r in rows if r[name + "_preclamp_" + label] is not None]
            summary[name + "_preclamp_" + label] = op(vals) if vals else None
    for key in ("probability", "density_floor_probability", "density_floor_threshold_share", "density_floor_normalizer_share",
                "normalizer_regularizer_share", "probability_without_density_floor", "density_floor_probability_delta",
                "gradient_norm_raw", "gradient_norm_std"):
        vals = torch.tensor([r[key] for r in rows if r[key] is not None], dtype=torch.float64, device="cpu")
        for label in ("mean", "p95", "min", "max"):
            value = {"mean": vals.mean, "p95": lambda: vals.quantile(.95), "min": vals.min, "max": vals.max}[label]() if vals.numel() else None
            summary[key + "_" + label] = float(value) if value is not None and torch.isfinite(value) else None
    report["warnings"] = sorted({w for row in rows for w in row["warnings"]})
    if report["failures"]:
        report["status"] = "partial" if rows else "failed"
    elif report["warnings"] and any(r["probability"] is None or r["normalizer_nonfinite_count"] or r["threshold_nonfinite_count"] for r in rows):
        report["status"] = "partial"
    return report


# ----------------------------------------------------------------------
# Main stub for basic runtime check
# ----------------------------------------------------------------------

def main():
    import argparse
    import json
    import numpy as np

    parser = argparse.ArgumentParser(description="FP64 deployment runtime check and optional numerical diagnostics.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT_PATH))
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--global_row", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model_variant", default=None, help="Omitted: checkpoint default snapshot.")
    parser.add_argument("--numerical_diagnostics", action="store_true")
    parser.add_argument("--diagnostics_out", help="Optional JSON output; requires --numerical_diagnostics.")
    args = parser.parse_args()
    if args.diagnostics_out and not args.numerical_diagnostics:
        parser.error("--diagnostics_out requires --numerical_diagnostics.")
    device = torch.device(args.device)

    def timed(label, call):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.6f} s")
        return result, elapsed

    (model, scaler, ckpt, dtype), _ = timed("load_model_and_scaler", lambda: load_model_and_scaler(
        args.checkpoint, device=device, model_variant=args.model_variant))
    with np.load(args.data) as data:
        inputs = data["X_flat"] if "X_flat" in data else data["X"].reshape(len(data["X"]), -1)
        if args.global_row < 0 or args.global_row >= len(inputs):
            parser.error("--global_row is outside the dataset.")
        x_np = inputs[args.global_row]
    if x_np.dtype != np.float64 or x_np.shape != (int(ckpt["x_dim"]),) or not np.isfinite(x_np).all():
        raise ValueError("Driver input must be finite FP64 and match checkpoint x_dim.")
    x = torch.from_numpy(x_np).to(device=device, dtype=dtype)
    u0, alpha, x_space = 0.0, .95, "raw"
    print(f"Snapshot: {ckpt['loaded_model_variant']}; global row: {args.global_row}; quadrature: {model.quad_nodes.numel()}; density floor: {model.phi.density_floor:g}")
    value, tv = timed("constraint_value (uncached)", lambda: constraint_value(x, model, scaler, u0, alpha, x_space, device, dtype))
    (_, gradient), tg = timed("constraint_value_grad (uncached)", lambda: constraint_value_grad(x, model, scaler, u0, alpha, x_space, device, dtype))
    hessian, th = timed("constraint_hessian (uncached)", lambda: constraint_hessian(x, model, scaler, u0, alpha, x_space, device, dtype))
    cache, tc = timed("prepare_constraint_cache", lambda: prepare_constraint_cache(x, model, scaler, u0, alpha, x_space, device, dtype, max_order=2))
    cached, tvc = timed("constraint_value (cached)", lambda: constraint_value(x, model, scaler, u0, alpha, x_space, device, dtype, cache=cache))
    (_, gc), tgc = timed("constraint_value_grad (cached)", lambda: constraint_value_grad(x, model, scaler, u0, alpha, x_space, device, dtype, cache=cache))
    hc, thc = timed("constraint_hessian (cached)", lambda: constraint_hessian(x, model, scaler, u0, alpha, x_space, device, dtype, cache=cache))
    print(f"total_uncached: {tv + tg + th:.6f} s; total_cached: {tc + tvc + tgc + thc:.6f} s")
    print(f"Constraint values uncached/cached: {value.item()} / {cached.item()}")
    print(f"Gradient shapes uncached/cached: {gradient.shape} / {gc.shape}")
    print(f"Hessian shapes uncached/cached: {hessian.shape} / {hc.shape}")
    if args.numerical_diagnostics:
        try:
            report = constraint_numerical_diagnostics(x, model, scaler, include_gradients=True)
            report.update(global_row=args.global_row, model_variant=ckpt["loaded_model_variant"],
                          epoch=ckpt.get("model_variant_metrics", {}).get(ckpt["loaded_model_variant"], {}).get("epoch"))
        except Exception as exc:
            report = {"status": "failed", "failures": [f"{type(exc).__name__}: {exc}"]}
        print("Numerical diagnostics (excluded from runtime measurements):")
        print(json.dumps(report, indent=2, allow_nan=False))
        if args.diagnostics_out:
            output = Path(args.diagnostics_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
