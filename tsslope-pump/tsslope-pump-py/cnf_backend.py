"""Adapt standalone CNF inference to the existing Julia callback contract."""

from importlib import import_module
from pathlib import Path
import warnings

import numpy as np
import torch


def load_cnf(model_path, data_record, active_gen_only=True, *, backend="current",
             confidence=0.95, threshold=0.0, device="cpu", model_variant=None):
    if backend not in {"current", "exp007", "legacy_relu"}:
        raise ValueError(f"Unknown CNF backend {backend!r}; choose current, exp007, or legacy_relu.")
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("CNF confidence must be finite and between 0 and 1.")
    if not np.isfinite(threshold):
        raise ValueError("CNF threshold must be finite.")
    device = torch.device(device)
    path = str(Path(model_path).expanduser().resolve(strict=True))
    checkpoint = torch.load(path, map_location="cpu")
    architecture = import_module(".cnf_backends.current.architecture", __package__)
    variants = import_module(".cnf_backends.current.checkpoint_variants", __package__)
    spec = architecture.architecture_from_checkpoint(checkpoint)
    floor = architecture.validate_density_floor(checkpoint.get("density_floor", 1e-6))
    state, selected = variants.resolve_checkpoint_state_dict(checkpoint, model_variant)
    if any(t.dtype != torch.float64 for t in state.values() if torch.is_tensor(t) and t.is_floating_point()):
        raise TypeError("CNF checkpoint tensors must have dtype torch.float64.")

    if backend != "current" and floor != 1e-6:
        raise ValueError(f"{backend} hardcodes density_floor=1e-6; use current for this checkpoint.")
    if backend == "legacy_relu":
        if not spec.relu_fast_eligible:
            raise ValueError("legacy_relu only supports ReLU checkpoints; use exp007 or current.")
        if device.type != "cpu" or threshold != 0.0:
            raise ValueError("legacy_relu only supports its historical CPU / TSI threshold 0 setup.")
        root_state = checkpoint.get("state_dict", {})
        if state.keys() != root_state.keys() or any(not torch.equal(t, root_state[k]) for k, t in state.items()):
            raise ValueError("legacy_relu cannot select a snapshot different from checkpoint state_dict; use current.")
        warnings.warn("legacy_relu is historical only: its callbacks override confidence to 0.9 (10% risk).", RuntimeWarning)
        loader = import_module(".load_Surrogate", __package__)
        surrogate = loader.load_surrogate(path, data_record, "CNF", active_gen_only=active_gen_only)
        surrogate["alpha"] = 0.9
        surrogate["cnf_backend"] = backend
        surrogate["cnf_module"] = loader
        model, scaler = surrogate["model"], surrogate["scaler"]
    else:
        module = import_module(f".cnf_backends.{backend}.CNF_inference_HPC", __package__)
        if backend == "exp007":
            old_variants = import_module(".cnf_backends.exp007.checkpoint_variants", __package__)
            try:
                old_variants.resolve_checkpoint_state_dict(checkpoint, model_variant)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"exp007 cannot honor this checkpoint snapshot configuration; use current: {exc}") from exc
        model, scaler, checkpoint, dtype = module.load_model_and_scaler(
            path, device=device, override_dtype="float64", model_variant=model_variant,
        )
        model.requires_grad_(False)
        surrogate = {"model": model, "scaler": scaler, "ckpt": checkpoint,
                     "dtype": dtype, "u0": float(threshold), "alpha": float(confidence),
                     "device": device, "x_space": "raw", "model_type": "CNF",
                     "active_gen_only": active_gen_only, "cnf_backend": backend,
                     "cnf_module": module}

    for name, tensor in (("scaler_mean", scaler.mean_), ("scaler_std", scaler.std_)):
        if tensor.dtype != torch.float64 or tensor.numel() != int(checkpoint["x_dim"]) or not torch.isfinite(tensor).all():
            raise ValueError(f"CNF {name} must be finite FP64 with x_dim entries.")
    if not (scaler.std_ > 0).all():
        raise ValueError("CNF scaler_std must be strictly positive.")
    metrics = variants.checkpoint_variant_metrics(checkpoint).get(selected, {})
    surrogate["cnf_variant"] = selected
    surrogate["cnf_epoch"] = metrics.get("epoch")
    surrogate["cnf_checkpoint_path"] = path
    scaling = "identity" if torch.all(scaler.mean_ == 0) and torch.all(scaler.std_ == 1) else "saved standardizer"
    print(f"CNF: {backend} | module={surrogate['cnf_module'].__file__}\n"
          f"  checkpoint={path} | snapshot={selected} | epoch={surrogate['cnf_epoch']}\n"
          f"  FP64 {device} | raw per-unit [Pgen,Pload,Qgen,Qload] | scaler={scaling}\n"
          f"  hidden={list(spec.hidden)} | activations={architecture.format_activation_summary(spec.activations)} | "
          f"Q={model.quad_nodes.numel()} | density_floor={floor:g}\n"
          f"  TSI threshold={surrogate['u0']:g} | confidence={surrogate['alpha']:g} | feasible margin >= 0"
          + ("\n  WARNING: legacy_relu preserves the historical 90% confidence override." if backend == "legacy_relu" else ""))
    return surrogate


def _raw_input(surrogate, Pg, Qg, Pl, Ql, st_args):
    if st_args.get("reorder_pg", False):
        raise ValueError("CNF reorder_pg=true is unsupported by the current Julia derivative mapping; fix Julia first.")
    arrays = [np.asarray(v, dtype=np.float64).reshape(-1) for v in (Pg, Qg, Pl, Ql)]
    pg, qg, pl, ql = arrays
    if pg.size != qg.size or pl.size != ql.size:
        raise ValueError("CNF P/Q generator lengths and P/Q load lengths must match.")
    if any(not np.isfinite(v).all() for v in arrays):
        raise ValueError("CNF raw inputs contain NaN or Inf.")
    indices = np.asarray(st_args["gen_idx"])
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or indices.size == 0:
        raise ValueError("CNF gen_idx must be a nonempty one-dimensional array of zero-based integers.")
    if np.any(indices < 0) or np.any(indices >= pg.size) or np.unique(indices).size != indices.size:
        raise ValueError("CNF gen_idx contains duplicate or out-of-range indices.")
    for key, actual in (("numb_gen", pg.size), ("numb_active_gen", indices.size), ("numb_loads", pl.size)):
        if key in st_args and st_args[key] != actual:
            raise ValueError(f"CNF {key} disagrees with input lengths ({actual}).")
    if indices.size == pg.size and not np.array_equal(indices, np.arange(pg.size)):
        raise ValueError("CNF all-active nonidentity gen_idx is unsupported by the current Julia derivative mapping; fix Julia first.")
    x = np.concatenate((pg[indices], pl, qg[indices], ql))
    if x.size != int(surrogate["ckpt"]["x_dim"]):
        raise ValueError(f"CNF input has {x.size} features, checkpoint expects {surrogate['ckpt']['x_dim']}.")

    x = torch.as_tensor(x, dtype=torch.float64, device=surrogate["device"])
    model = surrogate["model"]
    if surrogate["cnf_backend"] == "current" and not getattr(model, "_cnf_clamp_warned", False):
        report = surrogate["cnf_module"].constraint_numerical_diagnostics(
            x, model, surrogate["scaler"], u0=surrogate["u0"],
            include_gradients=False,
        )
        stats = report["summary"]
        clipped = {
            name: stats[name + "_below_count"] + stats[name + "_above_count"]
            for name in ("normalizer", "threshold")
        }
        if any(clipped.values()):
            model._cnf_clamp_warned = True
            print(f"WARNING [CNF] pre-clamp output outside [-20,20]: "
                  f"clipped quadrature nodes {clipped}", flush=True)
    return x


def value(surrogate, Pg, Qg, st_args):
    x = _raw_input(surrogate, Pg, Qg, st_args["PL"], st_args["QL"], st_args)
    if surrogate["cnf_backend"] == "legacy_relu":
        module = import_module(".TSI_constraint", __package__)
        return module.TSI_constraint(surrogate, Pg, Qg, st_args)
    # HPC uses F - (1-alpha) <= 0; Julia uses the opposite feasibility margin.
    result = surrogate["cnf_module"].constraint_value(
        x, surrogate["model"], surrogate["scaler"], surrogate["u0"], surrogate["alpha"],
        "raw", surrogate["device"], surrogate["dtype"],
    )
    return -result.item()


def gradient(surrogate, Pg, Qg, Pl, Ql, st_args):
    x = _raw_input(surrogate, Pg, Qg, Pl, Ql, st_args)
    if surrogate["cnf_backend"] == "legacy_relu":
        module = import_module(".dTSI_dVdP", __package__)
        return module.dTSI_dVdP(surrogate, Pg, Qg, Pl, Ql, st_args)
    _, result = surrogate["cnf_module"].constraint_value_grad(
        x, surrogate["model"], surrogate["scaler"], surrogate["u0"], surrogate["alpha"],
        "raw", surrogate["device"], surrogate["dtype"],
    )
    return -result.detach().cpu().numpy().reshape(-1)


def hessian(surrogate, Pg, Qg, Pl, Ql, muTSI, st_args):
    x = _raw_input(surrogate, Pg, Qg, Pl, Ql, st_args)
    if surrogate["cnf_backend"] == "legacy_relu":
        module = import_module(".d2TSI_dV2dP2", __package__)
        return module.d2TSI_dV2dP2(surrogate, Pg, Qg, Pl, Ql, muTSI, st_args)
    result = surrogate["cnf_module"].constraint_hessian(
        x, surrogate["model"], surrogate["scaler"], surrogate["u0"], surrogate["alpha"],
        "raw", surrogate["device"], surrogate["dtype"],
    )
    return -result.detach().cpu().numpy()
