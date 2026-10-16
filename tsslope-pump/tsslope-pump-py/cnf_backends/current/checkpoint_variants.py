"""Shared helpers for selecting model states from CNF checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import hashlib
import json

import torch

try:
    from .architecture import DEFAULT_DENSITY_FLOOR, validate_density_floor
except ImportError:  # pragma: no cover - direct script execution fallback
    from architecture import DEFAULT_DENSITY_FLOOR, validate_density_floor


DEFAULT_MODEL_VARIANT = "best_val"
AVAILABLE_MODEL_VARIANTS = ("best_val", "best_train", "final", "best_val_bce", "best_val_brier", "best_val_objective")
SELECTION_VARIANTS = {"nll": "best_val", "binary_logloss": "best_val_bce", "brier": "best_val_brier", "objective": "best_val_objective"}

_VARIANT_ALIASES = {
    "best_val": "best_val",
    "best_validation": "best_val",
    "validation": "best_val",
    "val": "best_val",
    "best_train": "best_train",
    "best_training": "best_train",
    "training": "best_train",
    "train": "best_train",
    "final": "final",
    "last": "final",
    "last_epoch": "final",
    "best_val_bce": "best_val_bce",
    "best_val_brier": "best_val_brier",
    "best_val_objective": "best_val_objective",
}


def normalize_model_variant(model_variant: str | None, *, default: str = DEFAULT_MODEL_VARIANT) -> str:
    """Normalize user-facing checkpoint variant aliases to canonical names."""
    raw = default if model_variant is None else str(model_variant)
    key = raw.strip().lower().replace("-", "_")
    if key in {"default", ""}:
        key = default
    try:
        return _VARIANT_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_VARIANT_ALIASES))
        raise ValueError(f"Unknown checkpoint model variant '{raw}'. Expected one of: {allowed}.") from exc


def clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy a model state_dict to detached CPU tensors for checkpoint storage."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def resolve_checkpoint_state_dict(
    ckpt: Mapping[str, Any],
    model_variant: str | None = None,
) -> tuple[Mapping[str, torch.Tensor], str]:
    """
    Return the requested state_dict from a v3, v2, or legacy checkpoint.

    Legacy checkpoints only carry ``state_dict``. They are accepted for the
    default/best-validation variant because that was the only loadable state.
    """
    default_variant = normalize_model_variant(
        str(ckpt.get("default_model_variant", DEFAULT_MODEL_VARIANT))
    )
    requested = normalize_model_variant(model_variant, default=default_variant)

    model_states = ckpt.get("model_states")
    if isinstance(model_states, Mapping):
        state = model_states.get(requested)
        if isinstance(state, Mapping):
            return state, requested
        available = ", ".join(sorted(str(k) for k in model_states.keys()))
        raise KeyError(
            f"Checkpoint does not contain requested model_variant='{requested}'. "
            f"Available variants: {available or 'none'}."
        )

    if requested == DEFAULT_MODEL_VARIANT and isinstance(ckpt.get("state_dict"), Mapping):
        return ckpt["state_dict"], requested

    raise KeyError(
        f"Checkpoint does not contain model_states and cannot provide model_variant='{requested}'. "
        "Legacy checkpoints only support the default best_val variant."
    )


def variant_metric_from_histories(
    train_loss_hist: list[float],
    val_loss_hist: list[float],
    lr_hist: list[float],
    variant: str,
) -> dict[str, float | int | str | None]:
    """Derive variant metrics from histories when explicit metadata is missing."""
    if not train_loss_hist and not val_loss_hist:
        return {
            "epoch": None,
            "train_loss": None,
            "val_loss": None,
            "lr": None,
            "metric_name": variant,
            "metric_value": None,
        }

    n_epochs = max(len(train_loss_hist), len(val_loss_hist), len(lr_hist), 1)
    if variant == "best_train" and train_loss_hist:
        idx = min(range(len(train_loss_hist)), key=lambda i: train_loss_hist[i])
        metric_name = "train_loss"
        metric_value = train_loss_hist[idx]
    elif variant == "best_val" and val_loss_hist:
        idx = min(range(len(val_loss_hist)), key=lambda i: val_loss_hist[i])
        metric_name = "val_loss"
        metric_value = val_loss_hist[idx]
    else:
        idx = n_epochs - 1
        metric_name = "final_epoch"
        metric_value = None

    return {
        "epoch": idx + 1,
        "train_loss": train_loss_hist[idx] if idx < len(train_loss_hist) else None,
        "val_loss": val_loss_hist[idx] if idx < len(val_loss_hist) else None,
        "lr": lr_hist[idx] if idx < len(lr_hist) else None,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }


def checkpoint_variant_metrics(ckpt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return metrics for the standard checkpoint variants with legacy fallback."""
    explicit = ckpt.get("model_variant_metrics")
    metrics: dict[str, dict[str, Any]] = {}
    if isinstance(explicit, Mapping):
        for key, value in explicit.items():
            variant = normalize_model_variant(str(key))
            if isinstance(value, Mapping):
                metrics[variant] = dict(value)

    train_loss_hist = [float(v) for v in ckpt.get("hist_train_loss", [])]
    val_loss_hist = [float(v) for v in ckpt.get("hist_val_loss", [])]
    lr_hist = [float(v) for v in ckpt.get("hist_lr", [])]
    for variant in ("best_val", "best_train", "final"):
        if variant not in metrics:
            metrics[variant] = variant_metric_from_histories(train_loss_hist, val_loss_hist, lr_hist, variant)
    return metrics


def add_score_metadata(results, ckpt):
    """Expose retained-snapshot scores without loading dense prediction histories."""
    results["density_floor"] = validate_density_floor(ckpt.get("density_floor", DEFAULT_DENSITY_FLOOR))
    results.update({key: ckpt.get(key) for key in ("constraint_loss", "constraint_weight", "selection_metric",
        "training_status", "saved_epoch", "dataset_sha256", "split_fingerprints", "epoch_metrics", "numerical_diagnostics")})
    results["model_variant_metrics"] = checkpoint_variant_metrics(ckpt)
    rows = ckpt.get("val_rows")
    if rows is not None:
        provenance = {"dataset": ckpt.get("dataset_sha256", ckpt.get("data_file")),
                      "rows": sorted(int(v) for v in rows), "nan_y_policy": ckpt.get("nan_y_policy", "ignore"),
                      "y_floor_raw": ckpt.get("y_floor_raw", -100.0) if ckpt.get("nan_y_policy") == "floor_raw" else None}
        results["validation_cohort"] = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    else:
        results["validation_cohort"] = None
    for criterion, variant in SELECTION_VARIANTS.items():
        metric = results["model_variant_metrics"].get(variant, {})
        value = metric.get("val", {}).get(criterion)
        if criterion == "nll":
            value = metric.get("val_loss")
        results[f"best_{criterion}"] = value


def select_score_metadata(results, criterion):
    variant = SELECTION_VARIANTS[criterion]
    snapshot = results["model_variant_metrics"].get(variant, {})
    value = snapshot.get("val_loss") if criterion == "nll" else snapshot.get("val", {}).get(criterion)
    if value is None:
        raise ValueError(f"No saved validation {criterion} score/state. Use explicit --rank_by nll for legacy runs or post-hoc --constraint_diagnostics.")
    results.update(ranking_metric=criterion, ranking_score=value, selected_model_variant=variant,
                   selected_epoch=snapshot.get("epoch"), selected_val_nll=snapshot.get("val_loss"),
                   selected_train_nll=snapshot.get("train_loss"), selected_scores=snapshot.get("val", {}))
    results["selected_numerical_diagnostics"] = (results.get("numerical_diagnostics") or {}).get("snapshots", {}).get(variant)
    if results["selected_numerical_diagnostics"] is not None and results["selected_numerical_diagnostics"].get("epoch") != snapshot.get("epoch"):
        results["selected_numerical_diagnostics"] = None
    return value
