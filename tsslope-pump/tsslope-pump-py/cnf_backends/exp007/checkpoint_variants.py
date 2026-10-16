"""Shared helpers for selecting model states from CNF checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


DEFAULT_MODEL_VARIANT = "best_val"
AVAILABLE_MODEL_VARIANTS = ("best_val", "best_train", "final")

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
    model_variant: str | None = DEFAULT_MODEL_VARIANT,
) -> tuple[Mapping[str, torch.Tensor], str]:
    """
    Return the requested state_dict from a v2 or legacy checkpoint.

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
    if isinstance(explicit, Mapping):
        metrics: dict[str, dict[str, Any]] = {}
        for key, value in explicit.items():
            variant = normalize_model_variant(str(key))
            if isinstance(value, Mapping):
                metrics[variant] = dict(value)
        if all(variant in metrics for variant in AVAILABLE_MODEL_VARIANTS):
            return metrics

    train_loss_hist = [float(v) for v in ckpt.get("hist_train_loss", [])]
    val_loss_hist = [float(v) for v in ckpt.get("hist_val_loss", [])]
    lr_hist = [float(v) for v in ckpt.get("hist_lr", [])]
    return {
        variant: variant_metric_from_histories(train_loss_hist, val_loss_hist, lr_hist, variant)
        for variant in AVAILABLE_MODEL_VARIANTS
    }
