from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import torch.nn as nn


SUPPORTED_ACTIVATIONS: tuple[str, ...] = ("relu", "tanh", "softplus", "silu")
DEFAULT_DENSITY_FLOOR = 1e-6


def validate_density_floor(value: object) -> float:
    """Resolve a fixed model setting without accepting invalid numerical floors."""
    try:
        floor = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("density_floor must be finite and nonnegative.") from exc
    if isinstance(value, bool) or not math.isfinite(floor) or floor < 0:
        raise ValueError("density_floor must be finite and nonnegative.")
    return floor


def normalize_hidden(hidden: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(width) for width in hidden)


def normalize_activation_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Activation name must be a string, got {type(name).__name__}.")
    normalized = name.strip().lower()
    if normalized not in SUPPORTED_ACTIVATIONS:
        supported = ", ".join(SUPPORTED_ACTIVATIONS)
        raise ValueError(f"Unsupported activation '{name}'. Expected one of: {supported}.")
    return normalized


def resolve_activations(
    hidden: Sequence[int],
    activation: str = "relu",
    activations: Sequence[str] | None = None,
) -> tuple[str, ...]:
    hidden_tuple = normalize_hidden(hidden)
    if activations is None:
        resolved = tuple(normalize_activation_name(activation) for _ in hidden_tuple)
    else:
        resolved = tuple(normalize_activation_name(name) for name in activations)
        if len(resolved) != len(hidden_tuple):
            raise ValueError(
                f"Expected {len(hidden_tuple)} activations for hidden={hidden_tuple}, "
                f"got {len(resolved)}."
            )
    return resolved


def activation_module(name: str) -> nn.Module:
    normalized = normalize_activation_name(name)
    module_map: Mapping[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "softplus": nn.Softplus,
        "silu": nn.SiLU,
    }
    return module_map[normalized]()


@dataclass(frozen=True)
class ArchitectureSpec:
    hidden: tuple[int, ...]
    activations: tuple[str, ...]

    @property
    def uniform_activation(self) -> str | None:
        if not self.activations:
            return None
        first = self.activations[0]
        if all(name == first for name in self.activations):
            return first
        return None

    @property
    def activation(self) -> str:
        uniform = self.uniform_activation
        return uniform if uniform is not None else "mixed"

    @property
    def relu_fast_eligible(self) -> bool:
        return all(name == "relu" for name in self.activations)

    def to_metadata(self) -> dict[str, object]:
        return {
            "hidden": list(self.hidden),
            "activation": self.activation,
            "activations": list(self.activations),
            "activation_summary": format_activation_summary(self.activations),
            "architecture_summary": format_architecture_summary(self.hidden, self.activations),
            "relu_fast_eligible": self.relu_fast_eligible,
        }


def make_architecture_spec(
    hidden: Sequence[int],
    activation: str = "relu",
    activations: Sequence[str] | None = None,
) -> ArchitectureSpec:
    hidden_tuple = normalize_hidden(hidden)
    activations_tuple = resolve_activations(hidden_tuple, activation=activation, activations=activations)
    return ArchitectureSpec(hidden=hidden_tuple, activations=activations_tuple)


def architecture_from_checkpoint(ckpt: Mapping[str, object]) -> ArchitectureSpec:
    hidden = normalize_hidden(ckpt.get("hidden", ()))
    stored_activations = ckpt.get("activations")
    if stored_activations is None:
        return make_architecture_spec(hidden, activation="relu")
    return make_architecture_spec(hidden, activations=tuple(stored_activations))


def format_activation_summary(activations: Sequence[str]) -> str:
    activations_tuple = tuple(activations)
    if not activations_tuple:
        return "none"
    first = activations_tuple[0]
    if all(name == first for name in activations_tuple):
        return first
    return " -> ".join(activations_tuple)


def format_architecture_summary(hidden: Sequence[int], activations: Sequence[str]) -> str:
    hidden_list = list(normalize_hidden(hidden))
    return f"hidden={hidden_list}, activations={format_activation_summary(activations)}"


def build_hidden_layers(
    input_dim: int,
    hidden: Sequence[int],
    activation: str = "relu",
    activations: Sequence[str] | None = None,
) -> list[nn.Module]:
    spec = make_architecture_spec(hidden, activation=activation, activations=activations)
    layers: list[nn.Module] = []
    in_dim = input_dim
    for width, activation_name in zip(spec.hidden, spec.activations):
        layers.append(nn.Linear(in_dim, width))
        layers.append(activation_module(activation_name))
        in_dim = width
    return layers


def apply_architecture_metadata(target: dict[str, object], spec: ArchitectureSpec) -> None:
    target.update(spec.to_metadata())


def architecture_cli_kwargs(
    hidden: Sequence[int],
    activation: str = "relu",
    activations: Sequence[str] | None = None,
) -> dict[str, object]:
    spec = make_architecture_spec(hidden, activation=activation, activations=activations)
    kwargs: dict[str, object] = {
        "hidden": spec.hidden,
        "activation": spec.uniform_activation or "relu",
        "activations": spec.activations,
    }
    return kwargs
