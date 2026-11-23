"""What a trained fold leaves on disk, and how it is read back.

A fold directory holds ``hparams.json``, the trainer's configuration, and
``checkpoints/adam_<epoch>/``, one Orbax checkpoint per saved epoch with two items:

``model_state``
    the correction network, written by Equinox (:class:`EquinoxCheckpointHandler`);
``opt_state``
    the Adam state, written by Orbax's standard handler.

The hyperparameters include callables (the activation functions), which JSON
cannot hold. They are stored as ``{"__callable__": name}`` and looked up in
:data:`CALLABLE_REGISTRY` on restore, so a record says *which* activation was used
rather than holding an unusable repr.

Orbax records each item's handler as a type string in the step's
``_CHECKPOINT_METADATA``. Checkpoints written before this module existed name
``adm1_package.utils.EquinoxCheckpointHandler.EquinoxCheckpointHandler``, those
written since name this module. Restoring never reads the string: the trainer asks
for the network with :class:`EquinoxStateRestore`, which picks the handler.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

__all__ = [
    "CALLABLE_REGISTRY",
    "EquinoxCheckpointHandler",
    "EquinoxStateRestore",
    "EquinoxStateSave",
    "make_deserializable",
    "make_serializable",
]

# ---------------------------------------------------------------------------
# hparams.json
# ---------------------------------------------------------------------------
#: Name -> callable, for the callables a configuration may hold. Old records name
#: some of them in more than one way, so every name stays.
CALLABLE_REGISTRY = {
    'jax.nn.relu': jax.nn.relu,
    'jax.nn.gelu': jax.nn.gelu,
    'jax.nn.tanh': jax.nn.tanh,
    'jax.nn.sigmoid': jax.nn.sigmoid,
    'jax.nn.softmax': jax.nn.softmax,
    'jax.nn.swish': jax.nn.swish,
    'jax.nn.elu': jax.nn.elu,
    'jax.nn.leaky_relu': jax.nn.leaky_relu,
    'jax.nn.selu': jax.nn.selu,
    'jax.nn.hard_tanh': jax.nn.hard_tanh,
    'jax.nn.softplus': jax.nn.softplus,
    'jax.nn.identity': jax.nn.identity,
    'jax.nn.sparse_sigmoid': jax.nn.sparse_sigmoid,
    'relu': jax.nn.relu,
    'gelu': jax.nn.gelu,
    'tanh': jax.nn.tanh,
    'sigmoid': jax.nn.sigmoid,
    'swish': jax.nn.swish,
    'jnp.tanh': jnp.tanh,
    'jnp.sigmoid': jax.nn.sigmoid,  # jnp has no sigmoid
    'float32': jnp.float32,
    'float64': jnp.float64,
}


def _callable_name(func: Callable) -> str:
    """The registry name of ``func``; its repr (which cannot be restored) if it has none."""
    if hasattr(func, '__module__') and hasattr(func, '__name__'):
        for name, registered in CALLABLE_REGISTRY.items():
            if registered is func:
                return name
        full_name = f"{func.__module__}.{func.__name__}"
        if full_name in CALLABLE_REGISTRY:
            return full_name
        if func.__name__ in CALLABLE_REGISTRY:
            return func.__name__
    return str(func)


def make_serializable(d):
    """``d`` with every callable replaced by ``{"__callable__": name}`` and every other
    value JSON cannot hold by its ``str``."""
    if isinstance(d, dict):
        return {k: make_serializable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [make_serializable(v) for v in d]
    if callable(d):
        return {"__callable__": _callable_name(d)}
    try:
        json.dumps(d)
        return d
    except (TypeError, OverflowError):
        return str(d)


def make_deserializable(d: Any) -> Any:
    """The inverse of :func:`make_serializable` for the callables: names become callables."""
    if isinstance(d, dict):
        if "__callable__" in d:
            name = d["__callable__"]
            if name not in CALLABLE_REGISTRY:
                raise ValueError(f"Unknown callable '{name}'. Available callables: "
                                 f"{list(CALLABLE_REGISTRY.keys())}")
            return CALLABLE_REGISTRY[name]
        return {k: make_deserializable(v) for k, v in d.items()}
    if isinstance(d, list):
        return [make_deserializable(v) for v in d]
    if isinstance(d, tuple):
        return tuple(make_deserializable(v) for v in d)
    return d


# ---------------------------------------------------------------------------
# The network inside an Orbax checkpoint
# ---------------------------------------------------------------------------
class EquinoxCheckpointHandler(ocp.CheckpointHandler):
    """Writes an Equinox module's arrays to ``model.eqx`` and reads them back into a
    module of the same structure (the template)."""

    def save(self, directory: str, args: "EquinoxStateSave"):
        eqx.tree_serialise_leaves(os.path.join(directory, "model.eqx"), args.item,
                                  is_leaf=eqx.is_array_like)

    def restore(self, directory: str, args: "EquinoxStateRestore") -> eqx.Module:
        return eqx.tree_deserialise_leaves(os.path.join(directory, "model.eqx"), args.item,
                                           is_leaf=eqx.is_array_like)


@ocp.args.register_with_handler(EquinoxCheckpointHandler, for_save=True)
@dataclass
class EquinoxStateSave(ocp.args.CheckpointArgs):
    item: eqx.Module


@ocp.args.register_with_handler(EquinoxCheckpointHandler, for_restore=True)
@dataclass
class EquinoxStateRestore(ocp.args.CheckpointArgs):
    #: A module of the stored structure; its arrays are replaced by the stored ones.
    item: eqx.Module
