"""The experiment options retired before this release, and how the paper's records read.

Every path the paper's models do not use was taken out of the package: the
literature-anchored ("global") hybrids and hybrids on the installed calibration, the
memory of the operating conditions, the spline forcing, networks transferred between
variants, the pooled loss scale, the campaign weighting, the synthetic samples, every
optimizer but Adam, and the deployment model.

The paper's run directories still record those options, every one at the value that
changes nothing in the code that remains, e.g. ``"memory_days": 0.0`` in each fold's
``hparams.json``. :func:`drop_retired` reads such a record as this code does: a retired
field at that value is dropped, at any other value the record is refused with
:func:`retired`, rather than read as a model it is not.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["REFUSAL", "RETIRED_FIELDS", "retired", "drop_retired"]

#: What every refusal of a retired option says.
REFUSAL = "was retired before this release"

#: Retired fields of a recipe (``Hyperparameters``) or of a fold's ``hparams.json``, each
#: with the value at which it changed nothing in the code that remains, and what it was.
RETIRED_FIELDS: dict[str, tuple[Any, str]] = {
    "memory_days": (0.0, "the memory of the operating conditions"),
    "synthetic_ratio": (0.0, "the synthetic (perturbed) training samples"),
    "campaign_balance": (False, "the campaign weighting of the loss"),
    "validation_every": (0, "a validation interval of its own"),
    "loss_scale": ("reactor", "the pooled loss scale"),
    # hparams.json: the scales of the pooled loss, none for each reactor's own range
    "objective_scale": (None, "the pooled loss scale"),
}


def retired(feature: str, where: str | None = None) -> ValueError:
    """The error for a record or a request that needs ``feature``, which was retired."""
    prefix = f"{where}: " if where else ""
    return ValueError(f"{prefix}{feature} {REFUSAL}")


def drop_retired(record: Mapping[str, Any], where: str | None = None) -> dict[str, Any]:
    """``record`` without its retired fields (:data:`RETIRED_FIELDS`).

    A field at the value that changed nothing is dropped silently; at any other value the
    record describes a model this code no longer builds, and :func:`retired` is raised.
    """
    kept = dict(record)
    for name, (inert, feature) in RETIRED_FIELDS.items():
        if name not in kept:
            continue
        value = kept.pop(name)
        if value != inert:
            raise retired(f"{feature} ({name}={value!r})", where)
    return kept
