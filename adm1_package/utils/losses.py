"""Residual scaling for the multi-objective training loss.

The loss compares objectives that live on very different scales -- biogas
production in L/d (interpercentile range of order 5), acid concentrations in g/L
(order 0.3), pH (order 0.3) -- so residuals have to be scaled before they are
summed. The paper's loss scales them by the interpercentile range
:math:`Q_{0.9} - Q_{0.1}` of each objective:

    range = max(Q_0.9 - Q_0.1, std)

The standard deviation guards against a degenerate range when an objective sits
at its detection limit for most of a time series; it is a numerical safeguard,
not a reweighting, because it only ever applies when the interpercentile range
is the smaller of the two.

The range is not floored at 1. Such a floor is unit dependent: it would leave
every objective whose spread is below one unit (pH, the acids, TAN) effectively
unnormalized, which would scale their weight in the loss by 0.12-0.40.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["robust_target_range"]


def robust_target_range(targets: jnp.ndarray, axis: int, where=None,
                        epsilon: float = 1e-6) -> jnp.ndarray:
    """Per-objective scale used to normalize residuals.

    Parameters
    ----------
    targets:
        Measured values, reduced over ``axis`` (the time axis).
    axis:
        Time axis: ``0`` for a single sample, ``1`` for a batch.
    where:
        The points the loss scores. Only they set the scale, so neither the
        spin-up window nor a short record's padded tail can change the loss.
        Default: every point.
    epsilon:
        Lower bound, so a constant objective cannot produce a division by zero.
    """
    if where is not None:
        targets = jnp.where(where, targets, jnp.nan)
    target_range = jnp.nanpercentile(targets, 90, axis=axis, keepdims=True) - jnp.nanpercentile(
        targets, 10, axis=axis, keepdims=True
    )
    target_range = jnp.maximum(target_range, jnp.nanstd(targets, axis=axis, keepdims=True))
    # No scored point at all leaves NaN; those residuals are masked out anyway.
    return jnp.where(jnp.isfinite(target_range), jnp.maximum(target_range, epsilon), epsilon)
