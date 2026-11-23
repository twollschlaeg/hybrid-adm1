"""How the forcing runs between its daily records.

The feed, the temperature and the influent are recorded once a day; the solver needs
them at every time. Every dataset builds its forcing with :func:`forcing_coefficients`,
as the coefficients of a ``diffrax.CubicInterpolation``.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["FORCINGS", "resolve_forcing", "forcing_coefficients", "smoothstep_coefficients"]

#: How the forcing is interpolated between the daily records: ``smoothstep``
#: (:func:`forcing_coefficients`). The cubic ``spline`` before it was retired; runs and
#: calibrations made with it, which record no forcing or ``"spline"``, are refused.
FORCINGS = ("smoothstep",)


def resolve_forcing(forcing: str | None) -> str:
    """Validate a forcing (:data:`FORCINGS`). ``None``, a record made before the forcing
    was recorded, and ``"spline"`` name the retired spline."""
    if forcing in (None, "spline"):
        from adm1_package.experiments.retired import retired

        raise retired("the spline forcing (a record with forcing 'spline' or none)")
    if forcing not in FORCINGS:
        raise ValueError(f"unknown forcing {forcing!r}; expected one of {', '.join(FORCINGS)}")
    return forcing


def forcing_coefficients(times, values, forcing: str = "smoothstep"):
    """``diffrax.CubicInterpolation`` coefficients of a forcing recorded once a day.

    On ``[t_i, t_(i+1)]`` the forcing moves from the record ``v_i`` to the next one,
    ``v_(i+1)``, along a smoothstep with zero slope at both ends. It passes through every
    record at its time stamp; it never leaves the range of the two values, so a day without
    feed cannot drive the inflow negative; and it is continuously differentiable, so the
    solver steps as it did through the cubic spline (``dx.backward_hermite_coefficients``)
    this replaces. That spline overshot a step by about 15% on the next day and went
    negative after a day without feed. A hold with the jump times passed to the solver took
    up to 2.8 times the steps and 3.3 times the gradient time.

    A feed recorded at a reading is given after it, and a gas reading is the volume since
    the previous one. Against that the forcing runs half a day ahead on average, which
    makes up for comparing the model's gas rate at a reading's time stamp with a reading
    that averages the day before. R4 recalibrated on rice-bran fit with it as well as with
    the spline (loss 0.185 and 0.186), and worse with the feed held from each record to the
    next (0.212) or with this smoothstep reaching each record a day late (0.211).

    The forcing passes through its values at ``times``, so one built again from them after
    a change (the input noise) is built the same way. ``times`` and ``values`` (time on the
    first axis) must be finite.
    """
    resolve_forcing(forcing)
    return smoothstep_coefficients(times, values)


def smoothstep_coefficients(times, values):
    """``diffrax.CubicInterpolation`` coefficients through ``values`` at ``times``, from one
    to the next along a smoothstep: zero slope at every knot, never outside the two
    values."""
    times, values = jnp.asarray(times), jnp.asarray(values)
    span = (times[1:] - times[:-1])[:, None]
    start, change = values[:-1], values[1:] - values[:-1]
    return (-2.0 * change / span**3, 3.0 * change / span**2, jnp.zeros_like(start), start)
