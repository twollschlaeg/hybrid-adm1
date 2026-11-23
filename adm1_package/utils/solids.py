"""Total and volatile solids of the reduced models.

The reduced models carry only the degradable part of the feed: the influent
particulates enter multiplied by their degradabilities ``f_ch``, ``f_pr`` and
``f_li``. The rest of the feed's dry matter, its non-degradable organics and its
ash, is no state. Nor is ``1000 - S_h2o`` the total solids: the water balance never
loses the dry matter that leaves as biogas, so it stays near the feed's, about
twice what the reactors measure.

Measured volatile solids (VS, oTS) are the remaining particulates, the biomass and
the non-degradable organics; total solids (TS) add the ash. Neither of the two
missing pools reacts with anything, so each only washes out,

    dY/dt = q_in(t) / V_liq * (Y_in(t) - Y),

a linear filter of the feed. It is integrated here, after the solve, on the
solution's time grid, rather than as two more states of the stiff system: every
other output stays exactly what it was, and so does the solver's cost.

``X_inert`` is fed by the non-degradable ``1 - f`` of the influent particulates,
``X_ash`` by the feed's dry matter ``1000 - S_h2o`` less its particulates. The
crude carbohydrates are the dry matter less crude ash, protein and fat, so that is
the crude ash. Both pools start where the first measurement puts them: ``X_inert``
at the measured VS less the model's own solids, ``X_ash`` at TS less VS. Without a
measured start they start at the first feed, their steady state.

Dissolved species are left out: volatile acids and ammonium carbonate largely
evaporate at 105 degC, and the reactors' TS is not corrected for them.

Concentrations are g/L; the measurements are g/kg of fresh matter, which is the same
at the density of water that the records assume for ``S_h2o = 1000 - TS``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import diffrax as dx
import jax
import jax.numpy as jnp

__all__ = ["SOLIDS", "POOLS", "PARTICULATES", "biomass", "washout", "solids"]

#: The measured quantities, in the order the datasets load them.
SOLIDS = ("TS", "VS")
#: The two pools the models lack: non-degradable organics and ash.
POOLS = ("X_inert", "X_ash")
#: The degradable particulates, fed through the factors of the same order.
PARTICULATES = ("X_ch", "X_pr", "X_li")
_FACTORS = ("f_ch", "f_pr", "f_li")


def biomass(states: Sequence[str]) -> tuple[str, ...]:
    """The biomass among a model's states: every particulate that is not fed."""
    return tuple(name for name in states if name.startswith("X_") and name not in PARTICULATES)


def washout(times, inflow, concentration, V_liq, start):
    """Pools that only wash out, ``dY/dt = inflow / V_liq * (concentration - Y)``.

    ``inflow`` (L/d) and ``concentration`` (g/L, one column per pool) are the values at
    ``times``; between two of them the dilution rate and the load are their means, and
    each interval is integrated exactly. That is exact for a constant feed, second order
    otherwise, and holds a pool where no time passes, as on the datasets' padded tail.
    """
    rate = inflow / V_liq
    load = rate[:, None] * concentration
    step = jnp.diff(times)
    mean_rate = 0.5 * (rate[:-1] + rate[1:])
    mean_load = 0.5 * (load[:-1] + load[1:])
    decay = jnp.exp(-mean_rate * step)
    # (1 - exp(-D dt)) / D, which tends to dt without inflow
    flowing = mean_rate > 0
    gain = jnp.where(flowing, -jnp.expm1(-mean_rate * step) / jnp.where(flowing, mean_rate, 1.0),
                     step)

    def advance(pools, interval):
        decay_k, gain_k, load_k = interval
        pools = pools * decay_k + gain_k * load_k
        return pools, pools

    _, path = jax.lax.scan(advance, start, (decay, gain, mean_load))
    return jnp.concatenate([start[None], path], axis=0)


def solids(
    solution: Mapping[str, Any],
    input_data: Mapping[str, Any],
    params: Mapping[str, Any],
    factors: Mapping[str, Any],
    biomass: Sequence[str],
) -> dict[str, Any]:
    """The reactor's TS and VS and the two pools behind them, over ``solution["time"]``.

    ``input_data`` is the sample's forcing, whose last four columns are ``S_h2o``,
    ``X_ch``, ``X_pr`` and ``X_li`` in every variant; its ``start`` may hold the first
    measured ``TS`` and ``VS``. ``factors`` holds ``f_ch``, ``f_pr`` and ``f_li``, one
    value or one per time point (a hybrid's corrected ones). ``biomass`` names the
    solution's biomass states.
    """
    times = jnp.asarray(solution["time"])
    forcing = dx.CubicInterpolation(input_data["time"], input_data["coeffs"])
    knots = jax.vmap(forcing.evaluate)(times)
    inflow = knots[:, 1]
    water, feed = knots[:, -4], knots[:, -3:]
    fed = inflow > 0

    degradable = jnp.stack([jnp.broadcast_to(jnp.asarray(factors[name], dtype=float), times.shape)
                            for name in _FACTORS], axis=1)
    inert_in = jnp.sum((1.0 - degradable) * feed, axis=1)
    # A day without feed may carry no composition at all (water 0): no ash then.
    ash_in = jnp.where(fed, jnp.maximum(1000.0 - water - jnp.sum(feed, axis=1), 0.0), 0.0)

    held = sum(jnp.asarray(solution[name]) for name in (*PARTICULATES, *biomass))

    start = input_data.get("start", {})
    measured_vs = jnp.asarray(start.get("VS", jnp.nan), dtype=float)
    measured_ts = jnp.asarray(start.get("TS", jnp.nan), dtype=float)
    has_vs = jnp.isfinite(measured_vs)
    has_both = has_vs & jnp.isfinite(measured_ts)
    vs0 = jnp.where(has_vs, measured_vs, 0.0)
    ts0 = jnp.where(has_both, measured_ts, 0.0)
    first = jnp.argmax(fed)
    inert0 = jnp.where(has_vs, jnp.maximum(vs0 - held[0], 0.0), inert_in[first])
    ash0 = jnp.where(has_both, jnp.maximum(ts0 - vs0, 0.0), ash_in[first])

    pools = washout(times, inflow, jnp.stack([inert_in, ash_in], axis=1), params["V_liq"],
                    jnp.stack([inert0, ash0]))
    inert, ash = pools[:, 0], pools[:, 1]
    volatile = held + inert
    return {"X_inert": inert, "X_ash": ash, "VS": volatile, "TS": volatile + ash}
