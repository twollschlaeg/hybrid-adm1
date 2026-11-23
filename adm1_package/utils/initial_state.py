"""Where a simulation starts: calibrated states, or a warm-up of the model itself.

``calibrated``
    The per-reactor states in ``<campaign>/R{2,3,4}-IVs_<reactor>.csv``, fitted
    together with the parameters to each reactor's own data. The datasets
    overwrite whatever the first measurements determine: the measured states
    (TAN, water, the acids) and the gas phase, the dissolved CH4 and CO2 in
    equilibrium with that gas phase, and in R2, R3 and the full ADM1 the rest of
    the acid-base chemistry at the first measured pH (:func:`measured_chemistry`).
    The files supply the biology: biomass and particulates. A reactor that was
    never calibrated has no such file.

``warmup``
    The model's own quasi-steady state at the reactor's first-week operating
    conditions, built from nothing a new reactor lacks:

    1. **Forcing.** Temperature, loading rate and influent are averaged over
       the first :data:`FORCING_DAYS` days of the record and held constant.
    2. **Seed.** The measured states take their first measured value, every
       other state a generic value (:data:`GENERIC_SEED`). The warm-up forgets
       the seed: started from the calibrated states instead, it ends within
       0.3% on LaRA, and within 4% on rice-bran for R4. The mechanistic R2/R3
       do not settle on rice-bran: at its measured pH above 8 free ammonia
       holds acetoclastic growth close to washout.
    3. **Warm-up.** The model runs :data:`WARMUP_DAYS` days at that forcing.
       TAN and water are held at their first measurement: they leave only with
       the effluent, on long hydraulic retention times (far longer for
       rice-bran than for LaRA), so they would never settle, and holding them
       gives the biology the measured ammonia. R2/R3 feed no ions, so ``S_cat``
       and ``S_an`` would only wash out; instead the pH is held at its first
       measured value, the net ion charge being whatever the charge balance
       needs for it. That keeps the free ammonia, and with it the ammonia and
       pH inhibition, at the measured start. The network of a UDE is part of the
       model, so a hybrid warms up with its own corrections, and a network that
       reads time sees the start time throughout.
    4. **Start.** The measured states and the gas phase are set from the first
       measurements, as for ``calibrated``. R2/R3 then put the weak acids in
       equilibrium at the measured pH and set the ions so that the start
       reproduces that pH exactly.

In a mechanistic probe across all 15 reactors this cost +0.8% biogas and
-1.4% methane RMSE (R4, median over reactors) against the calibrated states,
+0.7% biogas, +2.8% acetate and -12% pH (R3). It helps LaRA-temp and hurts the
LaRA-substrate reactors, which were changing substrate at the start: their
first biogas is far below the steady state of their first-week feed. Scaling
the warm-up's loading to the first biogas would fix that, but a network would
then see a feed it never saw in training.
"""

from __future__ import annotations

from dataclasses import dataclass

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from adm1_package.utils.gas import dissolved_co2_at_equilibrium
from adm1_package.utils.solver import DT0, MAX_STEPS, make_solver, make_stepsize_controller

__all__ = [
    "INITIAL_STATES",
    "WARMUP_DAYS",
    "FORCING_DAYS",
    "GENERIC_SEED",
    "resolve_initial_state",
    "ConstantForcing",
    "first_week_forcing",
    "StateLayout",
    "measured_chemistry",
    "start_measurements",
    "warm_up",
    "warm_start",
]

INITIAL_STATES = ("calibrated", "warmup")

#: Length of the warm-up. The slowest remaining modes are biomass decay and
#: particulate hydrolysis, 20-50 days; 200 days leaves 0.3% of the seed on LaRA.
WARMUP_DAYS = 200.0
#: The warm-up runs at the mean forcing of this many days from the start.
FORCING_DAYS = 7

#: Where the unmeasured states start the warm-up; anything not listed starts at
#: zero. Particulates build up from the feed, and biomass that grows on its own
#: substrate needs a positive start. The acid-base states and the ions are set
#: from the measured pH instead. X_su, X_aa, X_fa and X_h2 are the full ADM1's,
#: whose warm-up caps the particulates its calibration starts from.
GENERIC_SEED: dict[str, float] = {
    "S_ch4": 0.02,
    "S_IC": 5.0,
    "X_bac": 1.0,
    "X_ac": 0.5,
    "X_pro": 0.2,
    "X_va": 0.2,
    "X_bu": 0.2,
    "X_su": 0.5,
    "X_aa": 0.5,
    "X_fa": 0.2,
    "X_h2": 0.2,
}

#: Gas constant in L bar/(K mol); the models write the van 't Hoff terms with R * 100.
_R = 0.083145


def resolve_initial_state(method: str | None = None) -> str:
    """Validate an initial-state option; ``None`` is the original ``calibrated``."""
    method = "calibrated" if method is None else method
    if method not in INITIAL_STATES:
        raise ValueError(
            f"unknown initial state {method!r}; expected one of {', '.join(INITIAL_STATES)}"
        )
    return method


class ConstantForcing(eqx.Module):
    """A forcing that is the same at every time, with the interface of the
    input interpolation the right-hand sides read."""

    value: jax.Array

    def evaluate(self, t0, t1=None, left=True):
        return self.value

    def derivative(self, t, left=True):
        return jnp.zeros_like(self.value)


def first_week_forcing(interp, t0):
    """Mean of ``interp`` over the first :data:`FORCING_DAYS` days, from ``t0``."""
    days = t0 + jnp.arange(FORCING_DAYS, dtype=float)
    return jax.vmap(interp.evaluate)(days).mean(axis=0)


@dataclass(frozen=True)
class StateLayout:
    """A model's state vector, as far as the warm-up needs to know it.

    ``held`` states keep their start value during the warm-up. ``acids`` lists
    the weak acids of the charge balance as ``(total, ionized, molar mass,
    dissociation constant)``; a layout with acids carries a pH, which the
    warm-up holds at the measured one.
    """

    names: tuple[str, ...]
    held: tuple[str, ...] = ("S_TAN", "S_h2o")
    acids: tuple[tuple[str, str, float, str], ...] = ()

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def has_ph(self) -> bool:
        return bool(self.acids)

    @property
    def free(self) -> np.ndarray:
        """1 for the states the warm-up integrates, 0 for the held ones."""
        return np.asarray([0.0 if name in self.held else 1.0 for name in self.names])

    @property
    def seed(self) -> np.ndarray:
        return np.asarray([GENERIC_SEED.get(name, 0.0) for name in self.names])


def _dissociation(params, temperature):
    """K_a of CO2 and of ammonium, and K_w, at ``temperature`` (as in R2/R3)."""
    def van_t_hoff(enthalpy):
        return jnp.exp(-enthalpy / (_R * 100) * (1 / temperature - 1 / 298.15))

    return (params["K_a_co2"] * van_t_hoff(7646), params["K_a_TAN"] * van_t_hoff(51965),
            params["K_w"] * van_t_hoff(55900))


def _set_ions(layout, y, params, temperature, pH):
    """``S_cat`` and ``S_an`` for which the charge balance gives ``pH``.

    The models solve S_H^2 + phi S_H - K_w = 0 with phi = S_cat - S_an + S_nh4/17
    - S_hco3/44 - sum(ionized acid / molar mass). Only the difference of the two
    ions enters, so one of them is zero.
    """
    _, _, K_w = _dissociation(params, temperature)
    S_H = 10.0 ** (-pH)
    i = layout.index
    S_nh4 = jnp.clip(y[i("S_TAN")] - y[i("S_nh3")], 0, None)
    net = K_w / S_H - S_H - S_nh4 / 17 + y[i("S_hco3_i")] / 44
    for _, ion, molar_mass, _ in layout.acids:
        net = net + y[i(ion)] / molar_mass
    y = y.at[i("S_cat")].set(jnp.maximum(net, 0.0))
    return y.at[i("S_an")].set(jnp.maximum(-net, 0.0))


def _equilibrate(layout, y, params, temperature, pH):
    """The weak acids, CO2 and ammonia in equilibrium at ``pH``, and the ions."""
    K_a_co2, K_a_TAN, _ = _dissociation(params, temperature)
    S_H = 10.0 ** (-pH)
    i = layout.index
    y = y.at[i("S_hco3_i")].set(K_a_co2 * y[i("S_IC")] / (K_a_co2 + S_H))
    y = y.at[i("S_nh3")].set(K_a_TAN * y[i("S_TAN")] / (K_a_TAN + S_H))
    for total, ion, _, constant in layout.acids:
        y = y.at[i(ion)].set(params[constant] * y[i(total)] / (params[constant] + S_H))
    return _set_ions(layout, y, params, temperature, pH)


def measured_chemistry(layout, y, params, temperature, pH):
    """``y`` with its acid-base chemistry set from the first measurements (R2, R3 and
    the full ADM1).

    The inorganic carbon is the dissolved CO2 in equilibrium with the head space
    (``S_gas_co2``, which the datasets set from the first gas measurement) plus the
    bicarbonate that goes with it at ``pH``. The weak acids, bicarbonate and ammonia
    are in equilibrium at ``pH``, and the ions close the charge balance. The model
    then starts at the measured pH with nothing to exchange: no CO2 to absorb or
    release, no acid to dissociate. Of the chemistry in ``y`` only the measured totals
    (TAN, the acids) and the gas phase are read.
    """
    K_a_co2, _, _ = _dissociation(params, temperature)
    S_H = 10.0 ** (-pH)
    i = layout.index
    co2 = dissolved_co2_at_equilibrium(y[i("S_gas_co2")] * _R * temperature / 44, temperature)
    y = y.at[i("S_IC")].set(co2 * (K_a_co2 + S_H) / S_H)
    return _equilibrate(layout, y, params, temperature, pH)


def start_measurements(components, measured, output, targets):
    """What a dataset measured at the start, for the warm-up initialisation.

    ``measured`` marks the entries of the initial values taken from the first
    measurement: the ``measured`` components, the two gas states, and the dissolved
    methane in equilibrium with them, which
    :func:`adm1_package.utils.gas.correct_gas_initial_values` derives from the first gas
    measurement. ``pH`` is the first measured pH, or NaN where the outputs carry none.
    Both are arrays, so they batch and trace like the rest of the sample.

    Where the outputs include the total and volatile solids, ``TS`` and ``VS`` are their
    first measurements, NaN if there is none: where the models' solids start
    (:mod:`adm1_package.utils.solids`).
    """
    indices = [components.index(name) for name in measured]
    indices += [len(components) - 2, len(components) - 1]
    if "S_ch4" in components:
        indices.append(components.index("S_ch4"))
    mask = jnp.zeros(len(components)).at[jnp.asarray(indices)].set(1.0)
    pH = targets[0, output.index("pH")] if "pH" in output else jnp.nan
    start = {"measured": mask, "pH": jnp.asarray(pH, dtype=float)}
    for name in ("TS", "VS"):
        if name in output:
            start[name] = jnp.asarray(targets[0, output.index(name)], dtype=float)
    return start


def warm_up(layout, rhs, args_with, params, initial, start, forcing, t0):
    """The model's state after :data:`WARMUP_DAYS` days at ``forcing``.

    ``rhs`` is the model's right-hand side and ``args_with(forcing)`` its
    arguments around a forcing. ``start`` is what the dataset measured at the
    start: ``measured`` marks the entries of ``initial`` that come from
    measurements, and ``pH`` is the first measured pH. The unmeasured entries of
    ``initial`` are not read.
    """
    temperature, pH = forcing.value[0], start["pH"]
    y = jnp.where(start["measured"] > 0, initial, jnp.asarray(layout.seed))
    if layout.has_ph:
        y = eqx.error_if(y, jnp.isnan(pH), "the warm-up needs the first measured pH")
        y = _equilibrate(layout, y, params, temperature, pH)
    free = jnp.asarray(layout.free)

    def pinned(y, params, temperature, pH):
        return _set_ions(layout, y, params, temperature, pH) if layout.has_ph else y

    def warm_rhs(t, y, args):
        model_args, params, temperature, pH, t0 = args
        # The forcing is constant, and the clock stands at t0: a network that
        # reads time sees the start throughout.
        return free * rhs(t0, pinned(y, params, temperature, pH), model_args)

    solution = dx.diffeqsolve(
        dx.ODETerm(warm_rhs),
        make_solver(),
        t0=0.0,
        t1=WARMUP_DAYS,
        dt0=DT0,
        y0=y,
        args=(args_with(forcing), params, temperature, pH, t0),
        saveat=dx.SaveAt(t1=True),
        stepsize_controller=make_stepsize_controller(),
        max_steps=MAX_STEPS,
        throw=True,
    )
    return pinned(solution.ys[-1], params, temperature, pH)


def warm_start(layout, rhs, args_with, params, initial, input_data):
    """The initial state from a warm-up, with the first measurements put back.

    ``input_data`` is the dataset's input dict: the forcing, and under
    ``start`` what was measured at the start (see :func:`warm_up`).
    """
    ts = input_data["time"]
    t0 = ts[0]
    interp = dx.CubicInterpolation(ts, input_data["coeffs"])
    forcing = ConstantForcing(first_week_forcing(interp, t0))
    start = input_data["start"]
    y = warm_up(layout, rhs, args_with, params, initial, start, forcing, t0)
    y = jnp.where(start["measured"] > 0, initial, y)
    if layout.has_ph:
        y = _equilibrate(layout, y, params, interp.evaluate(t0)[0], start["pH"])
    return y
