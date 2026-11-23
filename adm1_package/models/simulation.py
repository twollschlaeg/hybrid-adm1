"""The scaffolding every ADM1 variant shares: options, the solve and the reported output.

A variant (``models/<variant>/model.py``) holds its state vector (``STATES``), its
Petersen matrix and its mechanistic right-hand side ``_rhs(t, y, (params, forcing))``; its
hybrid (``ude.py``) holds a right-hand side of its own, ``_rhs(t, y, (params, forcing,
mlp, key, scaling))``, whose rates the network corrects. Both write ``_calculate_output``,
the algebraic outputs along a solution. Everything around them lives here:

* :class:`Simulator` -- the options and the solve: the forcing interpolation, the
  optional warm-up (:mod:`adm1_package.utils.initial_state`) and ``diffeqsolve`` at the
  solver settings of :mod:`adm1_package.utils.solver`;
* :class:`Mechanistic` and :class:`Hybrid` -- ``__call__`` for the two kinds, the total
  and volatile solids after the solve (:mod:`adm1_package.utils.solids`), and the hybrid's
  network evaluation and reported corrections;
* :func:`unpack` -- the state vector by name, for the right-hand sides;
* :func:`solution_arrays`, :func:`column`, :func:`columns`, :func:`state_output` and
  :func:`biogas` -- the pieces of every ``_calculate_output``.

The numerical settings in which the variants differ are class attributes, set and
documented by each variant: ``STATE_BOUNDS``, the interval the right-hand side clips the
states to, and ``BOUNDED``, whether it also clips the proton concentration, the
head-space pressure and the outflow (:mod:`adm1_package.models.chemistry`). The stored
calibrations and runs were made with these, so they stay as they are:

==========================  ===============  =========
variant                     STATE_BOUNDS     BOUNDED
==========================  ===============  =========
R4, mechanistic and hybrid  (0, 1e10)        no
R3, mechanistic             (0, 1e10)        no
R3, hybrid                  (1e-12, 1e8)     yes
R2, mechanistic and hybrid  (1e-12, 1e8)     yes
full ADM1                   (1e-12, 1e8)     yes
==========================  ===============  =========

The reported gas flows use the same clipped states as the right-hand side; the reported
pH uses the unclipped ones. Further differences are noted where they live: the mechanistic
R2 decays X_va and X_bu at one rate, ``k_dec_Xc4``, its hybrid at two; only the hybrids
know the TAN-inflow correction ``f_TAN_in``; the full ADM1's growth always depends on
temperature (its ``arrhenius`` option is a no-op).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar, Mapping

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from adm1_package.utils.corrections import CorrectionLayout, layout_for, with_arrhenius
from adm1_package.utils.gas import biogas_outflow
from adm1_package.utils.initial_state import StateLayout, resolve_initial_state, warm_start
from adm1_package.utils.nn_architectures import network_input
from adm1_package.utils.solids import biomass, solids
from adm1_package.utils.solver import DT0, MAX_STEPS, make_solver, make_stepsize_controller

__all__ = ["Simulator", "Mechanistic", "Hybrid", "solution_arrays", "unpack", "column",
           "columns", "state_output", "biogas"]


class Simulator(eqx.Module):
    """Options and the solve, shared by every mechanistic and hybrid ADM1 variant.

    ``options`` (all optional):

    ``initial_state``
        ``"calibrated"`` (the default) or ``"warmup"``
        (:mod:`adm1_package.utils.initial_state`).
    ``arrhenius``
        Mechanistic R2-R4: the rates of
        :func:`~adm1_package.utils.corrections.activation_energies` at the reactor's
        temperature, the parameters being rates at 35 C.
    ``correction``
        Hybrids: how network outputs become parameters
        (:class:`~adm1_package.utils.corrections.CorrectionOptions`).

    A variant adds its own in ``DEFAULT_OPTIONS``.
    """

    options: dict

    #: The variant's key in the registry and in utils/corrections.py.
    VARIANT: ClassVar[str]
    #: The state vector.
    STATES: ClassVar[StateLayout]
    #: Options a variant fills in where the caller gives none.
    DEFAULT_OPTIONS: ClassVar[Mapping[str, Any]] = {}
    #: The interval the right-hand side clips the states to before it reads them.
    STATE_BOUNDS: ClassVar[tuple[float, float]]
    #: Whether the right-hand side also clips the proton concentration, the head-space
    #: pressure and its outflow (chemistry.S_H_BOUNDS, P_GAS_BOUNDS, Q_GAS_BOUNDS).
    BOUNDED: ClassVar[bool]
    #: Whether the solution reports the total and volatile solids (utils/solids.py).
    REPORTS_SOLIDS: ClassVar[bool] = True

    def __init__(self, options: Mapping[str, Any] | None = None):
        self.options = {**self.DEFAULT_OPTIONS, **(options or {})}

    def _rhs(self, t, y, args):
        raise NotImplementedError

    def _calculate_output(self, sol: dx.Solution, args: tuple) -> dict:
        raise NotImplementedError

    def _solve(self, initial: Float[Array, " states"], input_data: dict, args: tuple,
               args_with) -> dx.Solution:
        """The solution at the forcing's time points, from ``initial`` or, with the
        ``warmup`` option, from the model's own warm-up. ``args_with(forcing)`` builds the
        right-hand side's arguments around another forcing."""
        ts = input_data['time']
        if resolve_initial_state(self.options.get('initial_state')) == 'warmup':
            initial = warm_start(self.STATES, self._rhs, args_with, args[0], initial, input_data)
        # Solver and tolerances live in utils/solver.py, so that every variant and its
        # hybrid integrate at the same accuracy.
        return dx.diffeqsolve(
            terms=dx.ODETerm(self._rhs),
            solver=make_solver(),
            t0=ts[0],
            t1=ts[-1],
            dt0=DT0,
            y0=initial,
            args=args,
            saveat=dx.SaveAt(ts=ts),
            stepsize_controller=make_stepsize_controller(),
            max_steps=MAX_STEPS,
            throw=True,
            adjoint=dx.RecursiveCheckpointAdjoint(),
        )


class Mechanistic(Simulator):
    """A mechanistic variant: ``model(params, initial, input_data)``."""

    def _at_temperature(self, params, T_op):
        """``params`` with the Arrhenius option applied (utils/corrections.py)."""
        if self.options.get('arrhenius', False):
            # The rates, calibrated at 35 C, at the reactor's temperature; the hybrids'
            # physics option uses the same factors.
            return with_arrhenius(params, T_op, self.VARIANT)
        return params

    def __call__(self, params: dict, initial: Float[Array, " states"],
                 input_data: dict) -> tuple[dict, dx.Solution]:
        '''Solve the model ODE.

        ``params`` are the model parameters, ``initial`` the initial state and
        ``input_data`` the sample's forcing (``time``, ``coeffs`` and, for the warm-up,
        ``start``). Returns the solution with its algebraic components, inhibition terms
        and solids as a dict, and the diffrax solution.
        '''
        interp = dx.CubicInterpolation(input_data['time'], input_data['coeffs'])
        args = (params, interp)
        sol = self._solve(initial, input_data, args, lambda forcing: (params, forcing))
        sol_dic = self._calculate_output(sol, args)
        if self.REPORTS_SOLIDS:
            # The total and volatile solids, after the solve: the feed's inerts and ash
            # only wash out.
            sol_dic.update(solids(sol_dic, input_data, params, params, biomass(self.STATES.names)))
        return sol_dic, sol


class Hybrid(Simulator):
    """A hybrid variant: ``model(params, initial, input_data, mlp, key)``.

    The network reads the z-scored forcing (utils/nn_architectures.py) and its outputs
    become parameters through the variant's correction layout (utils/corrections.py).
    """

    def _corrections(self) -> CorrectionLayout:
        '''How network outputs become parameters (utils/corrections.py).'''
        return layout_for(self.VARIANT, self.options.get('correction'))

    @staticmethod
    def _network(T_op, q_in, y_in, params, mlp, key, scaling):
        """The raw network output at one point of the forcing."""
        scaled_input = network_input(T_op, q_in / params['V_liq'], y_in, scaling)
        return mlp(scaled_input, key)

    def _corrections_along(self, forcing, params, mlp, key, scaling):
        """The temperature along a solution, and what the network set there: the layout,
        the corrected parameters, the raw output and the degradabilities, (T, 3).
        ``forcing`` is the forcing at the solution's times."""
        T_op = forcing[:,0]
        q_in = forcing[:,1]
        y_in = forcing[:,2:]
        scaled_input = network_input(T_op, q_in / params['V_liq'], y_in, scaling)
        mlp_out = jax.vmap(mlp, in_axes=(0, None))(scaled_input, key)
        # The same mapping the right-hand side used, so the reported rates are the
        # ones that were integrated.
        layout = self._corrections()
        corrected = layout.resolve(mlp_out, params, T_op)
        degradabilities = jnp.stack([corrected[name] for name in layout.factor_names], axis=1)
        return T_op, (layout, corrected, mlp_out, degradabilities)

    @staticmethod
    def _network_output(layout, corrected, mlp_out, degradabilities) -> dict:
        """What the network did, for the figures and the trainer."""
        return {
            'nutrient_factors': degradabilities,
            # Every rate, network-driven or held fixed, in registry order, so the
            # interpretability figure can index this by ModelSpec.rate_names.
            'dynamic_rates': jnp.stack([corrected[name] for name in layout.rate_names], axis=1),
            **{name: corrected[name] for name in layout.extra_names},
            # Before the link: what the trainer's correction penalty measures.
            'network_output': mlp_out,
        }

    def __call__(self, params: dict, initial: Float[Array, " states"], input_data: dict,
                 mlp, key: jax.Array) -> tuple[dict, dx.Solution]:
        '''Solve the model ODE with the network ``mlp`` (dropout ``key``).

        As :meth:`Mechanistic.__call__`; ``input_data`` also carries the network's input
        scaling (``means``, ``stds``). The solution adds the network's output.
        '''
        interp = dx.CubicInterpolation(input_data['time'], input_data['coeffs'])
        scaling = (input_data['means'], input_data['stds'])
        args = (params, interp, mlp, key, scaling)
        # The network is part of the model, so it shapes its own warm-up.
        sol = self._solve(initial, input_data, args,
                          lambda forcing: (params, forcing, mlp, key, scaling))
        sol_dic = self._calculate_output(sol, args)
        # The total and volatile solids, after the solve, with the degradabilities the
        # network set.
        factors = dict(zip(self._corrections().factor_names,
                           jnp.moveaxis(sol_dic['nutrient_factors'], -1, 0)))
        sol_dic.update(solids(sol_dic, input_data, params, factors, biomass(self.STATES.names)))
        return sol_dic, sol


def solution_arrays(sol: dx.Solution):
    """The saved times and states of a solution."""
    if sol.ys is None:
        raise ValueError("Solution 'y' is None. Ensure the ODE solver returned a valid solution.")
    if sol.ts is None:
        raise ValueError("Solution 't' is None. Ensure the ODE solver returned a valid solution.")
    return sol.ts, sol.ys


def unpack(states: StateLayout, y) -> SimpleNamespace:
    """The entries of a state vector by name, ``s.S_ac``."""
    return SimpleNamespace(**{name: y[i] for i, name in enumerate(states.names)})


def column(states: StateLayout, y, name: str):
    """One state along a solution (time first)."""
    return y[:,states.index(name)]


def columns(states: StateLayout, y, *names: str) -> SimpleNamespace:
    """The named states along a solution, by name, read in the order given."""
    return SimpleNamespace(**{name: column(states, y, name) for name in names})


def state_output(t, y, states: StateLayout, read: SimpleNamespace | None = None) -> dict:
    """The time and every state along a solution, in state order.

    ``read`` holds columns an output has already read (:func:`columns`), which are
    reported as they are; the others are read here.

    An output reads its states in the order its algebra has always read them, and only
    then reports them: the values would be the same in any order, but the order of the
    operations decides how XLA fuses them, and with it the last bits of the gradients.
    """
    read = vars(read) if read is not None else {}
    return {'time': t, **{name: read[name] if name in read else column(states, y, name)
                          for name in states.names}}


def biogas(y, t, interp, T_op, params, partial_pressures, p_h2o, state_bounds,
           transfer_rates):
    """Biogas and methane outflow in NL/d (``q_gasout``, ``ch4_out``) along a solution.

    From the head-space mole balance rather than k_p (p_gas - p_norm), which amplifies
    the solver's error in p_gas (utils/gas.py). The gas transfer rates are the
    right-hand side's: ``transfer_rates(clipped, T_op, params)`` takes the states clipped
    to the right-hand side's ``state_bounds``. ``partial_pressures`` are p_ch4 and p_co2,
    and for the full ADM1 also p_h2 (its transfer rates then include H2's).
    """
    clipped = jnp.clip(y, *state_bounds)
    rates = transfer_rates(clipped, T_op, params)
    dT_dt = jax.vmap(interp.derivative)(t)[:,0]
    p_ch4, p_co2, *p_h2 = partial_pressures
    hydrogen = {'p_h2': p_h2[0], 'rho_h2': rates[2]} if p_h2 else {}
    return biogas_outflow(p_ch4, p_co2, p_h2o, T_op, dT_dt, rates[0], rates[1], params,
                          **hydrogen)
