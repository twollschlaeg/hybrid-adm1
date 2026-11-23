"""Single source of truth for the ODE solver settings.

Every ADM1 variant and every UDE in this package integrates with the same
scheme and the same tolerances. Keeping them here matters for two reasons:

* **reproducibility** -- the paper names a concrete solver and its tolerances,
  and there must be one place to read them off;
* **fair comparison** -- the mechanistic benchmark and its hybrid counterpart
  have to be integrated at the same accuracy. Otherwise part of the measured
  difference between model variants could be integration error rather than
  model quality.

Override the defaults through the environment when profiling or when running on
an accelerator, e.g. ``ADM1_SOLVER_RTOL=1e-6 ADM1_SOLVER_ATOL=1e-7``.
"""

from __future__ import annotations

import os

import diffrax as dx

__all__ = [
    "RTOL",
    "ATOL",
    "PCOEFF",
    "ICOEFF",
    "DCOEFF",
    "MAX_STEPS",
    "DT0",
    "make_solver",
    "make_stepsize_controller",
    "solver_settings",
]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


#: Relative and absolute tolerance of the adaptive step size controller.
RTOL: float = _env_float("ADM1_SOLVER_RTOL", 1e-4)
ATOL: float = _env_float("ADM1_SOLVER_ATOL", 1e-5)

#: PID controller gains.
PCOEFF: float = _env_float("ADM1_SOLVER_PCOEFF", 0.2)
ICOEFF: float = _env_float("ADM1_SOLVER_ICOEFF", 0.3)
DCOEFF: float = _env_float("ADM1_SOLVER_DCOEFF", 0.0)

#: Initial step size and step budget per solve.
DT0: float = _env_float("ADM1_SOLVER_DT0", 1e-3)
MAX_STEPS: int = int(_env_float("ADM1_SOLVER_MAX_STEPS", 1e6))


def make_solver() -> dx.AbstractSolver:
    """Kvaerno3: a 3rd-order, L-stable ESDIRK (implicit Runge-Kutta) scheme.

    The ADM1 right-hand sides are stiff (gas transfer and hydrolysis act on very
    different time scales), so an implicit scheme is required.
    """
    return dx.Kvaerno3()


def make_stepsize_controller(
    rtol: float | None = None, atol: float | None = None
) -> dx.AbstractStepSizeController:
    """Adaptive PID controller wrapped in diffrax's step clipping."""
    return dx.ClipStepSizeController(
        controller=dx.PIDController(
            rtol=RTOL if rtol is None else rtol,
            atol=ATOL if atol is None else atol,
            pcoeff=PCOEFF,
            icoeff=ICOEFF,
            dcoeff=DCOEFF,
        )
    )


def solver_settings() -> dict[str, float | int | str]:
    """The settings as a plain dict, for logging into run metadata."""
    return {
        "solver": "diffrax.Kvaerno3 (ESDIRK, implicit Runge-Kutta, order 3)",
        "rtol": RTOL,
        "atol": ATOL,
        "pcoeff": PCOEFF,
        "icoeff": ICOEFF,
        "dcoeff": DCOEFF,
        "dt0": DT0,
        "max_steps": MAX_STEPS,
        "stepsize_controller": "diffrax.ClipStepSizeController(diffrax.PIDController)",
    }
