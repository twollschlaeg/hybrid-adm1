"""One description per ADM1 variant, used by training, tuning and evaluation.

Anything that describes *what a model variant is* lives here exactly once, so
that training, tuning and evaluation cannot drift apart, e.g. in whether the
rice-bran replicates are averaged or which outputs are scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = ["ModelSpec", "MODEL_KEYS", "get_spec", "DATASETS"]

#: Experimental campaigns used in the study.
DATASETS = ["LaRA-temp", "LaRA-substrate", "rice-bran"]

MODEL_KEYS = ("r2", "r3", "r4")

#: Total and volatile solids, the diagnostics of every reduced variant; the same
#: names as ``adm1_package.utils.solids.SOLIDS``, which this module cannot import
#: without pulling in JAX.
SOLIDS = ("TS", "VS")


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build, train and score one ADM1 variant."""

    key: str
    #: Name used in tables and figures.
    display_name: str
    #: Every quantity the dataset loads and the evaluation can score.
    outputs: tuple[str, ...]
    #: Loss weights. Entries at 0.0 are loaded and scored but not fitted.
    objective_weights: dict[str, float]
    #: Neural-network output width, fixed by the UDE right-hand side.
    nn_out_size: int
    #: Objectives reported per model in the paper's tables.
    reported_objectives: tuple[str, ...]
    #: Influent species carried in the forcing signal, in column order.
    influent_species: tuple[str, ...]
    #: Kinetic parameters the network corrects, in the order the UDE stacks them
    #: into ``solution["dynamic_rates"]``. Taken from the ``jnp.stack`` call at
    #: the end of each ``ude.py``; the interpretability figure indexes that array
    #: by position, so a wrong order here mislabels a curve with no error raised.
    rate_names: tuple[str, ...] = ()
    #: Influent degradability factors, in the order the network emits them into
    #: ``solution["nutrient_factors"]``. The same three for every variant.
    factor_names: tuple[str, ...] = ("f_ch", "f_pr", "f_li")
    #: Measured quantities that are scored and drawn but never fitted: the total and
    #: volatile solids, which the models compute from their particulates, biomass and
    #: the feed's inerts and ash (utils/solids.py). Only ``dataset(diagnostics=True)``
    #: loads them. Training leaves them out, so its targets, and the output noise drawn
    #: for them, stay as they were.
    diagnostics: tuple[str, ...] = ()
    #: Extra keyword arguments for the dataset constructor.
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    #: Options passed to the mechanistic and hybrid simulators.
    mech_options: dict[str, Any] = field(default_factory=dict)
    ude_options: dict[str, Any] = field(default_factory=dict)
    #: Class names where they differ from ``ADM1{KEY}`` and ``{KEY}Dataset``.
    class_names: dict[str, str] = field(default_factory=dict)

    # -- factories (imported lazily so `--help` does not pull in JAX) -------
    def dataset(self, data_dir: str | Path, diagnostics: bool = False, **overrides: Any):
        """Build the canonical dataset for this variant.

        ``diagnostics`` appends :attr:`diagnostics` to the loaded outputs, for
        evaluation and figures; a measurement that the data do not hold is masked.
        """
        factory = self._import("dataset")
        kwargs = {
            "root_dir": str(data_dir),
            "data_to_use": list(DATASETS),
            "output": list(self.outputs) + (list(self.diagnostics) if diagnostics else []),
            **self.dataset_kwargs,
            **overrides,
        }
        return factory(**kwargs)

    def mechanistic(self, *, initial_state=None, arrhenius: bool = False):
        """Build the mechanistic simulator.

        ``initial_state`` is where a simulation starts, ``"calibrated"`` or
        ``"warmup"`` (see :mod:`adm1_package.utils.initial_state`); ``None`` is
        the calibrated start. ``arrhenius`` makes the rates of
        :func:`~adm1_package.utils.corrections.activation_energies` temperature
        dependent, as the full ADM1's are: the parameters are then rates at 35 C.
        """
        options = dict(self.mech_options)
        _set_initial_state(options, initial_state)
        options["arrhenius"] = bool(arrhenius)
        return self._import("mech")(options=options)

    def ude(self, correction=None, *, initial_state=None):
        """Build the hybrid simulator.

        ``correction`` selects how network outputs become parameters -- a
        :class:`~adm1_package.utils.corrections.CorrectionOptions`, its dict
        form, or ``None`` for the default. ``initial_state`` is as for
        :meth:`mechanistic`.
        """
        from adm1_package.utils.corrections import CorrectionOptions

        options = dict(self.ude_options)
        options["correction"] = CorrectionOptions.from_dict(correction)
        _set_initial_state(options, initial_state)
        return self._import("ude")(options=options)

    def correction_layout(self, correction=None):
        """Which network outputs the UDE reads under ``correction``."""
        from adm1_package.utils.corrections import layout_for

        return layout_for(self.key, correction)

    def trainer_class(self):
        """The trainer, shared by every variant."""
        from adm1_package.utils.ude_trainer import UDETrainer

        return UDETrainer

    @property
    def objective_list(self) -> list[str]:
        """Objectives that actually enter the loss (positive weight)."""
        return [name for name in self.outputs if self.objective_weights.get(name, 0.0) > 0.0]

    @property
    def drawn_outputs(self) -> tuple[str, ...]:
        """What figures and inspections show: the outputs, then the diagnostics."""
        return tuple(self.outputs) + tuple(self.diagnostics)

    def nn_in_size(self, dataset) -> int:
        """Network input width, read off the dataset rather than hardcoded.

        The input vector is ``[temperature, inflow / V_liq, *y_in]``, exactly as
        long as the normalization statistics the dataset computes. Deriving it
        here stops the value from drifting away from the data.
        """
        return int(dataset[0][0]["input"]["means"].shape[0])

    def _import(self, what: str) -> Callable:
        import importlib

        module = importlib.import_module(f"adm1_package.models.{self.key}")
        variant = self.key.upper()
        name = self.class_names.get(what) or {
            "dataset": f"{variant}Dataset",
            "mech": f"ADM1{variant}",
            "ude": f"ADM1{variant}UDE",
        }[what]
        return getattr(module, name)


def _set_initial_state(options: dict[str, Any], initial_state: str | None) -> None:
    if initial_state is not None:
        from adm1_package.utils.initial_state import resolve_initial_state

        options["initial_state"] = resolve_initial_state(initial_state)


_SPECS: dict[str, ModelSpec] = {
    "r2": ModelSpec(
        key="r2",
        display_name="ADM1-R2",
        outputs=("q_gasout", "ch4_out", "S_ac", "S_va", "S_bu", "S_pro", "S_TAN", "S_h2o", "pH"),
        objective_weights={
            "q_gasout": 2.0,   # biogas production rate -- primary objective
            "ch4_out": 2.0,    # methane production rate -- primary objective
            "S_ac": 1.5,       # acetic acid, a process-stability indicator
            "S_va": 0.0,       # valeric acid: mostly below the detection limit
            "S_bu": 0.0,       # butyric acid: mostly below the detection limit
            "S_pro": 1.0,      # propionic acid
            "S_TAN": 1.5,      # total ammoniacal nitrogen
            # Water content is scored, not fitted: it varies little within a reactor and
            # the models miss it by far more than it varies, so scaled by its range
            # it outweighs gas and TAN together and drives the fit.
            "S_h2o": 0.0,
            "pH": 1.0,
        },
        nn_out_size=15,
        reported_objectives=("q_gasout", "ch4_out", "S_ac", "S_pro", "S_TAN"),
        influent_species=(
            "S_va", "S_bu", "S_pro", "S_ac", "S_TAN", "S_h2o", "X_ch", "X_pr", "X_li",
        ),
        rate_names=(
            "k_hyd_ch", "k_hyd_pr", "k_hyd_li",
            "k_m_va", "k_m_bu", "k_m_pro", "k_m_ac",
            "k_dec_Xbac", "k_dec_Xva", "k_dec_Xbu", "k_dec_Xpro", "k_dec_Xac",
        ),
        diagnostics=SOLIDS,
        ude_options={},
        mech_options={},
    ),
    "r3": ModelSpec(
        key="r3",
        display_name="ADM1-R3",
        outputs=("q_gasout", "ch4_out", "S_ac", "S_TAN", "S_h2o", "pH"),
        objective_weights={
            "q_gasout": 2.0,
            "ch4_out": 2.0,
            "S_ac": 1.5,
            "S_TAN": 1.5,
            "S_h2o": 0.0,      # scored, not fitted (see R2)
            "pH": 1.0,
        },
        nn_out_size=9,
        reported_objectives=("q_gasout", "ch4_out", "S_ac", "S_TAN"),
        influent_species=("S_ac", "S_TAN", "S_h2o", "X_ch", "X_pr", "X_li"),
        rate_names=(
            "k_hyd_ch", "k_hyd_pr", "k_hyd_li", "k_m_ac", "k_dec_Xbac", "k_dec_Xac",
        ),
        diagnostics=SOLIDS,
    ),
    "r4": ModelSpec(
        key="r4",
        display_name="ADM1-R4",
        outputs=("q_gasout", "ch4_out", "S_TAN", "S_h2o"),
        objective_weights={
            "q_gasout": 2.0,
            "ch4_out": 2.0,
            "S_TAN": 1.5,
            "S_h2o": 0.0,      # scored, not fitted (see R2)
        },
        nn_out_size=7,
        reported_objectives=("q_gasout", "ch4_out", "S_TAN"),
        influent_species=("S_TAN", "S_h2o", "X_ch", "X_pr", "X_li"),
        rate_names=("k_hyd_ch", "k_hyd_pr", "k_hyd_li", "k_dec_Xbac"),
        diagnostics=SOLIDS,
        ude_options={},
    ),
}

#: The full ADM1, which the variants reduce. It is not in MODEL_KEYS: the study does not
#: train or score it. scripts/calibrate/base_calibration.py fits it after R2, on R2's
#: objectives, with the benchmarks' two stages.
_SPECS["base"] = ModelSpec(
    key="base",
    display_name="ADM1",
    outputs=_SPECS["r2"].outputs,
    objective_weights=dict(_SPECS["r2"].objective_weights),
    nn_out_size=0,
    reported_objectives=_SPECS["r2"].reported_objectives,
    influent_species=_SPECS["r2"].influent_species,
    class_names={"dataset": "BaseDataset", "mech": "ADM1Base"},
)


def get_spec(key: str) -> ModelSpec:
    """Look up a variant by key (``r2``, ``r3``, ``r4``, or ``base`` for the full ADM1;
    case insensitive)."""
    normalized = key.strip().lower()
    if normalized not in _SPECS:
        raise KeyError(f"unknown model {key!r}; expected one of {', '.join(_SPECS)}")
    return _SPECS[normalized]


