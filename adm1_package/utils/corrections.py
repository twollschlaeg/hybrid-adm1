"""How the correction network's outputs become model parameters.

The plain mapping, hardcoded in each right-hand side as

    k = clip(literature + network_output, 1e-10, 10)
    f = sparse_sigmoid(network_output)

is what a structural probe of trained models found to be the main thing holding
the hybrid back (see below). This module describes the mapping once, for all
variants.

Parametrization
---------------
``calibrated``
    A multiplicative link around the reactor's own *calibrated* value::

        k = k_calibrated * exp(b + NN(x))
        f = sigmoid(logit(f_calibrated) + b + NN(x))

    ``b`` is the network's output-layer bias, one trainable offset per parameter.
    At zero network output this is exactly the mechanistic benchmark, so the
    network only has to learn where the mechanism is wrong. A parameter the
    reactor's calibration lacks starts from the target's reference value.

The ``global`` parametrization, literature anchors with a learned global base for
reactors that were never calibrated, was retired before this release (see
``experiments/retired.py``). A record that names it is refused.

Why the plain mapping holds the models back
---------------------------------------------
Measured on trained R2/R3/R4 folds:

* ``clip(anchor + x, 1e-10, 10)`` has zero gradient below the floor. A decay
  rate anchored at 0.02 that is pushed below zero never recovers; several were
  clipped 100% of the time.
* ``sparse_sigmoid`` is flat outside (-1, 1): its gradient there is exactly 0.
  Lipid degradability sat pinned at 1.0 in R3 for 68% of the trajectory.
* Additive anchors span two orders of magnitude (0.02 to 2.0), so one shared
  output layer has to produce corrections of very different sizes. A
  multiplicative link makes every output a relative change.

Pruning
-------
Some outputs had no measurable effect on any fitted observable, for reasons
that are structural rather than accidental:

* protein and lipid hydrolysis are saturated -- ``k/(k+D)`` is about 0.85-0.99 at
  the study's dilution rate of 0.033 1/d -- so the steady state does not depend on
  them;
* the valerate and butyrate uptake rates feed objectives whose loss weight is
  zero, so no gradient reaches them;
* lipid degradability calibrates to ~1.0 in every campaign;
* the biomass decay rates for the lumped acidogens (and, in R2, for valerate and
  butyrate degraders) were clipped dead.

With ``prune=True`` those are held at their reference value instead of being
network outputs. That removes degrees of freedom that could only overfit, which
matters with twelve training reactors. Decay rates that did move an observable
(acetate and propionate degraders) are kept.

Narrow corrections
------------------
``corrected`` names the parameters the network changes; every other one is held,
under ``calibrated`` at the reactor's calibrated value, and the list overrides
``prune``. ``bound`` squashes each change smoothly into ``(-bound, bound)``: a
log-change for rates (``bound=log 2`` keeps them within a factor of two of their
anchor) and a logit-change for degradability factors, which are otherwise
unbounded. Without ``bound``, rates keep the wide ``LOG_BOUND`` below.

``exact_anchors`` makes "at zero output it is the benchmark" hold for every reactor.
A factor's anchor is kept inside ``FRACTION_BAND`` before the logit, so one that
calibrates outside it (a degradability at or close to 1) starts at the band's edge,
and 1% less degradable protein moves the gas of some reactors noticeably. With the
option the cut-off part is added back as an offset. The network moves the factor
exactly as before, so values may exceed 1 by at most 0.01. Off by default, so that
runs trained without it still reproduce.

``corrected`` and ``bound`` answer what calibrated hybrids with wide corrections did
on reactors they had not trained on. Their biogas and methane improved. But R2/R3
raised acetoclast decay many-fold, acetate blew up, and TAN followed the protein
degradability. Correcting only the carbohydrate hydrolysis, close to the
calibration, keeps the network to what transferred.

Additional physics
------------------
``arrhenius``
    Temperature dependence of the biological rates. The full ADM1 in this
    package applies an Arrhenius factor to every uptake rate; the reduced
    variants dropped it, so their biology does not respond to temperature at
    all, and the probe found the network gives temperature only a small share
    of its input attribution. Activation energies are the Weinrich (2017) values the
    full model uses. Hydrolysis in the reduced models lumps the acidogenic step
    behind it, so ch/pr/li hydrolysis take the sugar, amino-acid and
    fatty-acid energies. Note that Arrhenius alone is monotonic: it cannot
    produce the drop in biogas that follows a fast 37->55 C shift. That needs
    inhibition, hence the next option. The mechanistic models have the same
    option (``spec.mechanistic(arrhenius=True)``, :func:`with_arrhenius`), so a
    calibrated hybrid with this physics is, at zero output, its mechanistic model
    with it; their rates are then rates at 35 C.
``nh3_inhibition``
    Make the free-ammonia inhibition constant ``K_I_nh3`` a network output
    (R2 and R3; R4 has no inhibition term). Free ammonia rises steeply with
    temperature and is the classic thermophilic stressor, and it reaches biogas
    through methanogenesis without going through the TAN stoichiometry.
``tan_inflow``
    A network-corrected factor on the influent TAN concentration. The probe
    found TAN and biogas tied together through the degradability factors, so
    correcting TAN dragged biogas with it; this gives TAN a lever of its own.

Half-saturation constants
-------------------------
The Monod constants of the R2 and R3 uptake rates (``K_ac``; R2 also ``K_pro``,
``K_va``, ``K_bu``) are not network outputs. Each reactor keeps its campaign's own
calibrated value, and these differ between campaigns. A campaign without one
gets the ADM1 default (Batstone et al., 2002) in grams of acid, the value the
campaign calibrations started from.

The network needs no correction of its own on K: in a mechanistic probe, fixing
``K_ac`` at the literature 0.14 g/L and rescaling ``k_m_ac`` (which the network
does correct) on a coarse grid left biogas and methane unchanged and cost a
little acetate RMSE on the LaRA campaigns and nothing on rice-bran. A
``references`` entry such as ``("K_ac", 0.3389)`` moves a constant, e.g. to a
campaign's calibrated value.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import jax.nn as jnn

__all__ = [
    "PARAMETRIZATIONS",
    "PHYSICS",
    "HALF_SATURATION",
    "Target",
    "CorrectionOptions",
    "CorrectionLayout",
    "DEFAULT",
    "layout_for",
    "arrhenius_factor",
    "activation_energies",
    "with_arrhenius",
]

#: How network outputs become parameters; ``global`` was retired (see above).
PARAMETRIZATIONS = ("calibrated",)
PHYSICS = ("arrhenius", "nh3_inhibition", "tan_inflow")

#: Gas constant in J/(mol K), and the reference temperature of the Weinrich
#: activation energies -- the same pair the full ADM1 in models/base uses.
R_GAS = 8.3145
T_REFERENCE = 308.15

#: A log-change larger than this in either direction is squashed smoothly,
#: which keeps exp() finite without the dead gradient a hard clip has. e^3 = 20,
#: comfortably more than the literature-to-calibrated spread (at most ~4x).
LOG_BOUND = 3.0

#: Degradability factors are kept inside this band before taking the logit, so
#: a calibrated value such as 0.9997 does not start the network on sigmoid's
#: nearly flat tail.
FRACTION_BAND = (0.01, 0.99)

# Activation energies in J/mol, Weinrich (2017), as in models/base/model.py.
_E_SUGAR, _E_AMINO, _E_FATTY = 33447.56500805, 13282.43293645, 20165.1320716
_E_VALERATE = _E_BUTYRATE = 16005.96578368
_E_PROPIONATE, _E_ACETATE = 17005.40065541, 27362.37923619


@dataclass(frozen=True)
class Target:
    """One model parameter the network can correct.

    ``reference`` is the literature value a parameter starts from where the
    reactor's calibration lacks it (for degradability factors, the sigmoid
    midpoint 0.5). ``fixed`` is the value held when such a target is pruned; it
    defaults to ``reference``.
    """

    name: str
    kind: str  # "rate" (positive), "fraction" (0..1) or "factor" (positive, ~1)
    reference: float
    activation_energy: float | None = None
    fixed: float | None = None

    @property
    def fixed_value(self) -> float:
        return self.reference if self.fixed is None else self.fixed

    @property
    def key(self) -> str:
        return self.name


_FRACTIONS = (
    Target("f_ch", "fraction", 0.5),
    Target("f_pr", "fraction", 0.5),
    # Calibrates close to 1 in all three campaigns. Complete degradability is
    # also the standard ADM1 assumption for lipids.
    Target("f_li", "fraction", 0.5, fixed=1.0),
)
_HYDROLYSIS = (
    Target("k_hyd_ch", "rate", 0.02, _E_SUGAR),
    Target("k_hyd_pr", "rate", 2.0, _E_AMINO),
    Target("k_hyd_li", "rate", 0.76, _E_FATTY),
)

#: Every correctable parameter per variant, in output order: degradability
#: factors first, then rates. The split network relies on the factors leading.
CATALOGUE: dict[str, tuple[Target, ...]] = {
    "r4": _FRACTIONS + _HYDROLYSIS + (
        Target("k_dec_Xbac", "rate", 0.02),
    ),
    "r3": _FRACTIONS + _HYDROLYSIS + (
        Target("k_m_ac", "rate", 0.65, _E_ACETATE),
        Target("k_dec_Xbac", "rate", 0.02),
        Target("k_dec_Xac", "rate", 0.02),
    ),
    "r2": _FRACTIONS + _HYDROLYSIS + (
        Target("k_m_va", "rate", 0.45, _E_VALERATE),
        Target("k_m_bu", "rate", 0.70, _E_BUTYRATE),
        Target("k_m_pro", "rate", 0.7, _E_PROPIONATE),
        Target("k_m_ac", "rate", 0.65, _E_ACETATE),
        # Valerate and butyrate degraders share k_dec_Xc4 in the mechanistic R2.
        Target("k_dec_Xbac", "rate", 0.02),
        Target("k_dec_Xva", "rate", 0.02),
        Target("k_dec_Xbu", "rate", 0.02),
        Target("k_dec_Xpro", "rate", 0.02),
        Target("k_dec_Xac", "rate", 0.02),
    ),
}

#: Held at their reference value under ``prune=True``. See the module docstring.
PRUNED: dict[str, tuple[str, ...]] = {
    "r4": ("f_li", "k_hyd_pr", "k_hyd_li", "k_dec_Xbac"),
    "r3": ("f_li", "k_hyd_pr", "k_hyd_li", "k_dec_Xbac"),
    "r2": ("f_li", "k_hyd_pr", "k_hyd_li", "k_m_va", "k_m_bu", "k_dec_Xva", "k_dec_Xbu"),
}

#: Physics options and the extra network output each one adds, per variant.
_EXTRAS: dict[str, Target] = {
    "nh3_inhibition": Target("K_I_nh3", "rate", 0.026),
    "tan_inflow": Target("f_TAN_in", "factor", 1.0),
}
_APPLICABLE: dict[str, tuple[str, ...]] = {
    "r4": ("arrhenius", "tan_inflow"),
    "r3": PHYSICS,
    "r2": PHYSICS,
}

#: Monod half-saturation constants in g/L where a calibration lacks them: the ADM1
#: defaults (Batstone et al., 2002) in grams of acid, as in the ADM1+FBA code of
#: Weinrich et al. (2019). See "Half-saturation constants" above.
HALF_SATURATION: dict[str, float] = {
    "K_va": 0.09802,
    "K_bu": 0.1101,
    "K_pro": 0.06614403397270256,
    "K_ac": 0.14075205945222943,
}
#: The constants each variant's uptake rates use; R4 has no uptake step.
_HALF_SATURATION_USED: dict[str, tuple[str, ...]] = {
    "r4": (),
    "r3": ("K_ac",),
    "r2": ("K_va", "K_bu", "K_pro", "K_ac"),
}


@dataclass(frozen=True)
class CorrectionOptions:
    """Which parametrization, pruning and physics a UDE uses.

    Hashable, so it can sit in a static field and key a cache.
    """

    parametrization: str = "calibrated"
    prune: bool = True
    physics: tuple[str, ...] = ()
    #: Per-target overrides of ``Target.reference``, as ``(name, value)`` pairs.
    references: tuple[tuple[str, float], ...] = ()
    #: The parameters the network changes, overriding ``prune``; ``None``: as pruned.
    corrected: tuple[str, ...] | None = None
    #: How far a change may go (log for rates, logit for fractions); ``None``: the
    #: wide default for rates, unbounded fractions. See "Narrow corrections".
    bound: float | None = None
    #: Start a fraction anchored outside FRACTION_BAND at its anchor, not at the band's
    #: edge; ``False`` keeps what earlier runs trained with. See "Narrow corrections".
    exact_anchors: bool = False

    def __post_init__(self):
        if self.parametrization == "global":
            from adm1_package.experiments.retired import retired

            raise retired("the global (literature-anchored) parametrization")
        if self.parametrization not in PARAMETRIZATIONS:
            raise ValueError(
                f"unknown parametrization {self.parametrization!r}; "
                f"expected one of {', '.join(PARAMETRIZATIONS)}"
            )
        unknown = set(self.physics) - set(PHYSICS)
        if unknown:
            raise ValueError(f"unknown physics {sorted(unknown)}; expected any of {PHYSICS}")
        if self.corrected is not None and not self.corrected:
            raise ValueError("corrected names no parameter; leave it out to correct as pruned")
        if self.bound is not None and not self.bound > 0:
            raise ValueError(f"bound must be positive, not {self.bound}")
        # Canonical order, so equal option sets compare and hash equal.
        object.__setattr__(self, "physics", tuple(p for p in PHYSICS if p in self.physics))
        object.__setattr__(self, "references", tuple(sorted(dict(self.references).items())))
        if self.corrected is not None:
            object.__setattr__(self, "corrected", tuple(sorted(set(self.corrected))))
        if self.bound is not None:
            object.__setattr__(self, "bound", float(self.bound))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parametrization": self.parametrization,
            "prune": self.prune,
            "physics": list(self.physics),
            "references": dict(self.references),
            "corrected": None if self.corrected is None else list(self.corrected),
            "bound": self.bound,
            "exact_anchors": self.exact_anchors,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CorrectionOptions":
        """Read options from a recipe or a saved run; ``None`` is the default."""
        if payload is None:
            return DEFAULT
        if isinstance(payload, CorrectionOptions):
            return payload
        corrected = payload.get("corrected")
        return cls(
            parametrization=payload.get("parametrization", "calibrated"),
            prune=bool(payload.get("prune", True)),
            physics=tuple(payload.get("physics", ())),
            references=tuple(dict(payload.get("references", {})).items()),
            corrected=None if corrected is None else tuple(corrected),
            bound=payload.get("bound"),
            exact_anchors=bool(payload.get("exact_anchors", False)),
        )


#: What a run uses unless told otherwise.
DEFAULT = CorrectionOptions()


def arrhenius_factor(temperature, activation_energy: float):
    """``exp(-E/R (1/T - 1/T_ref))``: 1 at 35 C, rising with temperature."""
    return jnp.exp(-activation_energy / R_GAS * (1.0 / temperature - 1.0 / T_REFERENCE))


def activation_energies(variant: str) -> dict[str, float]:
    """The rates the ``arrhenius`` physics scales in ``variant``, with their energies (J/mol).

    The mechanistic models' ``arrhenius`` option reads the same table, so a calibrated
    hybrid with ``physics: arrhenius`` at zero output is its mechanistic model.
    """
    return {target.name: target.activation_energy for target in CATALOGUE[variant.lower()]
            if target.activation_energy is not None}


def with_arrhenius(params: Mapping[str, Any], temperature, variant: str) -> dict[str, Any]:
    """``params`` with every rate of :func:`activation_energies` at ``temperature``.

    The stored rates are then rates at T_REFERENCE (35 C). A new dict; ``params`` is
    left as it is.
    """
    scaled = dict(params)
    for name, energy in activation_energies(variant).items():
        if name in scaled:
            scaled[name] = scaled[name] * arrhenius_factor(temperature, energy)
    return scaled


def _bounded(delta, bound: float = LOG_BOUND):
    """A change squashed smoothly into (-bound, bound)."""
    return bound * jnp.tanh(delta / bound)


def _logit(value):
    low, high = FRACTION_BAND
    value = jnp.clip(value, low, high)
    return jnp.log(value) - jnp.log1p(-value)


@dataclass(frozen=True)
class CorrectionLayout:
    """The network's outputs for one variant under one set of options."""

    variant: str
    options: CorrectionOptions
    #: Every parameter this layout resolves, network-driven or held fixed.
    targets: tuple[Target, ...]
    #: The subset the network produces, in output order.
    outputs: tuple[str, ...]
    #: Physics options that apply to this variant (others are skipped).
    physics: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.outputs)

    @property
    def rate_names(self) -> tuple[str, ...]:
        """Kinetic parameters in the order ``solution["dynamic_rates"]`` stacks them."""
        return tuple(t.name for t in CATALOGUE[self.variant] if t.kind == "rate")

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in CATALOGUE[self.variant] if t.kind == "fraction")

    @property
    def n_factor_outputs(self) -> int:
        """How many outputs, at the front, are degradability factors: what the
        factor network of a split network produces."""
        count = 0
        for name in self.outputs:
            if name not in self.factor_names:
                break
            count += 1
        return count

    @property
    def extra_names(self) -> tuple[str, ...]:
        """Outputs added by physics options, beyond the catalogue."""
        catalogue = {t.name for t in CATALOGUE[self.variant]}
        return tuple(t.name for t in self.targets if t.name not in catalogue)

    @property
    def skipped_physics(self) -> tuple[str, ...]:
        return tuple(p for p in self.options.physics if p not in self.physics)

    @property
    def half_saturation_names(self) -> tuple[str, ...]:
        """Monod constants this variant's uptake rates read from :meth:`resolve`."""
        return _HALF_SATURATION_USED[self.variant]

    # -- the mapping ------------------------------------------------------
    def _reference(self, target: Target) -> float:
        return dict(self.options.references).get(target.name, target.reference)

    def _base(self, target: Target, params: Mapping[str, Any]):
        """The value a zero network output maps to: the reactor's calibrated value,
        else the reference."""
        if target.name in params:
            return params[target.name]
        return self._reference(target)

    def _held(self, target: Target, params: Mapping[str, Any]):
        """The value of a pruned target.

        The reactor's calibrated value, so that a zero network output still
        reproduces the mechanistic benchmark exactly with pruning on; the fixed
        value where the calibration lacks it.
        """
        if target.name in params:
            return params[target.name]
        return target.fixed_value

    def _link(self, target: Target, delta, params: Mapping[str, Any]):
        base = self._base(target, params)
        bound = self.options.bound
        if target.kind == "fraction":
            if bound is not None:
                delta = _bounded(delta, bound)
            value = jnn.sigmoid(_logit(base) + delta)
            if self.options.exact_anchors:
                # The band keeps the logit finite; the part of the anchor it cuts
                # off comes back as an offset, so a zero output is the anchor.
                low, high = FRACTION_BAND
                value = value + (base - jnp.clip(base, low, high))
            return value
        return base * jnp.exp(_bounded(delta, LOG_BOUND if bound is None else bound))

    def _half_saturation(self, name: str, params: Mapping[str, Any]):
        """The campaign's own where it has one, as :meth:`_base` does for the
        targets; otherwise the literature."""
        if name in params:
            return params[name]
        return dict(self.options.references).get(name, HALF_SATURATION[name])

    def resolve(self, network_output, params: Mapping[str, Any], temperature,
                ) -> dict[str, Any]:
        """Every correctable parameter's value, from one network evaluation.

        Works element-wise, so ``network_output`` may be one right-hand-side
        evaluation (shape ``(size,)``) or a whole trajectory (``(T, size)``), with
        ``temperature`` a scalar or a matching ``(T,)`` array.

        """
        index = {name: i for i, name in enumerate(self.outputs)}
        zero = jnp.zeros_like(jnp.asarray(temperature, dtype=float))
        values: dict[str, Any] = {}
        for target in self.targets:
            if target.name in index:
                delta = network_output[..., index[target.name]]
                value = self._link(target, delta, params)
            else:
                value = zero + self._held(target, params)
            if "arrhenius" in self.physics and target.activation_energy is not None:
                value = value * arrhenius_factor(temperature, target.activation_energy)
            values[target.name] = value
        for name in self.half_saturation_names:
            values[name] = self._half_saturation(name, params)
        return values


@functools.lru_cache(maxsize=None)
def _cached_layout(variant: str, options: CorrectionOptions) -> CorrectionLayout:
    if variant not in CATALOGUE:
        raise KeyError(f"no correction catalogue for {variant!r}")
    applicable = tuple(p for p in options.physics if p in _APPLICABLE[variant])
    targets = tuple(CATALOGUE[variant]) + tuple(
        _EXTRAS[p] for p in applicable if p in _EXTRAS
    )
    if options.corrected is not None:
        names = [t.name for t in targets]
        unknown = sorted(set(options.corrected) - set(names))
        if unknown:
            raise KeyError(f"ADM1-{variant.upper()} has no correctable parameter {unknown}; "
                           f"it has {', '.join(names)}")
        outputs = tuple(name for name in names if name in options.corrected)
    else:
        pruned = set(PRUNED[variant]) if options.prune else set()
        outputs = tuple(t.name for t in targets if t.name not in pruned)
    return CorrectionLayout(variant, options, targets, outputs, applicable)


def layout_for(variant: str, options: CorrectionOptions | Mapping[str, Any] | None = None
               ) -> CorrectionLayout:
    """The layout for a variant; ``None`` gives the default options."""
    if not isinstance(options, CorrectionOptions):
        options = CorrectionOptions.from_dict(options)
    return _cached_layout(variant.lower(), options)
