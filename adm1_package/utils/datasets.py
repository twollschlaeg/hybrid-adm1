"""Shared pieces of the ADM1 datasets: reading the files, the input scaling, batching.

* **deterministic sample order** -- :func:`list_input_files`. The order of a bare
  ``os.listdir`` is filesystem dependent. It would make the sample index -> reactor
  mapping differ between machines, which in turn would make cross-validation fold
  assignments irreproducible, and silently mis-group the rice-bran replicates
  (which are averaged in blocks of three consecutive files).
* **what was measured** -- :func:`read_outputs`, :func:`measured_targets` and
  :func:`measured_points`.
* **input normalization statistics** -- :class:`InputScalingMixin`. z-score
  statistics computed once over *every* loaded sample would, under
  cross-validation, leak the statistics of held-out reactors into the training
  normalization. The datasets therefore record the per-sample raw signals and can
  refit the statistics on an arbitrary index subset.
* **batching** -- :func:`custom_collate_fn` and :func:`create_data_loaders`. PyTorch
  supplies the sampling and shuffling; the batch stays JAX arrays.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import pandas as pd
import torch
import torch.utils.data as data

__all__ = [
    "list_input_files",
    "read_outputs",
    "measured_targets",
    "measured_points",
    "InputScalingMixin",
    "custom_collate_fn",
    "create_data_loaders",
]


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
_REACTOR_NUMBER = re.compile(r"(\d+)")


def _natural_key(file_name: str) -> tuple:
    """Sort key that orders ``input_R10`` before ``input_R90`` and ``F2`` before ``F10``."""
    parts = _REACTOR_NUMBER.split(file_name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def list_input_files(data_dir: str, prefix: str = "input", suffix: str = ".csv") -> list[str]:
    """Return the input files of ``data_dir`` in a deterministic, natural order.

    The order fixes the sample index -> reactor mapping, so it must not depend on
    the filesystem. It also fixes which files are averaged together when the
    rice-bran replicates are pooled.
    """
    files = [
        name
        for name in os.listdir(data_dir)
        if name.startswith(prefix) and name.endswith(suffix)
    ]
    return sorted(files, key=_natural_key)


def read_outputs(path: str, columns: Sequence[str]) -> pd.DataFrame:
    """The ``columns`` of an output file, in that order.

    A column the file lacks is NaN: a measurement that a campaign never took, or that
    an older copy of its files does not hold yet (:func:`measured_targets` masks it).
    Total solids are the exception. Files written before they had a column of their own
    hold them as the water content, ``S_h2o = 1000 - TS``, derived from the same
    measurement.
    """
    available = set(pd.read_csv(path, nrows=0).columns)
    wanted = [name for name in columns if name in available]
    derive_ts = "TS" in columns and "TS" not in available and "S_h2o" in available
    if derive_ts and "S_h2o" not in wanted:
        wanted.append("S_h2o")
    frame = pd.read_csv(path, usecols=wanted)
    if derive_ts:
        frame["TS"] = 1000.0 - frame["S_h2o"]
    for name in columns:
        if name not in frame:
            frame[name] = float("nan")
    return frame[list(columns)]


# ---------------------------------------------------------------------------
# What was measured
# ---------------------------------------------------------------------------
def measured_targets(values):
    """The interpolated measurements and their mask, ``(values, indices)``.

    Where the interpolation is NaN, a column that holds no measurement (see
    :func:`read_outputs`), the value becomes 0 and its mask 0: it is neither scored
    nor drawn, and no NaN can reach a loss. Every other point keeps its value and
    mask 1, as before.
    """
    missing = jnp.isnan(values)
    return jnp.where(missing, 0.0, values), jnp.where(missing, 0.0, 1.0)


#: The outputs read every day, whose records are filled where a day has no reading.
GAS_OUTPUTS = ("q_gasout", "ch4_out")


def measured_points(values, outputs):
    """Where the records hold a measurement, as a 0/1 mask like ``measured_targets``'.

    ``values`` are the outputs as read, before the interpolation (time on the first axis,
    ``outputs`` naming the last). A record without a value was not measured. The records
    copies a gas reading back onto the days without one and onto the first days, so a gas
    rate equal to the next record's is a fill; only the last of the equal ones was read.
    Lab values may repeat (a pH of 7.5 two weeks running), so only the gas is checked.
    The scores count these points; the loss fits the interpolation on every day.
    """
    values = jnp.asarray(values)
    copied = jnp.concatenate([values[:-1] == values[1:], jnp.zeros_like(values[:1], dtype=bool)])
    gas = jnp.array([name in GAS_OUTPUTS for name in outputs])
    return jnp.where(jnp.isfinite(values) & ~(copied & gas), 1.0, 0.0)


# ---------------------------------------------------------------------------
# Input scaling
# ---------------------------------------------------------------------------
class InputScalingMixin:
    """Fit and refit the z-score statistics of the neural-network inputs.

    The dataset subclasses record one entry per sample in ``_scaling_signals``
    (see :meth:`_record_scaling_signals`) and then call :meth:`_apply_scaling`.
    Experiment code can afterwards call :meth:`refit_scaling` with the training
    indices of a fold so that no held-out reactor contributes to the statistics.

    The layout of the statistics vector mirrors the network input vector:
    ``[temperature, inflow / V_liq, *y_in]``.
    """

    _scaling_signals: list[dict[str, jnp.ndarray]]

    def _init_scaling(self) -> None:
        self._scaling_signals = []
        self._scaling_indices: tuple[int, ...] | None = None

    def _record_scaling_signals(
        self,
        temperature: jnp.ndarray,
        hrt_inverse: jnp.ndarray,
        y_in: jnp.ndarray,
    ) -> None:
        """Store the raw signals of one sample so the statistics can be refitted."""
        self._scaling_signals.append(
            {
                "temperature": temperature,
                "hrt_inverse": hrt_inverse,
                "y_in": y_in,
            }
        )

    def _compute_scaling(
        self, indices: Sequence[int] | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(means, stds)`` computed over the given sample indices."""
        signals = self._scaling_signals
        if indices is not None:
            indices = [int(i) for i in indices]
            if not indices:
                raise ValueError("refitting the input scaling needs at least one sample")
            out_of_range = [i for i in indices if not 0 <= i < len(signals)]
            if out_of_range:
                raise IndexError(f"sample indices out of range: {out_of_range}")
            signals = [signals[i] for i in indices]

        temperature = jnp.concatenate([s["temperature"] for s in signals])
        hrt_inverse = jnp.concatenate([s["hrt_inverse"] for s in signals])
        y_in = jnp.concatenate([s["y_in"] for s in signals], axis=0)

        means = jnp.concatenate(
            [
                jnp.array([jnp.mean(temperature), jnp.mean(hrt_inverse)]),
                jnp.mean(y_in, axis=0),
            ]
        )
        stds = jnp.concatenate(
            [
                jnp.array([jnp.std(temperature), jnp.std(hrt_inverse)]),
                jnp.std(y_in, axis=0),
            ]
        )
        # Guard against constant columns.
        stds = jnp.maximum(stds, 1e-6)
        return means, stds

    def _apply_scaling(self, indices: Sequence[int] | None = None) -> None:
        """Compute the statistics and write them onto every sample."""
        means, stds = self._compute_scaling(indices)
        self._scaling_indices = None if indices is None else tuple(int(i) for i in indices)
        for entry in self.data:  # type: ignore[attr-defined]
            entry["input"]["means"] = means
            entry["input"]["stds"] = stds

    def refit_scaling(self, indices: Iterable[int]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Refit the input normalization on ``indices`` only.

        Call this with the *training* indices of a cross-validation fold before
        training or evaluating on that fold; every sample of the dataset then uses
        statistics that no held-out reactor contributed to.
        """
        indices = list(indices)
        self._apply_scaling(indices)
        return self.data[0]["input"]["means"], self.data[0]["input"]["stds"]  # type: ignore[attr-defined]

    @property
    def scaling_indices(self) -> tuple[int, ...] | None:
        """Sample indices the current statistics were fitted on (``None`` = all)."""
        return getattr(self, "_scaling_indices", None)

    @property
    def labels(self) -> list[str]:
        """Sample labels in dataset order."""
        return [target["label"] for target in self.targets]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def custom_collate_fn(batch: List[Tuple[dict, dict]]) -> Tuple[dict, dict]:
    """Stack samples into a batch, keeping JAX arrays.

    A sample is ``(inputs, targets)``: nested dicts of forcing, parameters, initial
    values and measurements, which PyTorch's default collate cannot stack. The
    labels stay a list, and ``object_indices`` (output -> column), which every
    sample shares, is taken from the first.
    """
    data, targets = zip(*batch)
    stacked_data = {'input': {}}
    stacked_data['input']['time'] = jnp.stack([d['input']['time'] for d in data])
    stacked_data['input']['coeffs'] = [jnp.stack([d['input']['coeffs'][j] for d in data])
                                       for j in range(len(data[0]['input']['coeffs']))]
    if 'stds' in data[0]['input']:
        stacked_data['input']['stds'] = jnp.stack([d['input']['stds'] for d in data])
        stacked_data['input']['means'] = jnp.stack([d['input']['means'] for d in data])
    if 'start' in data[0]['input']:
        # what was measured at the start, read by the warm-up initialisation
        stacked_data['input']['start'] = {
            key: jnp.stack([d['input']['start'][key] for d in data])
            for key in data[0]['input']['start']
        }
    stacked_data['initial_values'] = jnp.stack([d['initial_values'] for d in data])
    if 'parameters' in data[0]:
        stacked_data['parameters'] = {key: jnp.stack([d['parameters'][key] for d in data])
                                      for key in data[0]['parameters']}

    stacked_targets = {}
    stacked_targets['values'] = jnp.stack([t['values'] for t in targets])
    stacked_targets['indices'] = jnp.stack([t['indices'] for t in targets])
    if all('measured' in t for t in targets):
        # the records that hold a measurement, which the scores count
        stacked_targets['measured'] = jnp.stack([t['measured'] for t in targets])
    stacked_targets['label'] = [t['label'] for t in targets]
    if all('object_indices' in t for t in targets):
        stacked_targets['object_indices'] = targets[0]['object_indices']
    return stacked_data, stacked_targets


def create_data_loaders(*datasets: data.Dataset,
                        train: Union[bool, Sequence[bool]] = True,
                        batch_size: int = 128,
                        num_workers: int = 0,
                        collate_fn: Optional[Callable] = custom_collate_fn,
                        seed: int = 42):
    """One PyTorch ``DataLoader`` per dataset.

    A training loader (``train``, one flag per dataset or one for all) shuffles and
    drops its last incomplete batch; the others keep their order and every sample.
    Each loader shuffles with its own generator seeded with ``seed``.
    """
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    loaders = []
    for dataset, is_train in zip(datasets, train):
        loaders.append(data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=is_train,
                                       drop_last=is_train,
                                       collate_fn=collate_fn,
                                       num_workers=num_workers,
                                       generator=torch.Generator().manual_seed(seed)))
    return loaders
