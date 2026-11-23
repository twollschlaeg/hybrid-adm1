"""The dataset every ADM1 variant shares: one sample per reactor of the lab campaigns.

A sample is ``(x, y)``. ``x`` holds the forcing (``input``: the daily records of
temperature, inflow and influent joined along ``forcing.forcing_coefficients``, what was
measured at the start, and the network's input scaling), the initial state and the
campaign's parameters. ``y`` holds the measurements, interpolated where a day has none,
their mask and which records hold a reading. Every sample is padded to the longest
record: the forcing coefficients with zeros, the time grid and the targets with their
last value.

A variant subclasses :class:`ADM1Dataset` and sets what differs (class attributes):
its state vector, influent columns, parameters, the prefix of its calibrated initial
values, which states start at their first measurement, and the few choices its stored
calibrations were fitted with. The full ADM1 departs most (``base/dataset.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import ClassVar, Sequence

import diffrax as dx
import jax
import jax.numpy as jnp
import pandas as pd
import torch
from torch.utils.data import Dataset

from adm1_package.models.chemistry import KELVIN
from adm1_package.utils.datasets import (
    InputScalingMixin,
    list_input_files,
    measured_points,
    measured_targets,
    read_outputs,
)
from adm1_package.utils.forcing import forcing_coefficients, resolve_forcing
from adm1_package.utils.gas import correct_gas_initial_values
from adm1_package.utils.initial_state import StateLayout, measured_chemistry, start_measurements
from adm1_package.utils.parameters import load_params

__all__ = ["ADM1Dataset"]


@dataclass(frozen=True)
class _Reactor:
    """One reactor's records as read, before they become a sample."""

    name: str
    times: jax.Array
    temperature: jax.Array          # K
    inflow: jax.Array
    influent: jax.Array             # one column per INFLUENT species
    outputs: jax.Array              # time, then one column per output
    initial_values: jax.Array
    measured: list[str]             # the states that start at their first measurement


class ADM1Dataset(InputScalingMixin, Dataset):
    '''The samples of an ADM1 variant from the input and output files of the LaRA and/or
    rice-bran reactors.

    The outputs are interpolated with cubic Hermite splines to fill in missing data.
    The calibrated initial values of the unmeasured states are read from
    ``<campaign>/<IV_PREFIX>-IVs_<reactor>.csv`` (zero where a reactor has none); the
    measured states, the gas phase and, with a pH among the outputs, the acid-base
    chemistry start at the first measurement. The rice-bran replicates are averaged in
    threes.

    Parameters
    ----------
    root_dir: str
        Root directory where the data is stored.
    data_to_use: list
        Campaigns to use: 'LaRA-temp', 'LaRA-substrate', 'rice-bran'.
    output: list
        Output variables to use. Must contain 'q_gasout' and 'ch4_out'.
    forcing: str
        How the forcing runs between its daily records (``forcing.FORCINGS``): the
        ``smoothstep``, or the ``spline`` of runs and calibrations made before it.
    '''

    #: The state vector, the order of the initial values.
    STATES: ClassVar[StateLayout]
    #: The influent columns of the input files, in the order of the forcing.
    INFLUENT: ClassVar[tuple[str, ...]]
    #: The parameters read from ``<campaign>-parameters.json``; ``None`` reads all but
    #: ``DROPPED_PARAMETERS``.
    PARAMETERS: ClassVar[tuple[str, ...] | None] = None
    DROPPED_PARAMETERS: ClassVar[tuple[str, ...]] = ()
    #: Prefix of the calibrated initial values, ``<IV_PREFIX>-IVs_<reactor>.csv``.
    IV_PREFIX: ClassVar[str]
    #: The states that start at their first measurement, where they are outputs.
    MEASURED_STATES: ClassVar[tuple[str, ...]]
    #: Whether the dissolved CO2 state S_IC starts in equilibrium with the gas phase (R4,
    #: which transfers all of it); the variants with a pH set it with their chemistry.
    S_IC_FROM_GAS: ClassVar[bool] = False
    #: Whether a measurement that a campaign never took is masked: ``read_outputs`` fills
    #: an absent column with NaN, ``measured_targets`` gives it mask 0. Without it the
    #: outputs are read as they are (an absent column is an error) and every point is
    #: scored.
    MASK_MISSING: ClassVar[bool] = True
    #: Defaults of the constructor.
    DEFAULT_CAMPAIGNS: ClassVar[tuple[str, ...]] = ('LaRA-temp',)
    DEFAULT_OUTPUT: ClassVar[tuple[str, ...]]

    #: The campaign whose replicates are averaged, and how many there are of each setup:
    #: its reactors R10, R20, R30 are one setup, averaged at the third.
    REPLICATED_CAMPAIGN: ClassVar[str] = 'rice-bran'
    REPLICATES: ClassVar[int] = 3

    def __init__(self, root_dir: str = 'Daten', data_to_use: Sequence[str] | None = None,
                 output: Sequence[str] | None = None, forcing: str = "smoothstep"):
        data_to_use = list(self.DEFAULT_CAMPAIGNS) if data_to_use is None else data_to_use
        output = list(self.DEFAULT_OUTPUT) if output is None else output
        self.forcing = resolve_forcing(forcing)
        self.root_dir = root_dir
        self.dataset_names = data_to_use
        self.output = output
        self.data = []
        self.targets = []
        self._init_scaling()
        if 'q_gasout' not in output or 'ch4_out' not in output:
            raise ValueError('Output must contain q_gasout and ch4_out')

        max_length = 0
        for campaign in data_to_use:
            data_dir = f'{root_dir}/{campaign}'
            params = self._parameters(data_dir, campaign)
            replicates = []
            for file in list_input_files(data_dir):
                reactor = self._read_reactor(data_dir, file, output)
                # the longest record, before any replicates are cut to their shortest
                max_length = max(max_length, len(reactor.times))
                if campaign == self.REPLICATED_CAMPAIGN:
                    replicates.append(reactor)
                    if int(reactor.name[1]) % self.REPLICATES != 0:
                        continue
                    reactor = _average(replicates[-self.REPLICATES:])
                self._add_sample(campaign, reactor, params, output)

        # Input normalization statistics are fitted on every sample; call
        # dataset.refit_scaling(train_indices) to exclude held-out reactors.
        for sample, targets in zip(self.data, self.targets):
            _pad(sample, targets, max_length)
        self._apply_scaling()

    # -- reading -----------------------------------------------------------------------
    def _parameters(self, data_dir: str, campaign: str) -> dict:
        '''The campaign's calibrated parameters that the variant reads.'''
        param_values = load_params(f'{data_dir}/{campaign}-parameters.json').valuesdict()
        if self.PARAMETERS is not None:
            return {key: jnp.array(value) for key, value in param_values.items()
                    if key in self.PARAMETERS}
        return {key: jnp.array(value) for key, value in param_values.items()
                if key not in self.DROPPED_PARAMETERS}

    def _read_outputs(self, path: str, columns: list[str]) -> pd.DataFrame:
        if self.MASK_MISSING:
            return read_outputs(path, columns)
        return pd.read_csv(path, usecols=columns)[columns]

    def _read_reactor(self, data_dir: str, file: str, output: list[str]) -> _Reactor:
        '''One reactor's input and output files and its initial values.'''
        use_columns = ['time', 'inflow', 'temperature', *self.INFLUENT]
        input_data = pd.read_csv(f'{data_dir}/{file}', usecols=use_columns)[use_columns]
        file_suffix = file.split('_')[1]
        output_columns = ['time'] + output
        output_data = self._read_outputs(f'{data_dir}/output_{file_suffix}', output_columns)
        name = file_suffix.split('.')[0]

        # the calibrated initial values if they exist, otherwise zeroes; the measured
        # states start at their first measurement
        components = list(self.STATES.names)
        path = f'{data_dir}/{self.IV_PREFIX}-IVs_{name}.csv'
        if os.path.exists(path):
            initial_values = pd.read_csv(path, index_col=0).iloc[:, 0]
        else:
            initial_values = pd.Series(0, index=components)
        initial_values = initial_values[components]
        measured = [col for col in self.MEASURED_STATES if col in output_columns]
        initial_values[measured] = output_data[measured].iloc[0].values

        return _Reactor(
            name=name,
            times=jnp.array(input_data['time']),
            temperature=jnp.array(input_data['temperature'].to_numpy() + KELVIN),
            inflow=jnp.array(input_data['inflow']),
            influent=jnp.array(input_data.loc[:, self.INFLUENT[0]:self.INFLUENT[-1]]),
            outputs=jnp.array(output_data.to_numpy()),
            initial_values=jnp.array(initial_values.to_numpy()),
            measured=measured,
        )

    # -- the sample --------------------------------------------------------------------
    def _add_sample(self, campaign: str, reactor: _Reactor, params: dict, output: list[str]):
        times = reactor.times
        # Keep the raw signals so the z-score statistics can be refitted on the training
        # samples of a CV fold.
        self._record_scaling_signals(reactor.temperature, reactor.inflow / params['V_liq'],
                                     reactor.influent)

        # the forcing as recorded, from one day's value to the next along a smoothstep
        # (forcing.forcing_coefficients)
        input_coeffs = forcing_coefficients(
            times,
            jnp.concat([reactor.temperature.reshape(-1,1), reactor.inflow.reshape(-1,1),
                        reactor.influent], axis=1),
            self.forcing)
        sample_input = {'time': times, 'coeffs': input_coeffs}
        forcing = dx.CubicInterpolation(times, input_coeffs)

        # the measurements, interpolated where a day has none
        outputs = reactor.outputs
        output_coeffs = dx.backward_hermite_coefficients(
            outputs[:,0], outputs[:,1:], fill_forward_nans_at_end=True)
        interpolation = dx.CubicInterpolation(outputs[:,0], output_coeffs)
        measurements = jnp.clip(jax.vmap(interpolation.evaluate)(outputs[:,0]), 0, None)

        initial_values = self._initial_gas_phase(reactor.initial_values, params, forcing,
                                                 measurements)
        sample_input['start'] = self._start_measurements(reactor.measured, output, measurements)
        if 'pH' in output and self.STATES.has_ph:
            # inorganic carbon, ionized species and ions from the first measured pH and
            # gas phase, not from the calibrated file
            initial_values = measured_chemistry(
                self.STATES, initial_values, params, forcing.evaluate(times[0])[0],
                sample_input['start']['pH'])

        self.data.append({'input': sample_input, 'initial_values': initial_values,
                          'parameters': params})
        values, indices = self._targets(measurements)
        self.targets.append({'values': values, 'label': f'{campaign}_{reactor.name}',
                             'indices': indices,
                             'measured': measured_points(outputs[:, 1:], output),
                             'object_indices': dict(zip(output, range(len(output))))})

    def _initial_gas_phase(self, initial_values, params, forcing, measurements):
        '''The gas phase from the first gas measurement, and the dissolved methane (and in
        R4 S_IC) in equilibrium with it (utils/gas.py).'''
        return correct_gas_initial_values(
            initial_values, params, forcing, measurements,
            ch4_index=self.STATES.index('S_ch4'),
            co2_index=self.STATES.index('S_IC') if self.S_IC_FROM_GAS else None)

    def _start_measurements(self, measured, output, measurements) -> dict:
        '''What was measured at the start, for the warm-up (utils/initial_state.py).'''
        return start_measurements(list(self.STATES.names), measured, output, measurements)

    def _targets(self, measurements):
        '''The targets and their mask (see ``MASK_MISSING``).'''
        if self.MASK_MISSING:
            return measured_targets(measurements)
        return measurements, jnp.ones(measurements.shape)

    # -- access ------------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, slice):
            idx = range(*idx.indices(len(self.data)))
        if isinstance(idx, (list, range)):
            return [(self.data[i], self.targets[i]) for i in idx]
        return self.data[idx], self.targets[idx]


def _average(replicates: list[_Reactor]) -> _Reactor:
    '''The mean of a setup's replicates over their common length. Their time grid, initial
    values and measured start are the last replicate's.'''
    length = min(len(r.temperature) for r in replicates)

    def mean(field):
        return jnp.mean(jnp.stack([getattr(r, field)[:length] for r in replicates]), axis=0)

    return replace(replicates[-1], times=replicates[-1].times[:length],
                   temperature=mean('temperature'), inflow=mean('inflow'),
                   influent=mean('influent'), outputs=mean('outputs'))


def _pad(sample: dict, targets: dict, length: int):
    '''Pad a sample to ``length`` time points: the forcing coefficients (one interval
    fewer) and the masks with zeros, the time grid and the targets with their last value.'''
    coeffs = sample['input']['coeffs']
    sample['input']['coeffs'] = tuple(
        jnp.pad(jnp.array(c), ((0, length - c.shape[0] - 1), (0, 0)), constant_values=0)
        for c in coeffs)
    time = sample['input']['time']
    sample['input']['time'] = jnp.pad(time, (0, length - time.shape[0]), mode='edge')
    values = targets['values']
    targets['values'] = jnp.pad(values, ((0, length - values.shape[0]), (0, 0)), mode='edge')
    indices = targets['indices']
    targets['indices'] = jnp.pad(indices, ((0, length - indices.shape[0]), (0, 0)),
                                 mode='constant', constant_values=0)
    measured = targets['measured']
    targets['measured'] = jnp.pad(measured, ((0, length - measured.shape[0]), (0, 0)))
