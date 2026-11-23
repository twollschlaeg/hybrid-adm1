"""Training-set augmentation.

``add_noise`` replicates each sample with Gaussian noise on the measurements (and,
if asked, on the forcing), which regularizes the fit against measurement error. It
creates no independent experiments: it broadens the neighbourhood of the observed
operating conditions, it does not extend its range. (The synthetic samples from
perturbed forcing were retired before this release; see ``experiments/retired.py``.)
"""

from __future__ import annotations

import copy
from typing import Any, cast

import diffrax as dx
import jax
import jax.numpy as jnp
from torch.utils.data import Subset

from adm1_package.utils.forcing import forcing_coefficients

__all__ = ["add_noise"]


def add_noise(dataset, size: int = 1024, seed: int = 42,
              noise_output: float = 0.2, noise_input: float = 0.3):
    """A copy of ``dataset`` holding ``size`` noisy replicates of each sample.

    The measurements get Gaussian noise of ``noise_output`` times each column's
    standard deviation, clipped at zero. The forcing is read off its interpolation at
    the daily records, gets noise of ``noise_input`` times each column's standard
    deviation (keep it small), is clipped to physical values (temperature 0-100 degC,
    inflow and influent non-negative) and is interpolated again as the dataset's own.
    ``dataset`` may be a ``Subset``; the copy is of the dataset beneath it.
    """
    key = jax.random.key(seed)
    subkeys = jax.random.split(key, len(dataset))
    noisy_data = []
    noisy_targets = []
    # the noisy forcing is built as the dataset builds its own
    forcing = getattr(getattr(dataset, "dataset", dataset), "forcing", "smoothstep")
    
    for i in range(len(dataset)):
        data, targets = dataset[i]  # type: ignore
        assert isinstance(targets, dict) and 'values' in targets
        values = targets['values']
        keys = jax.random.split(subkeys[i], size)
        
        # Get original time and coefficients
        times = data['input']['time']
        coeffs = data['input']['coeffs']
        
        # Find valid time steps (exclude padded values)
        # Padded values are created with mode='edge', so they repeat the last value
        valid_mask = jnp.concatenate([jnp.array([True]), jnp.diff(times) > 0])
        n_valid = int(jnp.sum(valid_mask, dtype=int))
        times_valid = times[:n_valid]
        
        # Truncate coefficients to valid length
        coeffs_valid = tuple(c[:n_valid-1] for c in coeffs)  # coeffs are one shorter than times
        
        # Reconstruct original input values from interpolation
        interp = dx.CubicInterpolation(times_valid, coeffs_valid)
        original_inputs = jax.vmap(interp.evaluate)(times_valid)  # shape: (n_valid, n_inputs)
        
        # Calculate std for each input column
        input_stds = jnp.std(original_inputs, axis=0)
        
        def get_noisy_data(key):
            # Split key for output and input noise
            key_out, key_in = jax.random.split(key)
            
            # Add noise to outputs (only valid portion)
            noise_out = jax.random.normal(key_out, shape=(n_valid, values.shape[1]))
            noisy_values_valid = jnp.clip(
                values[:n_valid] + noise_out * noise_output * jnp.std(values[:n_valid], axis=0), 
                0, None
            )
            # Pad noisy values back to original length
            noisy_values = jnp.pad(
                noisy_values_valid, 
                ((0, values.shape[0] - n_valid), (0, 0)), 
                mode='edge'
            )
            
            # Add noise to raw input values
            noise_in = jax.random.normal(key_in, shape=original_inputs.shape)
            noisy_inputs = original_inputs + noise_in * noise_input * input_stds
            
            # Apply physical constraints
            # Temperature (first column)
            noisy_inputs = noisy_inputs.at[:, 0].set(jnp.clip(noisy_inputs[:, 0], 273.15, 373.15))  # 0-100°C
            # Inflow (second column)
            noisy_inputs = noisy_inputs.at[:, 1].set(jnp.clip(noisy_inputs[:, 1], 0, None))
            # Substrates (remaining columns)
            noisy_inputs = noisy_inputs.at[:, 2:].set(jnp.clip(noisy_inputs[:, 2:], 0, None))
            
            # through the noisy knots, as the dataset's forcing passes through its own
            noisy_coeffs = forcing_coefficients(times_valid, noisy_inputs, forcing)
            
            # Pad coefficients back to original length
            noisy_coeffs_padded = tuple(
                jnp.pad(c, ((0, coeffs[j].shape[0] - c.shape[0]), (0, 0)), constant_values=0)
                for j, c in enumerate(noisy_coeffs)
            )
            
            # Create noisy data dictionary
            noisy_data_dict = copy.deepcopy(data)
            noisy_data_dict['input']['coeffs'] = noisy_coeffs_padded
            
            return noisy_data_dict, noisy_values
            
        for j in range(size):
            noisy_data_dict, noisy_values = get_noisy_data(keys[j])
            noisy_data.append(noisy_data_dict)
            noisy_targets.append({
                'values': noisy_values, 
                'indices': targets['indices'],
                **{k: v for k, v in targets.items() if k not in ['values', 'indices']}
            })
    
    # Return a dataset with the same attributes as the input dataset type
    if isinstance(dataset, Subset):
        noisy_dataset = copy.deepcopy(dataset.dataset)
    else:
        noisy_dataset = copy.deepcopy(dataset)

    noisy_dataset_any = cast(Any, noisy_dataset)
    if not hasattr(noisy_dataset_any, 'data') or not hasattr(noisy_dataset_any, 'targets'):
        raise TypeError(
            "add_noise expects a dataset (or Subset of one) with 'data' and 'targets' attributes."
        )

    noisy_dataset_any.data = noisy_data
    noisy_dataset_any.targets = noisy_targets
    return noisy_dataset
