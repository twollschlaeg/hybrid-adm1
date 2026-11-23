"""The neural network used inside the UDE right-hand sides, and its input."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from typing import Callable
from jaxtyping import Array, Float


def _layers(in_size, out_size, width_size, depth, use_final_bias, zero_output, key):
    """``depth`` Linear layers from ``in_size`` to ``out_size``.

    ``zero_output`` zeroes the output layer only -- weight and bias -- so the
    network starts by returning exactly zero (the UDE starts at its base model)
    while the hidden layers keep the random initialisation the input dependence
    has to grow from.
    """
    sizes = [in_size] + [width_size] * (depth - 1) + [out_size]
    keys = jrandom.split(key, depth)
    layers = [
        eqx.nn.Linear(sizes[i], sizes[i + 1], use_bias=(i < depth - 1 or use_final_bias), key=keys[i])
        for i in range(depth)
    ]
    if zero_output:
        last = layers[-1]
        last = eqx.tree_at(lambda l: l.weight, last, jnp.zeros_like(last.weight))
        if last.bias is not None:
            last = eqx.tree_at(lambda l: l.bias, last, jnp.zeros_like(last.bias))
        layers[-1] = last
    return layers


class CorrectionMLP(eqx.Module):
    """Multi-layer perceptron producing the UDE correction terms.

    ``depth`` counts Linear layers, so a network with ``depth=n`` has ``n-1``
    hidden layers of ``width_size`` units followed by the output layer. Dropout
    is applied once, between the last hidden layer and the output layer, and the
    same dropout key is used at every right-hand-side evaluation within one
    solve, so a trajectory sees a single consistent mask.

    With ``factor_outputs > 0`` the network is split in two. The first
    ``factor_outputs`` outputs -- the influent degradability factors, which
    describe the substrate -- come from a network of their own that reads only
    the first ``factor_inputs`` inputs, the current forcing. The remaining
    outputs, the rate corrections, come from a network that reads every input.
    Both have the same width and depth. (With the memory of past conditions, now
    retired, the rate network read more inputs than the factor network.)

    ``regularization_loss`` returns the squared Frobenius norm of the weights of
    both, which the trainer scales by ``regularization_lambda``.
    """

    layers: list[eqx.nn.Linear]
    factor_layers: list[eqx.nn.Linear]
    dropout: eqx.nn.Dropout | eqx.nn.Identity
    activation: Callable = jax.nn.relu
    final_activation: Callable = eqx.nn.Identity
    factor_inputs: int = eqx.field(static=True, default=0)

    def __init__(self, in_size, out_size, width_size, depth,
                 activation, final_activation, use_final_bias,
                 dropout_rate=0.0, zero_output=False, factor_inputs=0, factor_outputs=0,
                 *, key, **kwargs):
        super().__init__(**kwargs)
        rate_key, factor_key = jrandom.split(key)
        self.layers = _layers(in_size, out_size - factor_outputs, width_size, depth,
                              use_final_bias, zero_output, rate_key)
        self.factor_layers = (
            _layers(factor_inputs, factor_outputs, width_size, depth,
                    use_final_bias, zero_output, factor_key)
            if factor_outputs else []
        )
        self.factor_inputs = int(factor_inputs) if factor_outputs else 0
        self.activation = activation
        self.final_activation = final_activation
        self.dropout = eqx.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else eqx.nn.Identity()

    def _forward(self, layers, x, key):
        for layer in layers[:-1]:
            x = self.activation(layer(x))
        x = self.dropout(x, key=key)
        return self.final_activation(layers[-1](x))

    def __call__(self, x: Float[Array, "in_size"], key) -> Float[Array, "out_size"]:
        rate_key, factor_key = jrandom.split(key)
        rates = self._forward(self.layers, x, rate_key)
        if not self.factor_layers:
            return rates
        factors = self._forward(self.factor_layers, x[:self.factor_inputs], factor_key)
        return jnp.concatenate([factors, rates])

    def regularization_loss(self):
        return sum(jnp.sum(layer.weight ** 2) for layer in self.layers + self.factor_layers)


def network_input(temperature, loading, influent, scaling):
    """The z-scored network input ``[T, q_in / V_liq, *y_in]``.

    Works on one right-hand-side evaluation (scalars and an influent vector) and
    on a whole trajectory (a leading time axis). ``scaling`` is the dataset's
    ``(means, stds)``.
    """
    features = jnp.concatenate([jnp.stack([temperature, loading], axis=-1), influent], axis=-1)
    return (features - scaling[0]) / scaling[1]
