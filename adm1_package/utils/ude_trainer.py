"""Training loop of the hybrid models, shared by R2, R3 and R4.

The variants differ only in the UDE they integrate and the objectives they fit,
both of which the caller passes in (see ``experiments/training.py``).

One fold's run directory, as :class:`UDETrainer` leaves it::

    hparams.json        the trainer's configuration (what evaluation rebuilds the fold from)
    log.txt             the losses of every epoch
    metrics/            eval_epoch_<epoch>.pkl per validation, best_eval.{pkl,json}
    checkpoints/        adam_<epoch>/ (see :mod:`adm1_package.utils.checkpoint`)

Training writes the directory; restoring a fold (:meth:`UDETrainer.load_from_checkpoint`)
only reads it.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax import random
from tqdm.auto import tqdm

from adm1_package.utils.checkpoint import (
    EquinoxStateRestore,
    EquinoxStateSave,
    make_deserializable,
    make_serializable,
)
from adm1_package.utils.losses import robust_target_range
from adm1_package.utils.nn_architectures import CorrectionMLP
from adm1_package.utils.optimizers import (
    LR_SCHEDULER,
    OPTIMIZER,
    adam_optimizer,
    build_learning_rate_schedule,
    build_length_schedule,
)

__all__ = ["UDETrainer", "TrainState", "scored_points", "objective_mse", "loss_targets"]


# ---------------------------------------------------------------------------
# The loss, piece by piece
# ---------------------------------------------------------------------------
def scored_points(indices, time, spin_up_days: float):
    """Which target points enter the loss.

    ``indices`` marks the measured points, with time on the second-to-last axis.
    The first point is never scored -- the initial state copies the measured
    values there -- and neither is anything in the first ``spin_up_days`` days,
    which depend on the initial state rather than on the model's dynamics.
    """
    scored = time >= time[..., :1] + spin_up_days
    indices = indices * scored[..., None]
    return indices.at[..., 0, :].set(0)


def objective_mse(residuals, scored):
    """Mean squared residual per objective (last axis) over the scored points."""
    scored = scored > 0
    squared = jnp.where(scored, residuals**2, 0.0)
    axes = tuple(range(squared.ndim - 1))
    return jnp.sum(squared, axis=axes) / jnp.sum(scored, axis=axes)


def loss_targets(targets):
    """The arrays of a batch's targets that the jitted loss reads (no labels)."""
    return {key: targets[key] for key in ("values", "indices", "object_indices")}


def first_part(batch, length):
    """The batch cut to the first fraction ``length`` of its time points (the curriculum
    of :func:`~adm1_package.utils.optimizers.build_length_schedule`)."""
    inputs, targets = batch
    forcing = inputs['input']
    cutoff = int(length * forcing['time'].shape[-1])
    forcing = {**forcing,
               'time': forcing['time'][..., :cutoff],
               'coeffs': [coeffs[..., :cutoff - 1, :] for coeffs in forcing['coeffs']]}
    targets = {**targets,
               'values': targets['values'][..., :cutoff, :],
               'indices': targets['indices'][..., :cutoff, :]}
    return {**inputs, 'input': forcing}, targets


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------
class TrainState(eqx.Module):
    """The network, the optimizer state, the number of optimizer steps taken and the
    PRNG key of the last step."""

    model: Any
    opt_state: Any
    step: int | jax.Array
    key: Any = None

    def update(self, model: Any, opt_state: Any, key: Any) -> "TrainState":
        """The state after one more optimizer step."""
        return TrainState(model=model, opt_state=opt_state, step=self.step + 1, key=key)

    def _replace(self, **kwargs) -> "TrainState":
        return TrainState(
            model=kwargs.get("model", self.model),
            opt_state=kwargs.get("opt_state", self.opt_state),
            step=kwargs.get("step", self.step),
            key=kwargs.get("key", self.key),
        )

    @classmethod
    def create(cls, *, model, opt_state, key) -> "TrainState":
        """A state before the first step."""
        return cls(step=0, model=model, opt_state=opt_state, key=key)


def _check_optimizer(optimizer_hparams: Dict[str, Any], where: Optional[str] = None) -> None:
    """Refuse an optimizer other than Adam on a warmup-cosine-decay learning rate: the
    others, and staged training with several optimizers, were retired."""
    name = optimizer_hparams.get('optimizer', OPTIMIZER)
    scheduler = optimizer_hparams.get('lr_scheduler', LR_SCHEDULER)
    if name != OPTIMIZER or scheduler != LR_SCHEDULER:
        from adm1_package.experiments.retired import retired

        raise retired(f"training with optimizer {name!r} and learning rate {scheduler!r} "
                      f"(only {OPTIMIZER!r} with {LR_SCHEDULER!r} remains)", where)


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------
class UDETrainer:
    """Trains the correction network of one UDE.

    Every keyword argument beyond the network and loss settings (the correction,
    the initial state and ``spin_up_days``) is recorded in the checkpoint's
    ``hparams.json``, which is what evaluation restores a fold from.

    ``correction_penalty`` adds ``lambda * mean(delta**2)`` over the raw network
    outputs along each trajectory (the UDE's ``network_output``) to the training
    loss: a prior that keeps a calibrated hybrid near its calibration. Unlike
    ``regularization_lambda`` it covers the output bias, which sets the level of
    every correction.

    The residuals are divided by each reactor's own range of the objective
    (``losses.robust_target_range``).

    Arguments of the loop:

    optimizer_hparams
        ``lr``, ``warmup``, ``end_value``, ``gradient_clip`` and ``length_schedule`` of
        Adam on a warmup-cosine-decay learning rate (:mod:`adm1_package.utils.optimizers`).
    batch_size
        Reactors per batch, at least 2: the loss integrates a batch with ``vmap``.
    logger_params
        Where the run directory is: ``log_dir``, relative to ``base_log_dir`` (default
        the configured run directory, ``experiments.config.run_dir``).
    check_val_every_n_epoch
        How often the held-out reactors are scored, and a checkpoint possibly kept.
    keep_checkpoints (keyword)
        Keep a checkpoint at every validation, not only when the validation loss
        improves; recorded with the other keywords.
    debug
        Run the steps without ``jit``.
    """

    def __init__(self,
                 in_size: int,
                 width_size: int,
                 out_size: int,
                 depth: int,
                 activation: Callable = jnn.relu,
                 final_activation: Callable = jnn.identity,
                 use_final_bias: bool = True,
                 dropout_rate: float = 0.0,
                 zero_output: bool = False,
                 factor_inputs: int = 0,
                 factor_outputs: int = 0,
                 ude=None,
                 seed: int = 42,
                 batch_size: int = 8,
                 objective_weights: Dict[str, float] | None = None,
                 objective_list: List[str] | None = None,
                 regularization_lambda: float = 0.0,
                 optimizer_hparams: Dict[str, Any] | None = None,
                 logger_params: Optional[Dict[str, Any]] = None,
                 enable_progress_bar: bool = True,
                 debug: bool = False,
                 check_val_every_n_epoch: int = 1,
                 verbose: bool = True,
                 **kwargs):
        if optimizer_hparams is None:
            raise TypeError("UDETrainer needs optimizer_hparams")
        if batch_size < 2:
            raise ValueError(f"batch_size must be at least 2, not {batch_size}")
        model_hparams = {
            "in_size": in_size,
            "width_size": width_size,
            "out_size": out_size,
            "depth": depth,
            "activation": activation,
            "final_activation": final_activation,
            "use_final_bias": use_final_bias,
            "dropout_rate": dropout_rate,
            "zero_output": zero_output,
            "factor_inputs": factor_inputs,
            "factor_outputs": factor_outputs,
        }
        self.optimizer_hparams = optimizer_hparams
        self.batch_size = batch_size
        self.seed = seed
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.verbose = verbose
        self.keep_checkpoints = bool(kwargs.get("keep_checkpoints", False))
        # Left in kwargs, so the record carries them and evaluation scores the same
        # window.
        self.spin_up_days = float(kwargs.get("spin_up_days", 0.0))
        self.correction_penalty = float(kwargs.get("correction_penalty", 0.0))
        self.ude = ude
        self.regularization_lambda = regularization_lambda
        self.objective_weights = dict(objective_weights or {})
        if objective_list:
            self.objective_list = list(objective_list)
        else:
            self.objective_list = [name for name, weight in self.objective_weights.items() if weight > 0.0]

        # What hparams.json records, in this order.
        self.config = {
            'model_class': CorrectionMLP.__name__,
            'model_hparams': make_serializable(model_hparams),
            'optimizer_hparams': make_serializable(optimizer_hparams),
            'batch_size': self.batch_size,
            'logger_params': make_serializable(logger_params),
            'enable_progress_bar': self.enable_progress_bar,
            'debug': self.debug,
            'check_val_every_n_epoch': check_val_every_n_epoch,
            'verbose': self.verbose,
            'seed': self.seed,
        }
        serialized_kwargs = make_serializable(kwargs)
        if isinstance(serialized_kwargs, dict):
            self.config.update(serialized_kwargs)

        self.log_dir = _resolve_log_dir(logger_params or {})
        #: The untrained network: the start of training, and the template a
        #: checkpoint is restored into.
        self.initial_model = CorrectionMLP(key=random.key(self.seed), **model_hparams)
        #: The network after the last epoch, set when a fold is restored.
        self.latest_model = None
        self._build_step_functions()
        self.state = TrainState(step=0, model=self.initial_model, opt_state=None,
                                key=random.key(self.seed))

    # -- the loss and the steps ----------------------------------------------
    def loss_fn(self, model, ude, data, key,
                regularization_lambda=None,
                correction_penalty=None,
                ) -> Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], jnp.ndarray]]:
        """The training loss of a batch and, as aux, the solution and the per-objective MSE.

        The scaled RMSE over the fitted objectives, plus the weight penalty and the
        correction penalty. The two penalties are arguments rather than reads of
        ``self``: this function is jitted, so a read of ``self`` would freeze into the
        first trace, and evaluation could not ask for the unpenalised value.
        ``None`` takes the trainer's own.
        """
        x, y = data
        # A fixed objective list, so the column selection is static under jit.
        obj_list = self.objective_list
        object_indices = y['object_indices']
        targets = y['values'][..., [object_indices[obj] for obj in obj_list]]
        indices = y['indices'][..., [object_indices[obj] for obj in obj_list]]
        indices = scored_points(indices, x['input']['time'], self.spin_up_days)
        weights = jnp.array(
            [self.objective_weights[obj] for obj in obj_list]
        )

        sol_dic, _ = eqx.filter_vmap(
            ude, in_axes=(0, 0, 0, None, None), out_axes=0
        )(x['parameters'], x['initial_values'], x['input'], model, key)
        pred_y = jnp.array([sol_dic[obj] for obj in obj_list]).T.swapaxes(0, 1)

        # A solve that produced non-finite values scores as a large loss
        # rather than poisoning the gradient.
        pred_finite = jnp.all(jnp.isfinite(pred_y))

        def compute_loss():
            target_range = robust_target_range(targets, axis=1, where=indices > 0)
            residuals = jnp.divide((pred_y - targets), target_range)
            return objective_mse(residuals, indices)

        def fallback_loss():
            return jnp.full(len(obj_list), 1e6)

        mse_objectives = jax.lax.cond(pred_finite, compute_loss, fallback_loss)
        mse_objectives = jnp.clip(mse_objectives, 0, 1e6)
        weighted_mse = jnp.sum(mse_objectives * weights) / jnp.sum(weights)
        rmse = jnp.sqrt(weighted_mse)

        penalty_weight = (
            self.regularization_lambda if regularization_lambda is None
            else regularization_lambda
        )
        # Guard against a non-finite solve, then add the penalty.
        rmse = (jnp.where(jnp.isfinite(rmse), rmse, 1e6)
                + penalty_weight * model.regularization_loss())
        # A Python float either way, so 0 leaves the term out of the trace.
        correction_weight = (
            self.correction_penalty if correction_penalty is None else correction_penalty
        )
        if correction_weight:
            rmse = rmse + correction_weight * jnp.mean(sol_dic['network_output'] ** 2)

        return rmse, (sol_dic, mse_objectives)

    def _update(self, model, ude, opt_state, data, key):
        """One optimizer step: the gradient of the loss, clipped elementwise to +-1e6,
        through the optimizer (which clips its global norm)."""
        params = eqx.filter(model, eqx.is_inexact_array)
        (loss, _), grads = eqx.filter_value_and_grad(self.loss_fn, has_aux=True)(
            model, ude, data, key)
        grads = jax.tree_util.tree_map(lambda x: jnp.clip(x, -1e6, 1e6), grads)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, {"loss": loss}

    def _build_step_functions(self):
        """The jitted update and evaluation loss, unjitted with ``debug``.

        Built per trainer, so that each one counts its own traces.
        """
        update = eqx.debug.assert_max_traces(self._update, max_traces=4)
        if self.debug:
            self._jitted_update, self._jitted_loss = update, self.loss_fn
        else:
            self._jitted_update = eqx.filter_jit(update)
            self._jitted_loss = eqx.filter_jit(self.loss_fn)

    def train_step(self, state: TrainState, batch) -> Tuple[TrainState, Dict]:
        """One optimizer step on the part of the batch the curriculum has reached."""
        batch = first_part(batch, self.length_schedule(state.step))
        _, subkey = jax.random.split(state.key)
        batch_arrays = (batch[0], loss_targets(batch[1]))
        model, opt_state, metrics = self._jitted_update(
            state.model, self.ude, state.opt_state, batch_arrays, subkey
        )
        return state.update(model=model, opt_state=opt_state, key=subkey), metrics

    def eval_step(self, state: TrainState, batch) -> Tuple[TrainState, Dict]:
        """The validation metrics of a batch, the network in inference mode.

        The loss is the unpenalised one. The L2 penalty belongs in the training
        gradient, not in a number used to *choose between* models: it falls as lambda
        falls with the network untouched, so leaving it in makes the validation
        metric -- and with it checkpoint selection and the Optuna objective -- reward
        a smaller lambda regardless of the fit. It is reported alongside, so it stays
        visible rather than merely absent. The correction penalty likewise;
        ``correction`` is how far the corrections go, ``mean(delta**2)``.
        """
        model = eqx.nn.inference_mode(state.model)
        batch_arrays = (batch[0], loss_targets(batch[1]))
        loss, (sol_dic, mse_objectives) = self._jitted_loss(
            model, self.ude, batch_arrays, state.key, 0.0, 0.0
        )
        penalty = self.regularization_lambda * model.regularization_loss()
        return state, {'loss': loss, 'penalty': penalty,
                       'correction': jnp.mean(sol_dic['network_output'] ** 2),
                       'mse_objectives': mse_objectives}

    # -- training --------------------------------------------------------------
    def _init_optimizer(self, num_epochs: int, num_steps_per_epoch: int):
        """The optimizer, the learning-rate and curriculum schedules, and the state
        before the first step."""
        hparams = copy.copy(self.optimizer_hparams)
        _check_optimizer(hparams)
        if not isinstance(num_epochs, int):
            raise TypeError("num_epochs must be an integer")
        self.length_schedule = build_length_schedule(
            hparams.get('length_schedule'), num_epochs, num_steps_per_epoch
        )
        lr_schedule = build_learning_rate_schedule(
            hparams.get('lr', 1e-3),
            hparams.get('warmup', 0),
            int(num_epochs * num_steps_per_epoch),
            end_value=hparams.get('end_value'),
        )
        self.optimizer = adam_optimizer(lr_schedule, gradient_clip=hparams.get('gradient_clip', 1.0))
        self.state = TrainState.create(
            model=self.initial_model,
            opt_state=self.optimizer.init(eqx.filter(self.initial_model, eqx.is_inexact_array)),
            key=random.key(self.seed),
        )

    def train_model(self, train_loader, val_loader, num_epochs: int) -> Dict[str, Any]:
        """Train for ``num_epochs`` epochs and return the metrics of the best validation.

        The held-out reactors are scored every ``check_val_every_n_epoch`` epochs. A
        checkpoint is kept whenever their loss improves (all of them with
        ``keep_checkpoints``), and one after the last epoch.
        """
        self._init_optimizer(num_epochs, len(train_loader))
        self._open_run_directory()
        try:
            return self._train(train_loader, val_loader, num_epochs)
        finally:
            self._close_logger()

    def _train(self, train_loader, val_loader, num_epochs: int) -> Dict[str, Any]:
        best_eval_metrics = None
        self.checkpoint_manager = ocp.CheckpointManager(
            directory=os.path.join(self.log_dir, 'checkpoints'),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=None if self.keep_checkpoints else 4,
                create=True,
                cleanup_tmp_directories=True,
                enable_async_checkpointing=False,
                step_prefix=OPTIMIZER,  # adam_<epoch>
            ),
        )
        for epoch_idx in self._tracker(iter(range(1, num_epochs + 1)), desc='Epochs'):
            train_metrics = self._train_epoch(train_loader)
            self.logger.info(f"Epoch {epoch_idx} train metrics: {train_metrics}")
            if epoch_idx % self.check_val_every_n_epoch != 0:
                continue
            eval_metrics = self.eval_model(val_loader, log_prefix='val/', epoch_idx=epoch_idx)
            self._save_metrics(f'eval_epoch_{str(epoch_idx).zfill(3)}', eval_metrics)
            self.logger.info(f"Epoch {epoch_idx} validation metrics: {eval_metrics}")
            if self._is_new_model_better(eval_metrics, best_eval_metrics):
                best_eval_metrics = eval_metrics
                best_eval_metrics.update(train_metrics)
                self.save_model(step=epoch_idx)
                self._save_metrics('best_eval', eval_metrics)
            elif self.keep_checkpoints:
                self.save_model(step=epoch_idx)
        self.save_model(step=num_epochs)
        self.checkpoint_manager.wait_until_finished()
        self.logger.info('success')
        return best_eval_metrics if best_eval_metrics is not None else {}

    def _train_epoch(self, train_loader) -> Dict[str, Any]:
        """One pass over the training reactors; the metrics are the mean over its steps."""
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        for batch in self._tracker(train_loader, desc='Training', leave=False):
            self.state, step_metrics = self.train_step(self.state, batch)
            for key in step_metrics:
                metrics['train/' + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics['epoch_time'] = time.time() - start_time
        return metrics

    def eval_model(self, data_loader, log_prefix: str = '',
                   epoch_idx: Optional[int] = None) -> Dict[str, Any]:
        """The evaluation metrics on ``data_loader``, each a mean over its reactors.

        Each batch's metrics are weighted by its number of reactors, so a short last
        batch counts for what it holds.
        """
        metrics = defaultdict(float)
        num_elements = 0
        for batch in data_loader:
            _, step_metrics = self.eval_step(self.state, batch)
            batch_size = batch[1]['values'].shape[0]
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size
            num_elements += batch_size
        metrics = {(log_prefix + key): (metrics[key] / num_elements) for key in metrics}
        metrics['optimizer'] = OPTIMIZER
        if epoch_idx is not None:
            metrics['epoch_idx'] = epoch_idx
        return metrics

    @staticmethod
    def _is_new_model_better(new_metrics: Dict[str, Any],
                             old_metrics: Optional[Dict[str, Any]]) -> bool:
        """Whether the validation loss improved; a tie keeps the earlier epoch."""
        return old_metrics is None or new_metrics['val/loss'] < old_metrics['val/loss']

    def _tracker(self, iterable: Iterable, **kwargs) -> Iterable:
        """``iterable`` in a tqdm progress bar, if the progress bar is enabled."""
        if self.enable_progress_bar:
            return iter(tqdm(iterable, **kwargs))
        return iterable

    # -- the run directory -----------------------------------------------------
    def _open_run_directory(self):
        """Create the run directory, write ``hparams.json`` and log to ``log.txt``
        (appended to, and to stdout with ``verbose``)."""
        os.makedirs(os.path.join(self.log_dir, 'metrics'), exist_ok=True)
        with open(os.path.join(self.log_dir, 'hparams.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        self.logger = logging.getLogger('Trainer')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handlers = [logging.FileHandler(os.path.join(self.log_dir, 'log.txt'), mode='a')]
        if self.verbose:
            handlers.append(logging.StreamHandler(sys.stdout))
        for handler in handlers:
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _close_logger(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

    def _save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """``metrics/<filename>.pkl``; the best validation also as JSON, for reading."""
        with open(os.path.join(self.log_dir, f'metrics/{filename}.pkl'), 'wb') as f:
            pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
        if filename == "best_eval":
            with open(os.path.join(self.log_dir, f'metrics/{filename}.json'), 'w') as f:
                json.dump(make_serializable(metrics), f, indent=4)

    def save_model(self, step: int = 0):
        """Checkpoint the network and the optimizer state as ``adam_<step>``."""
        self.checkpoint_manager.save(
            step=step,
            args=ocp.args.Composite(
                model_state=EquinoxStateSave(self.state.model),
                opt_state=ocp.args.StandardSave(self.state.opt_state),
            ),
        )

    # -- restoring a fold ------------------------------------------------------
    @classmethod
    def load_from_checkpoint(cls, checkpoint: str, verbose: bool = False) -> "UDETrainer":
        """The trainer of the run directory ``checkpoint``, its networks restored.

        ``state.model`` is the network of the best validation and ``latest_model``
        the one after the last epoch. If the best epoch's checkpoint is gone, the last
        one stands in for it. Nothing in the directory is written.

        ``verbose`` prints which epochs were restored.
        """
        checkpoint = os.path.abspath(checkpoint)
        hparams_file = os.path.join(checkpoint, 'hparams.json')
        if not os.path.isfile(hparams_file):
            raise FileNotFoundError(f"Could not find hparams file {hparams_file}")
        with open(hparams_file, 'r') as f:
            hparams = make_deserializable(json.load(f))
        if not isinstance(hparams, dict):
            raise TypeError(f"Expected hparams to be a dictionary, got {type(hparams)}")

        hparams.pop('model_class')
        hparams.update(hparams.pop('model_hparams'))
        # The directory we were handed *is* the run directory. The recorded
        # base_log_dir belongs to the machine that trained the model; runs have to
        # survive being copied between a workstation and the cluster.
        logger_params = dict(hparams.get('logger_params') or {})
        logger_params['log_dir'] = checkpoint
        logger_params.pop('base_log_dir', None)
        hparams['logger_params'] = logger_params
        # A checkpoint of a retired optimizer is refused before anything is restored.
        _check_optimizer(hparams.get('optimizer_hparams', {}), where=checkpoint)
        trainer = cls(**hparams)

        best_step = _best_epoch(os.path.join(checkpoint, 'metrics'))
        trainer.checkpoint_manager = ocp.CheckpointManager(
            directory=os.path.join(checkpoint, 'checkpoints'),
            options=ocp.CheckpointManagerOptions(step_prefix=OPTIMIZER, read_only=True),
        )
        available = trainer.checkpoint_manager.all_steps()
        latest_step = trainer.checkpoint_manager.latest_step()
        if latest_step is None:
            raise FileNotFoundError(f"no checkpoints in {checkpoint}")
        trainer.latest_model = trainer._restore_network(latest_step)
        if best_step not in available:
            best_step = latest_step
        best = (trainer.latest_model if best_step == latest_step
                else trainer._restore_network(best_step))
        trainer.state = TrainState.create(model=best, opt_state=None, key=random.key(trainer.seed))
        if verbose:
            print(f"Restored {checkpoint}: best epoch {best_step}, last epoch {latest_step} "
                  f"(of {list(available)})")
        return trainer

    def _restore_network(self, step: int):
        restored = self.checkpoint_manager.restore(
            step=step,
            args=ocp.args.Composite(model_state=EquinoxStateRestore(self.initial_model)),
        )
        return restored.model_state


def _resolve_log_dir(logger_params: Dict[str, Any]) -> str:
    """The absolute run directory: ``log_dir``, relative to ``base_log_dir`` (default the
    configured run directory) unless absolute. Orbax requires absolute paths."""
    log_dir = logger_params.get('log_dir', None)
    if log_dir and os.path.isabs(log_dir):
        return os.path.abspath(log_dir)
    if 'base_log_dir' in logger_params:
        base_log_dir = logger_params['base_log_dir']
    else:
        # Imported here to keep utils free of a dependency on the experiments layer.
        from adm1_package.experiments.config import run_dir
        base_log_dir = str(run_dir())
    if not log_dir:
        return os.path.abspath(os.path.join(base_log_dir, CorrectionMLP.__name__))
    return os.path.abspath(os.path.join(base_log_dir, log_dir))


def _best_epoch(metrics_dir: str) -> int:
    """The epoch of a run's best validation, from ``best_eval.pkl`` (or ``.json``)."""
    pickled = os.path.join(metrics_dir, 'best_eval.pkl')
    if os.path.exists(pickled):
        with open(pickled, 'rb') as f:
            best_eval_metrics = pickle.load(f)
    else:
        with open(os.path.join(metrics_dir, 'best_eval.json'), 'r') as f:
            best_eval_metrics = json.load(f)
    return best_eval_metrics.get('epoch_idx', 0)
