"""Which of a fold's checkpoints evaluation scores.

Training keeps the epoch with the lowest loss on the fold's held-out reactors. That
checkpoint is selected on the reactors it is then scored on, so its scores are
optimistic; evaluation scores the network after the fixed number of epochs instead,
unless asked for the selected one.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from adm1_package.experiments import evaluation
from adm1_package.experiments.registry import get_spec
from adm1_package.utils.corrections import DEFAULT
from adm1_package.utils.datasets import custom_collate_fn


def _network(value: float):
    layer = eqx.nn.Linear(1, 1, key=jax.random.PRNGKey(0))
    return eqx.tree_at(lambda linear: linear.weight, layer, jnp.full((1, 1), value))


@pytest.fixture
def fake_fold(monkeypatch, tmp_path):
    """A fold whose best checkpoint (epoch 60) and final one (epoch 200) differ."""
    fold_dir = tmp_path / "fold_1"
    (fold_dir / "metrics").mkdir(parents=True)
    (fold_dir / "hparams.json").write_text("{}", encoding="utf-8")
    (fold_dir / "metrics" / "best_eval.json").write_text(json.dumps({"epoch_idx": 60}),
                                                         encoding="utf-8")

    class Trainer:
        """Stands in for the trainer as load_from_checkpoint leaves it: the best
        network in its state, the final one beside it."""

        @classmethod
        def load_from_checkpoint(cls, path, verbose=False):
            trainer = cls()
            trainer.config = {"correction": DEFAULT.to_dict(), "initial_state": "calibrated",
                              "spin_up_days": 30.0, "forcing": "smoothstep",
                              "calibration": "per-fold"}
            trainer.state = SimpleNamespace(model=_network(1.0), key=None)
            trainer.latest_model = _network(2.0)
            trainer.checkpoint_manager = SimpleNamespace(latest_step=lambda: 200)
            return trainer

    spec = get_spec("r4")
    monkeypatch.setattr(type(spec), "trainer_class", lambda self: Trainer)
    return spec, fold_dir


def test_a_fold_is_scored_at_its_final_epoch_by_default(fake_fold):
    spec, fold_dir = fake_fold
    trainer, model = evaluation.restore_fold(spec, fold_dir)
    assert float(model.weight[0, 0]) == 2.0
    assert trainer.scored == ("last", 200)


def test_the_best_held_out_checkpoint_is_scored_only_when_asked_for(fake_fold):
    spec, fold_dir = fake_fold
    trainer, model = evaluation.restore_fold(spec, fold_dir, checkpoint="best")
    assert float(model.weight[0, 0]) == 1.0
    assert trainer.scored == ("best", 60)


def test_an_unknown_checkpoint_is_refused(fake_fold):
    spec, fold_dir = fake_fold
    with pytest.raises(ValueError, match="checkpoint"):
        evaluation.restore_fold(spec, fold_dir, checkpoint="first")


def test_written_results_say_which_checkpoint_they_scored(tmp_path):
    empty = pd.DataFrame()
    result = evaluation.EvaluationResult(empty, empty, empty, empty, empty,
                                         settings={"checkpoint": "last", "epochs": {"1": 200}})
    result.write(tmp_path)
    assert json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8")) == {
        "checkpoint": "last", "epochs": {"1": 200}}


@pytest.mark.requires_data
@pytest.mark.parametrize("keep, steps", [(True, [1, 2, 3, 4, 5, 6]), (False, [1, 6])])
def test_a_run_can_keep_a_checkpoint_at_every_validation(short_r4_samples, r4_trainer, keep, steps):
    """So a fold can later be scored at an epoch chosen without its held-out reactors.

    At learning rate 0 no validation improves on the first, so by default only the
    first (best) and the last are kept; more than four kept also outlives the
    default retention.
    """
    _, samples = short_r4_samples
    trainer = r4_trainer(keep_checkpoints=keep, check_val_every_n_epoch=1,
                         optimizer_hparams={"optimizer": "adam", "lr": 0.0})
    loader = DataLoader(samples, batch_size=2, collate_fn=custom_collate_fn)
    trainer.train_model(loader, loader, num_epochs=6)
    assert list(trainer.checkpoint_manager.all_steps()) == steps
