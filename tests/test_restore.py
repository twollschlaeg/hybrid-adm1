"""Restoring a trained fold reads its run directory and writes nothing to it.

Evaluation restores the folds of runs that hold the paper's results. Restoring once
rewrote each fold's ``hparams.json`` and attached a log handler to its ``log.txt``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest
from torch.utils.data import DataLoader

from adm1_package.experiments.registry import get_spec
from adm1_package.utils.datasets import custom_collate_fn


def _snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under ``directory``: its bytes and its modification time."""
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _trainer_handlers() -> list:
    """The files the trainer's logger writes to (pytest may add capture handlers)."""
    return [handler.baseFilename for handler in logging.getLogger("Trainer").handlers
            if isinstance(handler, logging.FileHandler)]


def _arrays(model) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))]


@pytest.mark.requires_data
def test_restoring_a_fold_writes_nothing_to_its_run_directory(short_r4_samples, r4_trainer,
                                                              tmp_path):
    """And restores the network training ended with, and the recipe it recorded."""
    _, samples = short_r4_samples
    trainer = r4_trainer()
    loader = DataLoader(samples, batch_size=2, collate_fn=custom_collate_fn)
    trainer.train_model(loader, loader, num_epochs=2)
    fold_dir = tmp_path / "fold_1"
    assert _trainer_handlers() == [], "training left a log file open"

    before = _snapshot(fold_dir)
    restored = get_spec("r4").trainer_class().load_from_checkpoint(str(fold_dir))
    assert _snapshot(fold_dir) == before
    assert _trainer_handlers() == []

    for stored, trained in zip(_arrays(restored.latest_model), _arrays(trainer.state.model)):
        np.testing.assert_array_equal(stored, trained)
    recorded = json.loads((fold_dir / "hparams.json").read_text())
    assert ({k: v for k, v in restored.config.items() if k != "logger_params"}
            == {k: v for k, v in recorded.items() if k != "logger_params"})


def test_a_stored_checkpoint_of_the_papers_runs_restores_read_only(paper_r4_run):
    """Its final and its best network, as stored. Checkpoints written before
    ``utils/checkpoint.py`` record their handler as
    ``adm1_package.utils.EquinoxCheckpointHandler``; restoring does not depend on it."""
    from adm1_package.utils.checkpoint import EquinoxCheckpointHandler

    stored_fold = paper_r4_run / "fold_1"
    metadata = json.loads((stored_fold / "checkpoints" / "adam_200" / "_CHECKPOINT_METADATA")
                          .read_text(encoding="utf-8"))
    assert metadata["item_handlers"]["model_state"] == (
        "adm1_package.utils.EquinoxCheckpointHandler.EquinoxCheckpointHandler")
    assert EquinoxCheckpointHandler.typestr() != metadata["item_handlers"]["model_state"]

    before = _snapshot(stored_fold)
    trainer = get_spec("r4").trainer_class().load_from_checkpoint(str(stored_fold))
    assert _snapshot(stored_fold) == before
    assert _trainer_handlers() == []

    best = json.loads((stored_fold / "metrics" / "best_eval.json").read_text(encoding="utf-8"))
    assert trainer.checkpoint_manager.latest_step() == 200
    for model, epoch in ((trainer.latest_model, 200), (trainer.state.model, best["epoch_idx"])):
        stored = eqx.tree_deserialise_leaves(
            stored_fold / "checkpoints" / f"adam_{epoch}" / "model_state" / "model.eqx",
            trainer.initial_model)
        for left, right in zip(_arrays(model), _arrays(stored)):
            np.testing.assert_array_equal(left, right)
    # a trained network, not the zero-output initialisation
    assert np.any(np.asarray(trainer.latest_model.layers[-1].weight) != 0)
