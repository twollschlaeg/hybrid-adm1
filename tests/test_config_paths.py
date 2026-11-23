"""Run-directory resolution.

Orbax writes checkpoints through tensorstore, whose Windows path handling copes
badly with non-ASCII characters and spaces. A repository under a path like
``C:/Users/Jane Müller/...`` hits both, and the failure appears late and
unhelpfully as an access error partway through training.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from adm1_package.experiments.config import (
    RunPaths,
    _needs_ascii_path,
    checkpoint_safe_dir,
    run_dir,
)

WINDOWS = os.name == "nt"


def test_plain_paths_are_not_flagged():
    assert not _needs_ascii_path(Path(r"C:\adm1-runs"))
    assert not _needs_ascii_path(Path("/scratch/adm1/runs"))


def test_spaces_and_non_ascii_are_flagged():
    assert _needs_ascii_path(Path(r"C:\Users\Jane Mueller\runs"))       # space
    assert _needs_ascii_path(Path("/home/jane/Müller/runs"))             # umlaut


def test_directory_is_created_if_absent(tmp_path):
    target = tmp_path / "deep" / "runs"
    assert not target.exists()
    resolved = checkpoint_safe_dir(target)
    assert resolved.exists() and resolved.is_dir()


def test_resolution_is_idempotent(tmp_path):
    once = checkpoint_safe_dir(tmp_path / "runs")
    twice = checkpoint_safe_dir(once)
    assert once == twice


def test_plain_path_is_returned_unchanged(tmp_path):
    # tmp_path is ASCII and space-free on the CI machines we care about; skip if not.
    if _needs_ascii_path(tmp_path):
        pytest.skip("temporary directory itself contains spaces or non-ASCII characters")
    resolved = checkpoint_safe_dir(tmp_path / "runs")
    assert resolved == (tmp_path / "runs").resolve()


def test_resolved_run_dir_points_at_the_same_directory(tmp_path):
    """The sanitized path must be the same directory, not a different one."""
    resolved = checkpoint_safe_dir(tmp_path / "runs")
    marker = resolved / "marker.txt"
    marker.write_text("hello", encoding="utf-8")
    assert (tmp_path / "runs" / "marker.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.skipif(not WINDOWS, reason="8.3 short paths are a Windows feature")
def test_windows_run_dir_is_usable_by_the_checkpoint_writer():
    resolved = run_dir()
    # Either the path was already plain, or it was replaced by a form that is.
    # If neither holds the code emits a RuntimeWarning, which this asserts against.
    assert not _needs_ascii_path(resolved) or resolved.exists()


def test_run_paths_sanitizes_an_explicit_root(tmp_path):
    paths = RunPaths.create("r4", "unit-test", root=tmp_path / "runs")
    assert paths.base.exists()
    assert paths.results.exists()
    assert paths.fold(1) == paths.base / "fold_1"
