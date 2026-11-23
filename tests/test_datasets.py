"""Consistency checks across the three model variants. Needs the reactor data."""

from __future__ import annotations

import pytest

from adm1_package.experiments.registry import MODEL_KEYS, get_spec

pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def datasets(installed_dataset):
    return {key: installed_dataset(key) for key in MODEL_KEYS}


def test_all_variants_enumerate_the_same_reactors(datasets):
    # A shared split is only meaningful if fold k means the same reactors for
    # every variant.
    orders = {key: dataset.labels for key, dataset in datasets.items()}
    reference = orders[MODEL_KEYS[0]]
    for key, labels in orders.items():
        assert labels == reference, f"{key} enumerates reactors differently"


def test_labels_are_unique(datasets):
    for key, dataset in datasets.items():
        labels = dataset.labels
        assert len(labels) == len(set(labels)), f"{key} has duplicate labels"


def test_rice_bran_replicates_are_averaged_into_three_samples(datasets):
    for key, dataset in datasets.items():
        rice = [label for label in dataset.labels if label.startswith("rice-bran")]
        assert sorted(rice) == ["rice-bran_R30", "rice-bran_R60", "rice-bran_R90"], key


def test_targets_and_inputs_share_the_padded_length(datasets):
    for key, dataset in datasets.items():
        inputs, targets = dataset[0]
        assert targets["values"].shape[0] == inputs["input"]["time"].shape[0], key
        assert targets["indices"].shape == targets["values"].shape, key


def test_every_requested_output_is_indexed(datasets):
    for key, dataset in datasets.items():
        spec = get_spec(key)
        _, targets = dataset[0]
        assert set(targets["object_indices"]) == set(spec.outputs), key
