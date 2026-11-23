"""The split is the piece the whole evaluation hangs on, so it is tested hardest."""

from __future__ import annotations

import json

import pytest

from adm1_package.experiments.splits import (
    DataSplit,
    campaign_of,
    load_split,
    make_split,
    save_split,
)

LABELS = (
    [f"LaRA-temp_F{i}" for i in range(1, 7)]
    + [f"LaRA-substrate_F{i}" for i in range(1, 7)]
    + [f"rice-bran_R{i}0" for i in (3, 6, 9)]
)


def test_campaign_of_strips_only_the_reactor_suffix():
    assert campaign_of("LaRA-substrate_F3") == "LaRA-substrate"
    assert campaign_of("rice-bran_R90") == "rice-bran"


def test_split_partitions_every_sample_exactly_once():
    split = make_split(LABELS, k_folds=4, seed=42)

    assigned = split.test + split.development
    assert sorted(assigned) == sorted(LABELS)
    assert len(assigned) == len(set(assigned)), "a sample appears in more than one part"


def test_test_set_holds_out_one_reactor_per_campaign():
    split = make_split(LABELS, k_folds=4, seed=42, test_per_campaign=1)
    campaigns = [campaign_of(label) for label in split.test]
    assert sorted(campaigns) == ["LaRA-substrate", "LaRA-temp", "rice-bran"]


def test_test_set_never_appears_in_any_fold():
    split = make_split(LABELS, k_folds=4, seed=42)
    for fold in split.folds:
        assert not set(fold) & set(split.test)


def test_training_and_validation_are_disjoint_and_cover_the_development_set():
    split = make_split(LABELS, k_folds=4, seed=42)
    for fold in range(1, split.k_folds + 1):
        train = set(split.training_labels(fold))
        validation = set(split.validation_labels(fold))
        assert not train & validation
        assert train | validation == set(split.development)


def test_each_sample_is_validated_exactly_once_across_the_folds():
    split = make_split(LABELS, k_folds=4, seed=42)
    validated = [label for fold in range(1, split.k_folds + 1)
                 for label in split.validation_labels(fold)]
    assert sorted(validated) == sorted(split.development)


def test_stratification_spreads_campaigns_across_folds():
    split = make_split(LABELS, k_folds=4, seed=42, stratified=True)
    # LaRA-temp has five non-test reactors over four folds, so every fold must
    # contain at least one and none may contain more than two.
    counts = [
        sum(1 for label in fold if campaign_of(label) == "LaRA-temp")
        for fold in split.folds
    ]
    assert min(counts) >= 1
    assert max(counts) <= 2


def test_split_is_deterministic_and_order_independent():
    first = make_split(LABELS, k_folds=4, seed=7)
    second = make_split(list(reversed(LABELS)), k_folds=4, seed=7)
    assert first.to_dict() == second.to_dict()


def test_different_seeds_give_different_splits():
    assert make_split(LABELS, seed=1).to_dict() != make_split(LABELS, seed=2).to_dict()


def test_roundtrip_through_json(tmp_path):
    split = make_split(LABELS, k_folds=4, seed=42)
    path = save_split(split, tmp_path / "split.json")
    assert load_split(path).to_dict() == split.to_dict()


def test_editing_a_splits_dictionary_leaves_the_split_alone():
    split = make_split(LABELS, k_folds=4, seed=42, test_per_campaign=1)
    before = json.loads(json.dumps(split.to_dict()))
    payload = split.to_dict()
    payload["folds"]["fold_1"].pop()
    payload["test"].pop()
    payload["all_labels"].pop()
    assert split.to_dict() == before


def test_a_split_holds_out_the_same_reactors_whatever_their_order():
    split = make_split(LABELS, k_folds=4, seed=42)
    payload = split.to_dict()
    payload["folds"]["fold_1"].reverse()
    assert DataSplit.from_dict(payload).same_assignment(split)
    payload["folds"]["fold_1"], payload["folds"]["fold_2"] = (payload["folds"]["fold_2"],
                                                              payload["folds"]["fold_1"])
    assert not DataSplit.from_dict(payload).same_assignment(split)


def test_indices_resolve_against_dataset_label_order():
    split = make_split(LABELS, k_folds=4, seed=42)
    shuffled = list(reversed(LABELS))

    train, validation = split.fold_indices(1, shuffled)
    assert [shuffled[i] for i in validation] == split.validation_labels(1)
    assert [shuffled[i] for i in train] == split.training_labels(1)


def test_missing_sample_raises_instead_of_silently_shrinking_a_fold():
    split = make_split(LABELS, k_folds=4, seed=42)
    incomplete = [label for label in LABELS if label != split.folds[0][0]]
    with pytest.raises(KeyError, match="does not contain"):
        split.fold_indices(1, incomplete)


def test_rejects_more_folds_than_samples():
    with pytest.raises(ValueError):
        make_split(LABELS, k_folds=99)


def test_rejects_holding_out_a_whole_campaign():
    with pytest.raises(ValueError, match="cannot hold out"):
        make_split(LABELS, k_folds=3, test_per_campaign=3)


def test_load_split_explains_how_to_create_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_splits"):
        load_split(tmp_path / "does-not-exist.json")


def test_summary_mentions_every_fold():
    split = make_split(LABELS, k_folds=4, seed=42)
    summary = split.summary()
    for fold in range(1, 5):
        assert f"fold {fold}" in summary
