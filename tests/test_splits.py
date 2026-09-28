from collections import Counter
from pathlib import Path

import pytest

from src.data.splits import (
    HELD_OUT_PAIRS,
    SITUATIONS,
    generate_splits,
    provenance_held_out_pairs,
)


DATASET_PATH = Path(__file__).parents[1] / "data" / "annotated" / "pub_particularized_implicatures.csv"


@pytest.fixture(scope="module")
def splits():
    return generate_splits(DATASET_PATH)


def _ids(rows):
    return {row["question_id"] for row in rows}


def test_fold_counts_and_label_balance(splits):
    for fold in splits.values():
        assert len(fold.training) == 192
        assert len(fold.validation) == 48
        assert len(fold.in_situation_evaluation) == 80
        assert len(fold.cross_situation_evaluation) == 80
        for rows, expected in (
            (fold.training, 96),
            (fold.validation, 24),
            (fold.in_situation_evaluation, 40),
            (fold.cross_situation_evaluation, 40),
        ):
            assert Counter(row["correct_answer"] for row in rows) == {"Yes": expected, "No": expected}


def test_fold_sets_are_disjoint(splits):
    for fold in splits.values():
        sets = [_ids(rows) for rows in (
            fold.training,
            fold.validation,
            fold.in_situation_evaluation,
            fold.cross_situation_evaluation,
        )]
        assert sum(map(len, sets)) == len(set().union(*sets))


def test_held_out_situations_are_isolated(splits):
    for fold in splits.values():
        held_out = set(fold.held_out_situations)
        for rows in (fold.training, fold.validation, fold.in_situation_evaluation):
            assert not held_out.intersection(row["situation"] for row in rows)
        assert {row["situation"] for row in fold.cross_situation_evaluation} == held_out


def test_each_situation_is_held_out_once_and_represented_four_times(splits):
    held_out_counts = Counter(
        situation for fold in splits.values() for situation in fold.held_out_situations
    )
    represented_counts = Counter(
        situation
        for fold in splits.values()
        for situation in SITUATIONS
        if situation not in fold.held_out_situations
    )
    assert held_out_counts == Counter({situation: 1 for situation in SITUATIONS})
    assert represented_counts == Counter({situation: 4 for situation in SITUATIONS})


def test_predefined_held_out_pairs_are_preserved(splits):
    assert tuple(fold.held_out_situations for fold in splits.values()) == HELD_OUT_PAIRS


def test_provenance_procedure_reproduces_authoritative_pairs():
    assert provenance_held_out_pairs() == HELD_OUT_PAIRS


def test_generation_is_deterministic():
    first = generate_splits(DATASET_PATH)
    second = generate_splits(DATASET_PATH)
    assert first == second