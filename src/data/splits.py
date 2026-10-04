"""Deterministic situation-level cross-validation splits."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SEED = 42
LABELS = ("Yes", "No")
SITUATIONS = (
    "book_preferences",
    "buying_a_flat",
    "colleagues_leaving_work",
    "food_preferences",
    "meeting_childhood_neighbour",
    "meeting_new_neighbour",
    "music_preferences",
    "switching_jobs",
    "travelling_to_meet_someone",
    "weekend_activities",
)
HELD_OUT_PAIRS = (
    ("switching_jobs", "food_preferences"),
    ("colleagues_leaving_work", "travelling_to_meet_someone"),
    ("meeting_new_neighbour", "music_preferences"),
    ("weekend_activities", "meeting_childhood_neighbour"),
    ("book_preferences", "buying_a_flat"),
)
REQUIRED_COLUMNS = {"question_id", "situation", "correct_answer"}


@dataclass(frozen=True)
class FoldSplit:
    """The four mutually exclusive instance sets for one fold."""

    fold: int
    held_out_situations: tuple[str, str]
    training: tuple[dict[str, str], ...]
    validation: tuple[dict[str, str], ...]
    in_situation_evaluation: tuple[dict[str, str], ...]
    cross_situation_evaluation: tuple[dict[str, str], ...]


def load_annotated_rows(source_path: str | Path) -> list[dict[str, str]]:
    """Load the semicolon-separated annotated dataset."""

    with Path(source_path).open(newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file, delimiter=";")
        fieldnames = set(reader.fieldnames or ())
        missing_columns = REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
        rows = [dict(row) for row in reader]

    _validate_source_rows(rows)
    return rows


def generate_splits(
    source_path: str | Path,
    seed: int = SEED,
) -> dict[int, FoldSplit]:
    """Generate and validate the fixed five-fold situation-level splits."""

    rows = load_annotated_rows(source_path)
    splits: dict[int, FoldSplit] = {}

    for fold_number, held_out in enumerate(HELD_OUT_PAIRS, start=1):
        held_out_set = set(held_out)
        represented = [situation for situation in SITUATIONS if situation not in held_out_set]
        rng = random.Random(seed)
        training: list[dict[str, str]] = []
        validation: list[dict[str, str]] = []
        in_situation_evaluation: list[dict[str, str]] = []

        for situation in represented:
            for label in LABELS:
                group = [
                    row
                    for row in rows
                    if row["situation"] == situation and row["correct_answer"] == label
                ]
                group.sort(key=lambda row: row["question_id"])
                rng.shuffle(group)
                training.extend(group[:12])
                validation.extend(group[12:15])
                in_situation_evaluation.extend(group[15:20])

        cross_situation_evaluation = [
            row for row in rows if row["situation"] in held_out_set
        ]
        splits[fold_number] = FoldSplit(
            fold=fold_number,
            held_out_situations=held_out,
            training=tuple(training),
            validation=tuple(validation),
            in_situation_evaluation=tuple(in_situation_evaluation),
            cross_situation_evaluation=tuple(cross_situation_evaluation),
        )

    validate_splits(splits, rows)
    return splits


def provenance_held_out_pairs(seed: int = SEED) -> tuple[tuple[str, str], ...]:
    """Reproduce the documented sorted, shuffled, consecutive pairing procedure."""

    situations = sorted(SITUATIONS)
    random.Random(seed).shuffle(situations)
    return tuple(zip(situations[::2], situations[1::2]))


def validate_splits(
    splits: Mapping[int, FoldSplit],
    source_rows: Sequence[Mapping[str, str]],
) -> None:
    """Raise ``ValueError`` unless all finalized split invariants hold."""

    rows_by_id = _rows_by_id(source_rows)
    if set(splits) != set(range(1, 6)):
        raise ValueError("Expected exactly folds 1 through 5")

    held_out_counts = {situation: 0 for situation in SITUATIONS}
    represented_counts = {situation: 0 for situation in SITUATIONS}

    for fold_number in range(1, 6):
        split = splits[fold_number]
        if split.fold != fold_number:
            raise ValueError(f"Fold key and fold value disagree for fold {fold_number}")
        if split.held_out_situations != HELD_OUT_PAIRS[fold_number - 1]:
            raise ValueError(f"Fold {fold_number} does not use its predefined held-out pair")

        held_out = set(split.held_out_situations)
        represented = set(SITUATIONS) - held_out
        if len(held_out) != 2 or len(represented) != 8:
            raise ValueError(f"Fold {fold_number} has invalid situation counts")

        sets = {
            "training": _ids(split.training),
            "validation": _ids(split.validation),
            "in-situation evaluation": _ids(split.in_situation_evaluation),
            "cross-situation evaluation": _ids(split.cross_situation_evaluation),
        }
        expected_sizes = {
            "training": 192,
            "validation": 48,
            "in-situation evaluation": 80,
            "cross-situation evaluation": 80,
        }
        for name, instance_ids in sets.items():
            if len(instance_ids) != expected_sizes[name]:
                raise ValueError(f"Fold {fold_number} has invalid {name} size")
            if not instance_ids <= rows_by_id.keys():
                raise ValueError(f"Fold {fold_number} contains an unknown instance")

        names = tuple(sets)
        for index, first_name in enumerate(names):
            for second_name in names[index + 1 :]:
                if sets[first_name] & sets[second_name]:
                    raise ValueError(f"Fold {fold_number} has overlapping split instances")

        for name in names[:3]:
            if any(rows_by_id[instance_id]["situation"] in held_out for instance_id in sets[name]):
                raise ValueError(f"Fold {fold_number} leaks a held-out situation into {name}")
        if {rows_by_id[instance_id]["situation"] for instance_id in sets[names[3]]} != held_out:
            raise ValueError(f"Fold {fold_number} cross-situation evaluation is incomplete")

        for situation in represented:
            represented_counts[situation] += 1
            _validate_group_counts(split, rows_by_id, situation, (12, 3, 5))
        for situation in held_out:
            held_out_counts[situation] += 1
            _validate_group_counts(split, rows_by_id, situation, (0, 0, 0), cross_count=40)

    if set(held_out_counts.values()) != {1}:
        raise ValueError("Every situation must be held out exactly once")
    if set(represented_counts.values()) != {4}:
        raise ValueError("Every situation must be represented exactly four times")


def _validate_source_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if len(rows) != 400:
        raise ValueError(f"Expected 400 source instances, found {len(rows)}")
    rows_by_id = _rows_by_id(rows)
    if len(rows_by_id) != len(rows):
        raise ValueError("Source instance identifiers must be unique")
    if {row["situation"] for row in rows} != set(SITUATIONS):
        raise ValueError("Source situations do not match the finalized ten situations")
    for situation in SITUATIONS:
        situation_rows = [row for row in rows if row["situation"] == situation]
        if len(situation_rows) != 40:
            raise ValueError(f"Situation {situation} must contain 40 instances")
        if any(sum(row["correct_answer"] == label for row in situation_rows) != 20 for label in LABELS):
            raise ValueError(f"Situation {situation} must contain 20 instances per label")


def _rows_by_id(rows: Sequence[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    return {row["question_id"]: row for row in rows}


def _ids(rows: Sequence[Mapping[str, str]]) -> set[str]:
    identifiers = [row["question_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("A split contains duplicate instance identifiers")
    return set(identifiers)


def _validate_group_counts(
    split: FoldSplit,
    rows_by_id: Mapping[str, Mapping[str, str]],
    situation: str,
    represented_counts: tuple[int, int, int],
    cross_count: int = 0,
) -> None:
    split_rows = (
        split.training,
        split.validation,
        split.in_situation_evaluation,
    )
    for rows, expected_count in zip(split_rows, represented_counts):
        selected = [row for row in rows if rows_by_id[row["question_id"]]["situation"] == situation]
        if len(selected) != expected_count * 2:
            raise ValueError(f"Invalid count for situation {situation} in fold {split.fold}")
        if any(sum(row["correct_answer"] == label for row in selected) != expected_count for label in LABELS):
            raise ValueError(f"Invalid label balance for situation {situation} in fold {split.fold}")

    cross_rows = [
        row
        for row in split.cross_situation_evaluation
        if rows_by_id[row["question_id"]]["situation"] == situation
    ]
    if len(cross_rows) != cross_count:
        raise ValueError(f"Invalid cross-situation count for situation {situation} in fold {split.fold}")