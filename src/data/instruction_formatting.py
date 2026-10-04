"""Instruction formatting for fine-tuning and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from src.data.splits import FoldSplit, LABELS


NO_HINT = "no_hint"
HINT = "hint"
Condition = Literal["no_hint", "hint"]


@dataclass(frozen=True)
class RenderedExample:
    """A traceable instruction/target pair."""

    question_id: str
    situation: str
    fold: int
    role: str
    instruction: str
    target: str
    condition: str | None = None


def render_instruction(row: Mapping[str, str], *, include_implied_meaning: bool = False) -> str:
    """Render one source row using the No-Hint or Hint template."""

    if row["correct_answer"] not in LABELS:
        raise ValueError(f"Unsupported answer label: {row['correct_answer']!r}")

    implied_meaning = (
        f"\n\nImplied meaning: {row['implied_meaning']}"
        if include_implied_meaning
        else ""
    )
    return (
        f"Context: {row['context']}\n\n"
        f"X: {row['speaker_x']}\n"
        f"Y: {row['speaker_y']}"
        f"{implied_meaning}\n\n"
        "What does Y's response imply?\n\n"
        "Options:\n"
        "- Yes\n"
        "- No\n\n"
        "Answer only with Yes or No."
    )


def render_training_examples(
    split: FoldSplit,
    condition: Condition = NO_HINT,
) -> tuple[RenderedExample, ...]:
    """Render a fold's training rows for the selected fine-tuning condition."""

    _validate_condition(condition)
    return tuple(
        _render_row(
            row,
            fold=split.fold,
            role="training",
            include_implied_meaning=condition == HINT,
            condition=condition,
        )
        for row in split.training
    )


def render_validation_examples(split: FoldSplit) -> tuple[RenderedExample, ...]:
    """Render validation rows with the No-Hint template."""

    return _render_evaluation_role(split.validation, split.fold, "validation")


def render_in_situation_evaluation_examples(
    split: FoldSplit,
) -> tuple[RenderedExample, ...]:
    """Render in-situation evaluation rows with the No-Hint template."""

    return _render_evaluation_role(
        split.in_situation_evaluation,
        split.fold,
        "in_situation_evaluation",
    )


def render_cross_situation_evaluation_examples(
    split: FoldSplit,
) -> tuple[RenderedExample, ...]:
    """Render cross-situation evaluation rows with the No-Hint template."""

    return _render_evaluation_role(
        split.cross_situation_evaluation,
        split.fold,
        "cross_situation_evaluation",
    )


def render_fold_examples(
    split: FoldSplit,
    condition: Condition = NO_HINT,
) -> tuple[RenderedExample, ...]:
    """Render every role in a fold using the experiment's formatting rules."""

    return (
        render_training_examples(split, condition)
        + render_validation_examples(split)
        + render_in_situation_evaluation_examples(split)
        + render_cross_situation_evaluation_examples(split)
    )


def _render_evaluation_role(
    rows: tuple[dict[str, str], ...],
    fold: int,
    role: str,
) -> tuple[RenderedExample, ...]:
    return tuple(
        _render_row(
            row,
            fold=fold,
            role=role,
            include_implied_meaning=False,
            condition=None,
        )
        for row in rows
    )


def _render_row(
    row: Mapping[str, str],
    *,
    fold: int,
    role: str,
    include_implied_meaning: bool,
    condition: str | None,
) -> RenderedExample:
    return RenderedExample(
        question_id=row["question_id"],
        situation=row["situation"],
        fold=fold,
        role=role,
        instruction=render_instruction(
            row,
            include_implied_meaning=include_implied_meaning,
        ),
        target=row["correct_answer"],
        condition=condition,
    )


def _validate_condition(condition: str) -> None:
    if condition not in (NO_HINT, HINT):
        raise ValueError(f"Unsupported formatting condition: {condition!r}")