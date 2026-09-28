from pathlib import Path

from src.data.instruction_formatting import (
    HINT,
    NO_HINT,
    render_cross_situation_evaluation_examples,
    render_fold_examples,
    render_in_situation_evaluation_examples,
    render_training_examples,
    render_validation_examples,
)
from src.data.splits import generate_splits


DATASET_PATH = Path(__file__).parents[1] / "data" / "annotated" / "pub_particularized_implicatures.csv"


def test_training_hint_adds_only_the_implied_meaning_line():
    split = generate_splits(DATASET_PATH)[1]
    no_hint = render_training_examples(split, NO_HINT)[0]
    hint = render_training_examples(split, HINT)[0]
    row = split.training[0]

    assert f"Implied meaning: {row['implied_meaning']}" not in no_hint.instruction
    assert f"Implied meaning: {row['implied_meaning']}" in hint.instruction
    assert hint.instruction.replace(
        f"\n\nImplied meaning: {row['implied_meaning']}", ""
    ) == no_hint.instruction


def test_validation_and_evaluation_never_expose_implied_meaning():
    split = generate_splits(DATASET_PATH)[1]
    rendered = (
        render_validation_examples(split)
        + render_in_situation_evaluation_examples(split)
        + render_cross_situation_evaluation_examples(split)
    )

    assert all(
        row["implied_meaning"] not in example.instruction
        for row, example in zip(
            split.validation + split.in_situation_evaluation + split.cross_situation_evaluation,
            rendered,
        )
    )
    assert all(example.condition is None for example in rendered)


def test_rendered_metadata_targets_and_roles_are_traceable():
    split = generate_splits(DATASET_PATH)[1]
    rendered = render_fold_examples(split, HINT)

    assert len(rendered) == 400
    assert all(example.target in {"Yes", "No"} for example in rendered)
    assert [example.question_id for example in rendered[:192]] == [
        row["question_id"] for row in split.training
    ]
    assert all(example.situation == row["situation"] for row, example in zip(split.training, rendered[:192]))
    assert all(example.fold == 1 for example in rendered)
    assert {example.role for example in rendered} == {
        "training",
        "validation",
        "in_situation_evaluation",
        "cross_situation_evaluation",
    }
    assert all(example.condition == HINT for example in rendered[:192])
    assert all(example.condition is None for example in rendered[192:])
