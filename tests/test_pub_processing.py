import pandas as pd
import pytest

from src.data.pub_processing import (
    EXPECTED_SITUATIONS,
    add_question_id,
    infer_situation,
    split_pretext,
)


def test_split_pretext() -> None:
    pretext = """Context: X wants to know what sorts of books Y likes to read.
X: Want to go to the library to browse?
Y: More than you know.
Implied meaning: Yes, Y would very much like to go."""

    result = split_pretext(pretext)

    assert result == {
        "context": "X wants to know what sorts of books Y likes to read.",
        "speaker_x": "Want to go to the library to browse?",
        "speaker_y": "More than you know.",
        "implied_meaning": "Yes, Y would very much like to go.",
    }


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ("X asks about Y's food preferences.", "food_preferences"),
        ("X asks what Y does on the weekend.", "weekend_activities"),
        ("X asks about Y's book preferences.", "book_preferences"),
        (
            "X meets a childhood neighbour.",
            "meeting_childhood_neighbour",
        ),
        ("X meets a new neighbour.", "meeting_new_neighbour"),
        ("X asks about a colleague leaving work.", "colleagues_leaving_work"),
        ("X asks about Y's music preferences.", "music_preferences"),
        ("X asks about travelling to meet someone.", "travelling_to_meet_someone"),
        ("X asks about buying a flat.", "buying_a_flat"),
        ("X asks about switching jobs.", "switching_jobs"),
    ],
)
def test_infer_situation(context: str, expected: str) -> None:
    assert infer_situation(context) == expected


def test_infer_situation_returns_unknown_for_unmatched_context() -> None:
    assert infer_situation("An unrelated context.") == "unknown"


def test_add_question_id_normalizes_question_text() -> None:
    df = pd.DataFrame(
        {
            "speaker_x": [
                "Do you like books?",
                "  do you like books?  ",
                "Do you like music?",
            ]
        }
    )

    result = add_question_id(df)

    assert result["question_id"].tolist() == [0, 0, 1]


def test_add_question_id_requires_question_column() -> None:
    df = pd.DataFrame({"speaker_y": ["Yes"]})

    with pytest.raises(KeyError, match="speaker_x"):
        add_question_id(df)
