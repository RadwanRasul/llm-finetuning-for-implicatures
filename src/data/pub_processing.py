"""PUB Task 3 loading and preprocessing utilities."""

from __future__ import annotations

import re

import pandas as pd
from datasets import load_dataset


PUB_REPOSITORY = "cfilt/PUB"
PUB_TASK_FILE = "data/task_3.zip"

EXPECTED_SITUATIONS = {
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
}


def split_pretext(pretext: str) -> dict[str, str | None]:
    """Split PUB pretext into context, dialogue, and implied meaning."""
    if pd.isna(pretext):
        return {
            "context": None,
            "speaker_x": None,
            "speaker_y": None,
            "implied_meaning": None,
        }

    text = str(pretext).strip()
    sections: dict[str, str] = {}
    current_key: str | None = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.match(
            r"^(Context|X|Y|Implied meaning)\s*:\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            key = match.group(1).strip().lower().replace(" ", "_")
            if key == "x":
                key = "speaker_x"
            elif key == "y":
                key = "speaker_y"

            sections[key] = match.group(2).strip()
            current_key = key
        elif current_key:
            sections[current_key] = (
                f"{sections.get(current_key, '')} {line}".strip()
            )

    return {
        "context": sections.get("context"),
        "speaker_x": sections.get("speaker_x"),
        "speaker_y": sections.get("speaker_y"),
        "implied_meaning": sections.get("implied_meaning"),
    }


def infer_situation(context: str) -> str:
    """Infer the project situation label from PUB's context text."""
    text = "" if pd.isna(context) else str(context).strip().lower()

    if "food preference" in text:
        return "food_preferences"
    if "weekend" in text:
        return "weekend_activities"
    if "book" in text:
        return "book_preferences"
    if "childhood neighbour" in text:
        return "meeting_childhood_neighbour"
    if "new neighbour" in text or "new neighbor" in text:
        return "meeting_new_neighbour"
    if "colleague" in text:
        return "colleagues_leaving_work"
    if "music preference" in text:
        return "music_preferences"
    if "travel" in text:
        return "travelling_to_meet_someone"
    if "flat" in text:
        return "buying_a_flat"
    if "job" in text:
        return "switching_jobs"

    return "unknown"


def add_question_id(
    df: pd.DataFrame,
    question_col: str = "speaker_x",
) -> pd.DataFrame:
    """Add a stable ID for each normalized question within this dataset."""
    df = df.copy()

    if "question_id" in df.columns:
        return df

    if question_col not in df.columns:
        raise KeyError(f"Column '{question_col}' not found")

    normalized = (
        df[question_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["question_id"] = pd.factorize(normalized)[0]
    return df


def prepare_pub_task3() -> pd.DataFrame:
    """Download and restructure PUB Task 3 for manual annotation."""
    dataset = load_dataset(
        PUB_REPOSITORY,
        data_files=PUB_TASK_FILE,
        split="train",
    )

    pub_df = pd.DataFrame(dataset).rename(
        columns={"correct answer": "correct_answer"}
    )

    split_df = pub_df["pretext"].apply(split_pretext).apply(pd.Series)
    pub_df = pd.concat([pub_df, split_df], axis=1).drop(columns=["pretext"])

    pub_df["situation"] = pub_df["context"].apply(infer_situation)
    pub_df = add_question_id(pub_df)

    unknown_count = int((pub_df["situation"] == "unknown").sum())
    if unknown_count:
        raise ValueError(
            f"Could not infer a situation for {unknown_count} PUB rows."
        )

    found_situations = set(pub_df["situation"].unique())
    unexpected = found_situations - EXPECTED_SITUATIONS
    missing = EXPECTED_SITUATIONS - found_situations
    if unexpected or missing:
        raise ValueError(
            "Situation validation failed. "
            f"Unexpected: {sorted(unexpected)}; missing: {sorted(missing)}"
        )

    column_order = [
        "id",
        "context",
        "situation",
        "question_id",
        "speaker_x",
        "speaker_y",
        "implied_meaning",
        "options",
        "correct_answer",
    ]
    existing = [column for column in column_order if column in pub_df.columns]
    remaining = [column for column in pub_df.columns if column not in existing]
    return pub_df[existing + remaining]
