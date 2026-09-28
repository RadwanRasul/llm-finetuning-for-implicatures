from __future__ import annotations

import csv
import io
import zipfile

import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.evaluation.aggregation import (
    INVALID_PREDICTION,
    calculate_fold_metrics,
    load_predictions,
)


def _row(gold: str, prediction: str = "", status: str = "failure") -> dict[str, object]:
    return {
        "model_size": "270m",
        "condition": "base",
        "fold": 1,
        "evaluation_setting": "in_situation",
        "gold_answer": gold,
        "parsed_prediction": prediction,
        "parse_status": status,
    }


def _prediction_csv(model: str, fold: int, condition: str) -> bytes:
    fields = [
        "model_size",
        "condition",
        "fold",
        "evaluation_setting",
        "gold_answer",
        "parsed_prediction",
        "parse_status",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for setting in ("in_situation", "cross_situation"):
        for gold in ("Yes", "No"):
            for index in range(40):
                failure = (
                    model == "270m"
                    and fold == 1
                    and condition == "base"
                    and setting == "in_situation"
                    and gold == "Yes"
                    and index == 0
                )
                writer.writerow(
                    {
                        "model_size": model,
                        "condition": condition,
                        "fold": fold,
                        "evaluation_setting": setting,
                        "gold_answer": gold,
                        "parsed_prediction": "" if failure else gold,
                        "parse_status": "failure" if failure else "success",
                    }
                )
    return output.getvalue().encode("utf-8")


def _inner_zip(prediction_path: str, predictions: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(prediction_path, predictions)
    return buffer.getvalue()


def _write_synthetic_result_archives(root) -> None:
    for model in ("270m", "1b", "4b"):
        with zipfile.ZipFile(root / f"gemma3_{model}.zip", "w") as outer:
            for fold in range(1, 6):
                for condition in ("base", "no_hint", "hint"):
                    prediction_path = (
                        "predictions.csv"
                        if condition == "base"
                        else "evaluation/predictions.csv"
                    )
                    run_name = f"gemma3_{model}_fold{fold}_{condition}.zip"
                    outer.writestr(
                        run_name,
                        _inner_zip(
                            prediction_path,
                            _prediction_csv(model, fold, condition),
                        ),
                    )


def test_sklearn_macro_f1_counts_parse_failure_as_incorrect() -> None:
    y_true = ["Yes", "Yes", "No", "No"]
    y_pred = ["Yes", INVALID_PREDICTION, "Yes", "No"]

    class_f1 = f1_score(
        y_true,
        y_pred,
        labels=["Yes", "No"],
        average=None,
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=["Yes", "No"],
        average="macro",
        zero_division=0,
    )

    assert class_f1[0] == pytest.approx(0.5)
    assert class_f1[1] == pytest.approx(2 / 3)
    assert macro_f1 == pytest.approx((0.5 + 2 / 3) / 2)


def test_fold_metrics_require_all_expected_groups() -> None:
    rows = []
    for gold in ("Yes", "No"):
        for _ in range(40):
            rows.append(_row(gold, gold, "success"))

    with pytest.raises(ValueError, match="groups mismatch"):
        calculate_fold_metrics(pd.DataFrame(rows))


def test_loader_supports_base_and_finetuned_nested_zip_layouts(tmp_path) -> None:
    _write_synthetic_result_archives(tmp_path)

    predictions = load_predictions(tmp_path)

    assert len(predictions) == 7200
    failures = predictions[predictions["parse_status"].eq("failure")]
    assert len(failures) == 1
    failure = failures.iloc[0]
    assert failure["model_size"] == "270m"
    assert failure["condition"] == "base"
    assert failure["fold"] == 1
    assert failure["parsed_prediction"] == ""
