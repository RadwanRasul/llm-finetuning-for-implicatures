"""Aggregate fold-level and summary metrics from nested Gemma 3 result ZIPs."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MODEL_SIZES = ("270m", "1b", "4b")
CONDITIONS = ("base", "no_hint", "hint")
EVALUATION_SETTINGS = ("in_situation", "cross_situation")
LABELS = ("Yes", "No")
INVALID_PREDICTION = "__PARSE_FAILURE__"
RUN_RE = re.compile(
    r"^gemma3_(?P<model>270m|1b|4b)_fold(?P<fold>[1-5])_(?P<condition>base|no_hint|hint)\.zip$"
)
GROUP_COLUMNS = ["model_size", "condition", "fold", "evaluation_setting"]
SUMMARY_COLUMNS = ["model_size", "condition", "evaluation_setting"]


def load_predictions(results_root: str | Path) -> pd.DataFrame:
    """Load and validate predictions from the three nested model archives."""

    root = Path(results_root)
    frames: list[pd.DataFrame] = []
    seen_runs: set[tuple[str, int, str]] = set()

    for model_size in MODEL_SIZES:
        outer_path = root / f"gemma3_{model_size}.zip"
        if not outer_path.is_file():
            raise FileNotFoundError(f"Missing model archive: {outer_path}")

        with zipfile.ZipFile(outer_path) as outer:
            run_names = [
                name for name in outer.namelist()
                if RUN_RE.fullmatch(Path(name).name)
            ]
            if len(run_names) != 15:
                raise ValueError(
                    f"{outer_path} must contain exactly 15 run ZIPs; found {len(run_names)}"
                )

            for run_name in run_names:
                match = RUN_RE.fullmatch(Path(run_name).name)
                assert match is not None
                run_model = match.group("model")
                fold = int(match.group("fold"))
                condition = match.group("condition")

                if run_model != model_size:
                    raise ValueError(
                        f"Run {run_name} belongs to model {run_model}, not {model_size}"
                    )

                key = (model_size, fold, condition)
                if key in seen_runs:
                    raise ValueError(f"Duplicate run archive for {key}")
                seen_runs.add(key)

                prediction_path = (
                    "predictions.csv"
                    if condition == "base"
                    else "evaluation/predictions.csv"
                )
                with zipfile.ZipFile(io.BytesIO(outer.read(run_name))) as inner:
                    if prediction_path not in inner.namelist():
                        raise ValueError(
                            f"{run_name} is missing expected {prediction_path}"
                        )
                    with inner.open(prediction_path) as predictions:
                        frame = pd.read_csv(predictions, keep_default_na=False)

                if len(frame) != 160:
                    raise ValueError(
                        f"{run_name} must contain exactly 160 predictions; found {len(frame)}"
                    )
                _validate_prediction_frame(
                    frame,
                    expected_model=model_size,
                    expected_fold=fold,
                    expected_condition=condition,
                    run_name=run_name,
                )
                frames.append(frame)

    expected_runs = {
        (model, fold, condition)
        for model in MODEL_SIZES
        for fold in range(1, 6)
        for condition in CONDITIONS
    }
    if seen_runs != expected_runs:
        missing = sorted(expected_runs - seen_runs)
        extra = sorted(seen_runs - expected_runs)
        raise ValueError(f"Run set mismatch. Missing={missing}; extra={extra}")

    predictions = pd.concat(frames, ignore_index=True)
    if len(predictions) != 7200:
        raise ValueError(f"Expected 7200 prediction records; found {len(predictions)}")
    return predictions


def calculate_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate Accuracy and Macro-F1 for every fold and evaluation setting."""

    expected_keys = {
        (model, condition, fold, setting)
        for model in MODEL_SIZES
        for condition in CONDITIONS
        for fold in range(1, 6)
        for setting in EVALUATION_SETTINGS
    }
    actual_keys = set(
        predictions[GROUP_COLUMNS].itertuples(index=False, name=None)
    )
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Fold/setting groups mismatch. Missing={missing}; extra={extra}"
        )

    records: list[dict[str, object]] = []
    grouped = predictions.groupby(GROUP_COLUMNS, sort=False, observed=True)

    for key, group in grouped:
        model_size, condition, fold, setting = key
        if len(group) != 80:
            raise ValueError(
                f"{model_size}/{condition}/fold {fold}/{setting} must contain "
                f"80 predictions; found {len(group)}"
            )

        gold_counts = group["gold_answer"].value_counts().to_dict()
        if gold_counts != {"Yes": 40, "No": 40}:
            raise ValueError(
                f"Unbalanced evaluation group for "
                f"{model_size}/{condition}/fold {fold}/{setting}: {gold_counts}"
            )

        parse_failure_mask = group["parse_status"].eq("failure")
        y_true = group["gold_answer"]
        y_pred = group["parsed_prediction"].mask(
            parse_failure_mask, INVALID_PREDICTION
        )

        # Parse failures use an out-of-label prediction so sklearn counts them
        # as incorrect while Macro-F1 remains defined over Yes and No only.
        accuracy = accuracy_score(y_true, y_pred)
        class_f1 = f1_score(
            y_true,
            y_pred,
            labels=list(LABELS),
            average=None,
            zero_division=0,
        )
        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
        parse_failures = int(parse_failure_mask.sum())
        correct = int(round(accuracy * len(group)))

        records.append(
            {
                "model_size": model_size,
                "condition": condition,
                "fold": int(fold),
                "evaluation_setting": setting,
                "n": len(group),
                "correct": correct,
                "incorrect": len(group) - correct,
                "parse_failures": parse_failures,
                "parse_failure_rate": parse_failures / len(group),
                "accuracy": accuracy,
                "f1_yes": class_f1[0],
                "f1_no": class_f1[1],
                "macro_f1": macro_f1,
            }
        )

    metrics = pd.DataFrame.from_records(records)
    if len(metrics) != 90:
        raise ValueError(f"Expected 90 fold-level metric rows; produced {len(metrics)}")

    order = {
        "model_size": MODEL_SIZES,
        "condition": CONDITIONS,
        "evaluation_setting": EVALUATION_SETTINGS,
    }
    for column, categories in order.items():
        metrics[column] = pd.Categorical(
            metrics[column], categories=categories, ordered=True
        )
    return metrics.sort_values(GROUP_COLUMNS).reset_index(drop=True)


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize five fold-level scores with mean and sample standard deviation."""

    group_sizes = fold_metrics.groupby(
        SUMMARY_COLUMNS, observed=True
    ).size()
    if len(group_sizes) != 18 or not group_sizes.eq(5).all():
        raise ValueError("Every model/condition/setting summary must contain five folds")

    fold_sets = fold_metrics.groupby(
        SUMMARY_COLUMNS, observed=True
    )["fold"].agg(lambda values: set(values))
    if not fold_sets.map(lambda folds: folds == set(range(1, 6))).all():
        raise ValueError("Every model/condition/setting summary must contain folds 1-5")

    summaries = (
        fold_metrics.groupby(SUMMARY_COLUMNS, observed=True)
        .agg(
            n_folds=("fold", "nunique"),
            n_predictions=("n", "sum"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            parse_failures_total=("parse_failures", "sum"),
        )
        .reset_index()
    )
    summaries["parse_failure_rate"] = (
        summaries["parse_failures_total"] / summaries["n_predictions"]
    )

    if len(summaries) != 18:
        raise ValueError(f"Expected 18 summary rows; produced {len(summaries)}")
    return summaries


def aggregate_results(
    results_root: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete aggregation and write auditable CSV outputs."""

    root = Path(results_root)
    destination = Path(output_dir) if output_dir is not None else root / "aggregated"

    predictions = load_predictions(root)
    fold_metrics = calculate_fold_metrics(predictions)
    summaries = summarize_fold_metrics(fold_metrics)

    destination.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(destination / "fold_metrics.csv", index=False)
    summaries.to_csv(destination / "summary_metrics.csv", index=False)
    return fold_metrics, summaries


def _validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_model: str,
    expected_fold: int,
    expected_condition: str,
    run_name: str,
) -> None:
    required = {
        "model_size",
        "condition",
        "fold",
        "evaluation_setting",
        "gold_answer",
        "parsed_prediction",
        "parse_status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{run_name} prediction CSV is missing columns: {sorted(missing)}"
        )

    checks = {
        "model_size": frame["model_size"].eq(expected_model),
        "condition": frame["condition"].eq(expected_condition),
        "fold": frame["fold"].eq(expected_fold),
        "evaluation_setting": frame["evaluation_setting"].isin(EVALUATION_SETTINGS),
        "gold_answer": frame["gold_answer"].isin(LABELS),
        "parse_status": frame["parse_status"].isin(["success", "failure"]),
    }
    for field, valid in checks.items():
        if not valid.all():
            bad = frame.loc[~valid, field].unique().tolist()
            raise ValueError(f"{run_name} contains invalid {field}: {bad}")

    success = frame["parse_status"].eq("success")
    if not frame.loc[success, "parsed_prediction"].isin(LABELS).all():
        bad = frame.loc[
            success & ~frame["parsed_prediction"].isin(LABELS),
            "parsed_prediction",
        ].unique().tolist()
        raise ValueError(f"{run_name} successful parse has invalid prediction: {bad}")

    failure_predictions = frame.loc[~success, "parsed_prediction"]
    if not failure_predictions.eq("").all():
        bad = failure_predictions.unique().tolist()
        raise ValueError(
            f"{run_name} failed parse must have an empty parsed_prediction; found {bad}"
        )
