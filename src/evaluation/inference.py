"""Gemma 3 inference and prediction recording for one evaluation run."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from src.data.instruction_formatting import (
    RenderedExample,
    render_cross_situation_evaluation_examples,
    render_in_situation_evaluation_examples,
)
from src.data.splits import generate_splits
from src.training.finetuning import (
    MODEL_CONFIGS,
    TRAINING_CONFIG,
    Gemma3Config,
    _library_versions,
    build_quantization_config,
    render_gemma_chat,
)


CONDITIONS = ("base", "no_hint", "hint")
EVALUATION_GENERATION_CONFIG = {"do_sample": False, "max_new_tokens": 8}
EvaluationCondition = Literal["base", "no_hint", "hint"]
EvaluationSetting = Literal["in_situation", "cross_situation"]


class EvaluationSequenceOverflowError(ValueError):
    """Raised when a native Gemma evaluation prompt exceeds 512 tokens."""


@dataclass(frozen=True)
class ParseResult:
    parsed_prediction: str | None
    parse_status: str
    parse_error: str | None = None


@dataclass(frozen=True)
class PredictionRecord:
    model_checkpoint: str
    model_size: str
    condition: str
    fold: int
    evaluation_setting: str
    question_id: str
    situation: str
    gold_answer: str
    raw_output: str
    parsed_prediction: str | None
    parse_status: str
    parse_error: str | None = None


def parse_prediction(raw_output: str) -> ParseResult:
    """Parse only an exact Yes/No response after whitespace stripping."""

    stripped = raw_output.strip()
    if stripped == "Yes":
        return ParseResult("Yes", "success")
    if stripped == "No":
        return ParseResult("No", "success")
    return ParseResult(
        parsed_prediction=None,
        parse_status="failure",
        parse_error="response must be exactly 'Yes' or 'No'",
    )


def resolve_adapter_path(
    training_root: str | Path,
    model_size: str,
    condition: EvaluationCondition,
    fold: int,
) -> Path | None:
    """Resolve the final adapter path, or no path for the Base condition."""

    if condition == "base":
        return None
    adapter_path = Path(training_root) / model_size / condition / f"fold_{fold}" / "adapter"
    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"Required final adapter does not exist for condition={condition}, "
            f"model_size={model_size}, fold={fold}: {adapter_path}"
        )
    return adapter_path


def evaluate_examples(
    examples: Sequence[RenderedExample],
    *,
    model: Any,
    tokenizer: Any,
    model_config: Gemma3Config,
    condition: EvaluationCondition,
    setting: EvaluationSetting,
    max_input_length: int = TRAINING_CONFIG.max_seq_length,
) -> list[PredictionRecord]:
    """Generate and record one prediction for every rendered evaluation example."""

    records: list[PredictionRecord] = []
    for example in examples:
        prompt, _ = render_gemma_chat(example, tokenizer)
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = _token_ids(encoded)
        input_length = len(input_ids)
        if input_length > max_input_length:
            raise EvaluationSequenceOverflowError(
                f"Input length {input_length} exceeds {max_input_length} tokens "
                f"(question_id={example.question_id}, fold={example.fold}, "
                f"evaluation_setting={setting}, condition={condition})"
            )

        model_inputs = _move_inputs_to_model(encoded, model)
        generated = model.generate(
            **model_inputs,
            **EVALUATION_GENERATION_CONFIG,
        )
        generated_ids = generated[0]
        completion_ids = generated_ids[input_length:]
        raw_output = tokenizer.decode(completion_ids, skip_special_tokens=True)
        parsed = parse_prediction(raw_output)
        records.append(
            PredictionRecord(
                model_checkpoint=model_config.checkpoint,
                model_size=model_config.model_size,
                condition=condition,
                fold=example.fold,
                evaluation_setting=setting,
                question_id=example.question_id,
                situation=example.situation,
                gold_answer=example.target,
                raw_output=raw_output,
                parsed_prediction=parsed.parsed_prediction,
                parse_status=parsed.parse_status,
                parse_error=parsed.parse_error,
            )
        )
    return records


def load_gemma3_for_evaluation(
    model_config: Gemma3Config,
    condition: EvaluationCondition,
    adapter_path: Path | None,
) -> tuple[Any, Any]:
    """Load the original checkpoint, optionally attaching its final PEFT adapter."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Gemma 3 evaluation requires torch and transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_config.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config.checkpoint,
        quantization_config=build_quantization_config(model_config),
        torch_dtype=getattr(torch, model_config.quantization_compute_dtype),
        device_map="auto",
    )
    if condition != "base":
        if adapter_path is None:
            raise ValueError(f"Adapter path is required for condition={condition}")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Fine-tuned Gemma 3 evaluation requires peft") from exc
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def evaluate_run(
    source_path: str | Path,
    evaluation_root: str | Path,
    training_root: str | Path,
    model_size: str,
    fold: int,
    condition: EvaluationCondition,
) -> Path:
    """Evaluate one model-size/fold/condition and write traceable artifacts."""

    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model size: {model_size!r}")
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported evaluation condition: {condition!r}")
    if fold not in range(1, 6):
        raise ValueError(f"Unsupported fold: {fold!r}")

    model_config = MODEL_CONFIGS[model_size]
    adapter_path = resolve_adapter_path(training_root, model_size, condition, fold)
    split = generate_splits(source_path)[fold]
    examples_by_setting = {
        "in_situation": render_in_situation_evaluation_examples(split),
        "cross_situation": render_cross_situation_evaluation_examples(split),
    }
    if any(len(examples) != 80 for examples in examples_by_setting.values()):
        raise ValueError(f"Fold {fold} does not contain exactly 80 instances per evaluation setting")

    output_dir = Path(evaluation_root) / model_size / condition / f"fold_{fold}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing evaluation artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_gemma3_for_evaluation(model_config, condition, adapter_path)
    records: list[PredictionRecord] = []
    for setting, examples in examples_by_setting.items():
        records.extend(
            evaluate_examples(
                examples,
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
                condition=condition,
                setting=setting,
            )
        )
    if len(records) != 160:
        raise ValueError(f"Expected 160 prediction records, produced {len(records)}")

    _write_predictions(output_dir / "predictions.csv", records)
    run_config = {
        "model": asdict(model_config),
        "condition": condition,
        "fold": fold,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "generation": EVALUATION_GENERATION_CONFIG,
        "max_input_length": TRAINING_CONFIG.max_seq_length,
        "evaluation_question_ids": {
            setting: [example.question_id for example in examples]
            for setting, examples in examples_by_setting.items()
        },
        "source_path": str(Path(source_path)),
        "library_versions": _library_versions("torch", "transformers", "peft", "bitsandbytes"),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def _write_predictions(path: Path, records: Sequence[PredictionRecord]) -> None:
    fields = [
        "model_checkpoint",
        "model_size",
        "condition",
        "fold",
        "evaluation_setting",
        "question_id",
        "situation",
        "gold_answer",
        "raw_output",
        "parsed_prediction",
        "parse_status",
        "parse_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as prediction_file:
        writer = csv.DictWriter(prediction_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _token_ids(encoded: Mapping[str, Any]) -> list[int]:
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _move_inputs_to_model(encoded: Any, model: Any) -> Any:
    if hasattr(encoded, "to"):
        return encoded.to(model.device)
    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one Gemma 3 model/fold/condition.")
    parser.add_argument("--source", type=Path, default=Path("data/annotated/pub_particularized_implicatures.csv"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--training-root", type=Path, default=Path("results/training"))
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    args = parser.parse_args()
    evaluate_run(
        args.source,
        args.evaluation_root,
        args.training_root,
        args.model_size,
        args.fold,
        args.condition,
    )


if __name__ == "__main__":
    main()