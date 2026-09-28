"""Gemma 3 QLoRA fine-tuning for one experiment run."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from src.data.instruction_formatting import HINT, NO_HINT, RenderedExample, render_training_examples, render_validation_examples
from src.data.splits import FoldSplit, generate_splits

MODEL_CHECKPOINTS = {
    "270m": "google/gemma-3-270m-it",
    "1b": "google/gemma-3-1b-it",
    "4b": "google/gemma-3-4b-it",
}
ModelSize = Literal["270m", "1b", "4b"]
Condition = Literal["no_hint", "hint"]

@dataclass(frozen=True)
class Gemma3Config:
    """Frozen model selection for the three Gemma 3 experiment sizes."""

    model_size: ModelSize
    checkpoint: str
    quantization_load_in_4bit: bool = True
    quantization_type: str = "nf4"
    quantization_double: bool = True
    quantization_compute_dtype: str = "float32"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    lora_target_modules: str = "all-linear"


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen training settings shared by every fine-tuning run."""

    num_train_epochs: int = 5
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 512
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    weight_decay: float = 0.0
    seed: int = 42


MODEL_CONFIGS = {
    size: Gemma3Config(model_size=size, checkpoint=checkpoint)
    for size, checkpoint in MODEL_CHECKPOINTS.items()
}
TRAINING_CONFIG = TrainingConfig()


class SequenceOverflowError(ValueError):
    """Raised when native Gemma formatting exceeds the frozen sequence limit."""


def select_fold_examples(
    source_path: str | Path,
    fold: int,
    condition: Condition,
) -> tuple[tuple[RenderedExample, ...], tuple[RenderedExample, ...]]:
    """Load one fixed fold and return condition-specific training plus No-Hint validation."""

    if condition not in (NO_HINT, HINT):
        raise ValueError(f"Unsupported training condition: {condition!r}")
    split = generate_splits(source_path)[fold]
    return render_training_examples(split, condition), render_validation_examples(split)


def render_gemma_chat(
    example: RenderedExample,
    tokenizer: Any,
) -> tuple[str, str]:
    """Render canonical formatter content through Gemma's native chat template."""

    user_message = [{"role": "user", "content": example.instruction}]
    conversation = user_message + [{"role": "assistant", "content": example.target}]
    prompt = tokenizer.apply_chat_template(
        user_message,
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt, full


def tokenize_completion_examples(
    examples: Sequence[RenderedExample],
    tokenizer: Any,
    max_seq_length: int = TRAINING_CONFIG.max_seq_length,
) -> list[dict[str, list[int]]]:
    """Tokenize examples and mask every prompt token from the supervised loss."""

    tokenized: list[dict[str, list[int]]] = []
    for example in examples:
        prompt, full = render_gemma_chat(example, tokenizer)
        prompt_ids = _token_ids(tokenizer(prompt, add_special_tokens=False))
        full_ids = _token_ids(tokenizer(full, add_special_tokens=False))
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Gemma chat template did not preserve the user prefix for "
                f"question_id={example.question_id!r}"
            )
        if len(full_ids) > max_seq_length:
            condition = example.condition or NO_HINT
            raise SequenceOverflowError(
                f"Sequence length {len(full_ids)} exceeds {max_seq_length} tokens "
                f"(question_id={example.question_id}, fold={example.fold}, "
                f"role={example.role}, condition={condition})"
            )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if not any(label != -100 for label in labels):
            raise ValueError(f"No supervised completion tokens for {example.question_id!r}")
        tokenized.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    return tokenized


def _token_ids(encoded: Mapping[str, Any]) -> list[int]:
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def build_quantization_config(model_config: Gemma3Config) -> Any:
    """Build the frozen bitsandbytes 4-bit configuration."""

    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("QLoRA requires torch and transformers to be installed") from exc
    return BitsAndBytesConfig(
        load_in_4bit=model_config.quantization_load_in_4bit,
        bnb_4bit_quant_type=model_config.quantization_type,
        bnb_4bit_use_double_quant=model_config.quantization_double,
        bnb_4bit_compute_dtype=getattr(torch, model_config.quantization_compute_dtype),
    )


def train_run(
    source_path: str | Path,
    output_root: str | Path,
    model_size: ModelSize,
    fold: int,
    condition: Condition,
) -> Path:
    """Execute and persist one five-epoch Gemma 3 QLoRA run."""

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Gemma 3 QLoRA requires torch, datasets, transformers, and peft") from exc

    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model size: {model_size!r}")
    if condition not in (NO_HINT, HINT):
        raise ValueError(f"Unsupported training condition: {condition!r}")

    model_config = MODEL_CONFIGS[model_size]
    training_config = TRAINING_CONFIG
    run_dir = Path(output_root) / model_size / condition / f"fold_{fold}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_examples, validation_examples = select_fold_examples(source_path, fold, condition)
    tokenizer = AutoTokenizer.from_pretrained(model_config.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_features = Dataset.from_list(tokenize_completion_examples(train_examples, tokenizer))
    validation_features = Dataset.from_list(
        tokenize_completion_examples(validation_examples, tokenizer, training_config.max_seq_length)
    )
    adapter_path = run_dir / "adapter"
    run_config = {
        "model": asdict(model_config),
        "training": asdict(training_config),
        "condition": condition,
        "fold": fold,
        "source_path": str(Path(source_path)),
        "training_question_ids": [example.question_id for example in train_examples],
        "validation_question_ids": [example.question_id for example in validation_examples],
        "adapter_path": str(adapter_path),
        "library_versions": _library_versions("torch", "transformers", "peft", "bitsandbytes", "trl"),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    model = AutoModelForCausalLM.from_pretrained(
        model_config.checkpoint,
        quantization_config=build_quantization_config(model_config),
        torch_dtype=getattr(torch, model_config.quantization_compute_dtype),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            bias=model_config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=model_config.lora_target_modules,
        ),
    )
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer_checkpoints"),
        num_train_epochs=training_config.num_train_epochs,
        learning_rate=training_config.learning_rate,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        max_grad_norm=training_config.max_grad_norm,
        gradient_checkpointing=training_config.gradient_checkpointing,
        weight_decay=training_config.weight_decay,
        optim=training_config.optim,
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_ratio=training_config.warmup_ratio,
        seed=training_config.seed,
        bf16=False,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        report_to=[],
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=validation_features,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
    )
    trainer.train()
    trainer.save_model(str(adapter_path))
    _write_training_history(run_dir / "training_history.csv", trainer.state.log_history, model_config, condition, fold)
    return adapter_path


def _library_versions(*packages: str) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _write_training_history(
    path: Path,
    log_history: Sequence[Mapping[str, Any]],
    model_config: Gemma3Config,
    condition: Condition,
    fold: int,
) -> None:
    fields = [
        "model_checkpoint",
        "model_size",
        "condition",
        "fold",
        "epoch",
        "training_step",
        "training_loss",
        "validation_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for record in log_history:
            if record.get("loss") is None and record.get("eval_loss") is None:
                continue
            writer.writerow(
                {
                    "model_checkpoint": model_config.checkpoint,
                    "model_size": model_config.model_size,
                    "condition": condition,
                    "fold": fold,
                    "epoch": record.get("epoch", ""),
                    "training_step": record.get("step", ""),
                    "training_loss": record.get("loss", ""),
                    "validation_loss": record.get("eval_loss", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Gemma 3 QLoRA fine-tuning fold.")
    parser.add_argument("--source", type=Path, default=Path("data/annotated/pub_particularized_implicatures.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("results/training"))
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--condition", choices=(NO_HINT, HINT), required=True)
    args = parser.parse_args()
    train_run(args.source, args.output_root, args.model_size, args.fold, args.condition)


if __name__ == "__main__":
    main()