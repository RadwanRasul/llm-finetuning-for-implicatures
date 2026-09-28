from pathlib import Path

import pytest

from src.data.instruction_formatting import HINT, NO_HINT, RenderedExample
from src.training.finetuning import (
    MODEL_CONFIGS,
    TRAINING_CONFIG,
    SequenceOverflowError,
    _write_training_history,
    select_fold_examples,
    tokenize_completion_examples,
)


DATASET_PATH = Path(__file__).parents[1] / "data" / "annotated" / "pub_particularized_implicatures.csv"


class FakeGemmaTokenizer:
    """Small deterministic stand-in for the native Gemma chat tokenizer."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        rendered = " ".join(f"<{message['role']}> {message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += " <assistant>"
        return rendered

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}


def test_gemma3_model_identifiers_are_frozen():
    assert {size: config.checkpoint for size, config in MODEL_CONFIGS.items()} == {
        "270m": "google/gemma-3-270m-it",
        "1b": "google/gemma-3-1b-it",
        "4b": "google/gemma-3-4b-it",
    }


def test_qlora_configuration_is_frozen():
    config = MODEL_CONFIGS["1b"]
    assert config.quantization_load_in_4bit is True
    assert config.quantization_type == "nf4"
    assert config.quantization_double is True
    assert config.quantization_compute_dtype == "float32"
    assert (config.lora_r, config.lora_alpha, config.lora_dropout) == (16, 32, 0.05)
    assert config.lora_bias == "none"
    assert config.lora_task_type == "CAUSAL_LM"
    assert config.lora_target_modules == "all-linear"


def test_training_configuration_is_frozen():
    assert TRAINING_CONFIG.num_train_epochs == 5
    assert TRAINING_CONFIG.learning_rate == 2e-4
    assert TRAINING_CONFIG.per_device_train_batch_size == 1
    assert TRAINING_CONFIG.gradient_accumulation_steps == 8
    assert TRAINING_CONFIG.max_seq_length == 512
    assert TRAINING_CONFIG.optim == "adamw_torch"
    assert TRAINING_CONFIG.lr_scheduler_type == "linear"
    assert TRAINING_CONFIG.warmup_ratio == 0.10
    assert TRAINING_CONFIG.max_grad_norm == 1.0
    assert TRAINING_CONFIG.gradient_checkpointing is True
    assert TRAINING_CONFIG.weight_decay == 0.0
    assert TRAINING_CONFIG.seed == 42


def test_training_condition_controls_only_training_formatting():
    no_hint_training, no_hint_validation = select_fold_examples(DATASET_PATH, 1, NO_HINT)
    hint_training, hint_validation = select_fold_examples(DATASET_PATH, 1, HINT)

    assert [example.question_id for example in no_hint_training] == [
        example.question_id for example in hint_training
    ]
    assert all(example.condition == NO_HINT for example in no_hint_training)
    assert all(example.condition == HINT for example in hint_training)
    assert no_hint_training[0].instruction != hint_training[0].instruction
    assert [example.instruction for example in no_hint_validation] == [
        example.instruction for example in hint_validation
    ]
    assert all(example.condition is None for example in no_hint_validation + hint_validation)


def test_completion_labels_exclude_prompt_and_implied_meaning():
    example = RenderedExample(
        question_id="q1",
        situation="situation",
        fold=1,
        role="training",
        instruction="Context: C\n\nImplied meaning: hidden\n\nAnswer only with Yes or No.",
        target="Yes",
        condition=HINT,
    )
    features = tokenize_completion_examples((example,), FakeGemmaTokenizer())
    feature = features[0]
    prompt, full = (
        FakeGemmaTokenizer().apply_chat_template(
            [{"role": "user", "content": example.instruction}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        FakeGemmaTokenizer().apply_chat_template(
            [
                {"role": "user", "content": example.instruction},
                {"role": "assistant", "content": example.target},
            ],
            tokenize=False,
            add_generation_prompt=False,
        ),
    )
    prompt_length = len(prompt.split())
    assert feature["labels"][:prompt_length] == [-100] * prompt_length
    assert feature["labels"][prompt_length:] == list(range(prompt_length, len(full.split())))
    assert len(feature["labels"]) == len(full.split())


def test_targets_are_exactly_yes_or_no():
    no_hint_training, validation = select_fold_examples(DATASET_PATH, 1, NO_HINT)
    assert {example.target for example in no_hint_training + validation} <= {"Yes", "No"}


def test_sequence_overflow_fails_without_truncation():
    example = RenderedExample(
        question_id="overflow-q",
        situation="situation",
        fold=3,
        role="validation",
        instruction="word " * 20,
        target="No",
    )
    with pytest.raises(SequenceOverflowError, match="question_id=overflow-q.*fold=3.*role=validation"):
        tokenize_completion_examples((example,), FakeGemmaTokenizer(), max_seq_length=5)


def test_training_history_preserves_events_and_leaves_unsynchronised_values_blank(tmp_path):
    history_path = tmp_path / "training_history.csv"
    log_history = (
        {"step": 1, "epoch": 0.2, "loss": 1.5},
        {"step": 2, "epoch": 1.0, "eval_loss": 1.1},
        {"step": 3, "epoch": 1.2, "loss": 0.9},
        {"step": 4, "epoch": 2.0, "eval_loss": 0.7},
        {"step": 5, "epoch": 2.2, "loss": None, "eval_loss": None},
    )

    _write_training_history(
        history_path,
        log_history,
        MODEL_CONFIGS["1b"],
        NO_HINT,
        1,
    )

    rows = history_path.read_text(encoding="utf-8").splitlines()
    assert rows == [
        "model_checkpoint,model_size,condition,fold,epoch,training_step,training_loss,validation_loss",
        "google/gemma-3-1b-it,1b,no_hint,1,0.2,1,1.5,",
        "google/gemma-3-1b-it,1b,no_hint,1,1.0,2,,1.1",
        "google/gemma-3-1b-it,1b,no_hint,1,1.2,3,0.9,",
        "google/gemma-3-1b-it,1b,no_hint,1,2.0,4,,0.7",
    ]