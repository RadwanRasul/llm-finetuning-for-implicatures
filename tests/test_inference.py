from pathlib import Path

import pytest

from src.data.instruction_formatting import (
    render_cross_situation_evaluation_examples,
    render_in_situation_evaluation_examples,
)
from src.data.splits import generate_splits
from src.evaluation.inference import (
    EVALUATION_GENERATION_CONFIG,
    EvaluationSequenceOverflowError,
    evaluate_examples,
    parse_prediction,
    resolve_adapter_path,
)
from src.training.finetuning import MODEL_CONFIGS


DATASET_PATH = Path(__file__).parents[1] / "data" / "annotated" / "pub_particularized_implicatures.csv"


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self, response="Yes"):
        self.response = response
        self.prompts = []
        self.roles = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        self.roles.append([message["role"] for message in messages])
        rendered = " ".join(message["content"] for message in messages)
        if add_generation_prompt:
            rendered += " <assistant>"
        return rendered

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert add_special_tokens is False
        assert return_tensors == "pt"
        self.prompts.append(text)
        return FakeBatch(input_ids=[[1, 2, 3]], attention_mask=[[1, 1, 1]])

    def decode(self, _tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return self.response


class FakeModel:
    device = "cpu"

    def __init__(self, response_ids=(4,)):
        self.response_ids = response_ids
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3, *self.response_ids]]


def test_parser_is_strict_and_preserves_whitespace_tolerance():
    assert parse_prediction(" Yes \n").parsed_prediction == "Yes"
    assert parse_prediction("No").parsed_prediction == "No"
    for response in ("Yes, because...", "The answer is Yes.", "No.", "YES", "yes", "Yes No"):
        result = parse_prediction(response)
        assert result.parsed_prediction is None
        assert result.parse_status == "failure"


def test_generation_config_is_frozen():
    assert EVALUATION_GENERATION_CONFIG == {"do_sample": False, "max_new_tokens": 8}


def test_evaluation_settings_use_existing_fold_members():
    split = generate_splits(DATASET_PATH)[1]
    in_situation = render_in_situation_evaluation_examples(split)
    cross_situation = render_cross_situation_evaluation_examples(split)

    assert len(in_situation) == 80
    assert len(cross_situation) == 80
    assert [example.question_id for example in in_situation] == [
        row["question_id"] for row in split.in_situation_evaluation
    ]
    assert [example.question_id for example in cross_situation] == [
        row["question_id"] for row in split.cross_situation_evaluation
    ]


def test_evaluation_uses_existing_no_hint_examples_and_preserves_failures():
    split = generate_splits(DATASET_PATH)[1]
    examples = render_in_situation_evaluation_examples(split)[:2]
    tokenizer = FakeTokenizer(response="Yes, because...")
    model = FakeModel()

    records = evaluate_examples(
        examples,
        model=model,
        tokenizer=tokenizer,
        model_config=MODEL_CONFIGS["1b"],
        condition="hint",
        setting="in_situation",
    )

    assert len(records) == 2
    assert all(record.parse_status == "failure" for record in records)
    assert all(record.parsed_prediction is None for record in records)
    assert all(record.raw_output == "Yes, because..." for record in records)
    assert all("implied_meaning" not in prompt for prompt in tokenizer.prompts)
    assert [roles for roles in tokenizer.roles if roles == ["user"]] == [["user"], ["user"]]
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 8


def test_input_overflow_fails_with_traceable_metadata():
    split = generate_splits(DATASET_PATH)[1]
    example = render_in_situation_evaluation_examples(split)[0]
    with pytest.raises(EvaluationSequenceOverflowError, match="question_id=.*fold=1.*evaluation_setting=in_situation.*condition=base"):
        evaluate_examples(
            (example,),
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            model_config=MODEL_CONFIGS["1b"],
            condition="base",
            setting="in_situation",
            max_input_length=2,
        )


def test_adapter_paths_are_final_and_condition_specific(tmp_path):
    no_hint = tmp_path / "1b" / "no_hint" / "fold_1" / "adapter"
    hint = tmp_path / "1b" / "hint" / "fold_1" / "adapter"
    no_hint.mkdir(parents=True)
    hint.mkdir(parents=True)

    assert resolve_adapter_path(tmp_path, "1b", "base", 1) is None
    assert resolve_adapter_path(tmp_path, "1b", "no_hint", 1) == no_hint
    assert resolve_adapter_path(tmp_path, "1b", "hint", 1) == hint
    with pytest.raises(FileNotFoundError, match="condition=no_hint.*fold=2"):
        resolve_adapter_path(tmp_path, "1b", "no_hint", 2)