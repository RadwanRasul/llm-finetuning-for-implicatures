# LLM Fine-Tuning for Particularized Implicatures

This repository contains code, data, and results for experiments on fine-tuning language models for the interpretation of particularized implicatures.

The experiments use Gemma 3 models and compare model performance before and after instruction-based fine-tuning, including fine-tuning with explicit pragmatic guidance.

## Data

The study uses data from Task 3 of the [PUB (Pragmatics Understanding Benchmark)](https://huggingface.co/datasets/cfilt/PUB).

The repository contains two derived data files:

- `data/processed/pub_processed.csv`: the processed PUB Task 3 data.
- `data/annotated/pub_particularized_implicatures.csv`: the manually reviewed subset used in the experiments.

The experimental subset contains 400 instances from 10 situations, with 40 instances per situation. Each situation contains 20 `Yes` and 20 `No` instances, giving a balanced dataset of 200 instances per class.

The data-processing implementation is available in `src/data/pub_processing.py` and can be run through `scripts/process_pub.py`.

## Experiment

The experiments use three instruction-tuned models from the Gemma 3 family:

- `google/gemma-3-270m-it`
- `google/gemma-3-1b-it`
- `google/gemma-3-4b-it`

Each model is evaluated under three conditions:

- **Base:** the original model without task-specific fine-tuning.
- **FT — No Hint:** instruction-based fine-tuning without explicit pragmatic guidance.
- **FT — Hint:** instruction-based fine-tuning with the implied meaning provided as explicit pragmatic guidance during training.

Both fine-tuned conditions are evaluated without the pragmatic hint.

The experiment uses five fixed situation-level folds. In each fold, eight situations are represented during fine-tuning and two situations are held out entirely. Evaluation is performed on unseen instances from situations represented during fine-tuning (**in-situation**) and unseen instances from situations not represented during fine-tuning (**cross-situation**).

Fine-tuning is performed using QLoRA.

## Repository Structure

```text
.
├── data/
│   ├── annotated/
│   └── processed/
├── notebooks/
│   └── gemma3_colab.ipynb
├── results/
│   └── aggregated/
├── scripts/
├── src/
├── tests/
├── .gitignore
├── .python-version
├── LICENSE
├── poetry.lock
├── pyproject.toml
└── README.md
```

The main directories contain:

- `data/`: processed PUB Task 3 data and the manually reviewed experimental subset.
- `notebooks/`: the Google Colab notebook used to run the experiments.
- `results/`: aggregated results.
- `scripts/`: scripts for data processing, training, evaluation and aggregation.
- `src/`: the main experiment implementation.
- `tests/`: tests for the experiment pipeline.

## Setup

The project uses Python 3.11 and Poetry for dependency management.

With `pyenv`, Python 3.11.9 can be installed and selected with:

```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

The first command is only necessary if Python 3.11.9 is not already installed.

Install the dependencies with:

```bash
poetry install
```

Commands can then be run inside the Poetry-managed environment using:

```bash
poetry run <command>
```

## Running the Experiment

The main scripts are:

```text
scripts/process_pub.py
scripts/train_gemma3.py
scripts/evaluate_gemma3.py
scripts/aggregate_results.py
```

The experiments were run in Google Colab. The notebook used to run the model sizes, conditions, and folds is available at `notebooks/gemma3_colab.ipynb`.

Access to the Gemma 3 models on Hugging Face and a Hugging Face token may be required. When using the Colab notebook, add the token as a Colab secret named `HF_TOKEN`.

## Evaluation

Accuracy is the primary evaluation metric. Macro-F1 is used as a secondary diagnostic metric.

Metrics are calculated separately for in-situation and cross-situation evaluation. For each model and condition, metrics are first calculated for each of the five folds and then summarized using the mean and sample standard deviation.

Model outputs that cannot be parsed as exactly `Yes` or `No` are counted as incorrect. Parse failures are reported separately.

The evaluation is descriptive and does not use statistical significance testing.

## Results

The aggregated results are available in:

```text
results/aggregated/
├── fold_metrics.csv
└── summary_metrics.csv
```

The full raw experiment archives are not included in this repository because of their size.

## Tests

The test suite can be run with:

```bash
poetry run pytest
```

## Data Attribution

The data included in this repository are derived from the **PUB (Pragmatics Understanding Benchmark)** dataset:

> Settaluri Sravanthi, Meet Doshi, Pavan Tankala, Rudra Murthy, Raj Dabre, and Pushpak Bhattacharyya. 2024. *PUB: A Pragmatics Understanding Benchmark for Assessing LLMs' Pragmatics Capabilities*. Findings of the Association for Computational Linguistics: ACL 2024, pages 12075–12097.

Original dataset: https://huggingface.co/datasets/cfilt/PUB

Paper: https://aclanthology.org/2024.findings-acl.719/

PUB is distributed under the MIT License. The files under `data/` in this repository are derived from PUB. Please cite the original PUB paper when using these derived data.

## License

The code in this repository is released under the MIT License. See `LICENSE` for details.
