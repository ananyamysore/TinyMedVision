# TinyMedVision

**A reproducible benchmark for efficiency, robustness, and reliability in medical image classification.**

TinyMedVision is an open-source research project investigating how lightweight convolutional neural networks trade off predictive performance, computational efficiency, and robustness under controlled image-quality changes.

Rather than evaluating models using accuracy alone, TinyMedVision compares compact architectures across multiple dimensions: discrimination, sensitivity, calibration, inference latency, model size, and performance degradation under image perturbations.

The goal is to make small medical-imaging experiments easier to reproduce, audit, and compare.

> **Research status:** This repository provides a framework for controlled experiments. Performance claims will be reported only after the corresponding experiments have been executed and their configurations documented. Results from synthetic data are used for software validation, not as evidence of clinical performance.

## Research questions

TinyMedVision is organized around five questions:

1. **Efficiency:** How do compact CNN architectures compare in parameter count, model size, inference latency, and memory consumption?
2. **Predictive performance:** What trade-offs emerge between computational cost and metrics such as ROC-AUC, average precision, sensitivity, and specificity?
3. **Image-quality robustness:** How does performance change under controlled variations in image resolution, noise, blur, compression, brightness, and contrast?
4. **Predictive reliability:** Do model confidence estimates remain meaningful as image quality changes?
5. **Reproducibility:** Can the same experiment be rerun with documented data splits, configurations, random seeds, and software versions?

These are research questions, not predetermined conclusions. The benchmark is designed to allow results to support, qualify, or contradict the initial hypotheses.

## Project scope

TinyMedVision focuses on lightweight image-classification models and reproducible experimental methodology.

| Component | Scope |
|---|---|
| Architectures | Compact baseline CNN, depthwise-separable CNN, compact residual CNN |
| Training | Configurable, reproducible training and checkpointing |
| Evaluation | Discrimination, threshold-dependent metrics, calibration |
| Robustness | Controlled image-quality perturbations |
| Efficiency | Parameters, serialized size, latency, and measurable memory usage |
| Validation | Unit, integration, regression, and synthetic-data tests |
| Reporting | Machine-readable results, figures, and run manifests |

The initial experiments focus on binary classification using a permitted, documented chest-radiograph dataset, such as PneumoniaMNIST, subject to verification of its current access terms and intended use.

The architecture is designed to accommodate additional datasets later, provided their provenance, labels, and evaluation splits are documented.

## Architecture

The package is organized into modular components under `src/tinymedvision/`.

```text
TinyMedVision/
├── src/
│   └── tinymedvision/
│       ├── __init__.py
│       ├── data.py
│       ├── transforms.py
│       ├── models.py
│       ├── training.py
│       ├── evaluation.py
│       ├── calibration.py
│       ├── robustness.py
│       ├── efficiency.py
│       ├── splitting.py
│       ├── experiments.py
│       ├── reporting.py
│       └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── examples/
├── configs/
├── docs/
├── benchmarks/
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

The final repository may differ as implementation evolves. Documentation should reflect the modules that actually exist.

## Experimental methodology

### 1. Model comparison

The benchmark compares a small set of lightweight CNN architectures:

- **Baseline CNN:** a simple reference model.
- **Depthwise-separable CNN:** an architecture using depthwise and pointwise convolutions to reduce computational cost.
- **Compact residual CNN:** a small residual architecture for comparison.

Models should use comparable datasets, input resolutions, data splits, and training budgets wherever practical. Any unavoidable differences must be recorded.

The objective is not to declare a universal winner, but to measure the trade-offs under specified experimental conditions.

### 2. Predictive evaluation

Where supported by the task and dataset, evaluation includes:

- ROC-AUC
- Average precision and precision-recall curves
- Sensitivity and specificity
- Precision and negative predictive value
- Confusion matrices
- Brier score
- Calibration curves
- Confidence intervals where statistically appropriate

Metrics that depend on a classification threshold must use a prespecified threshold or one selected on validation data. The test set must not be used to optimize a threshold and then serve as an unbiased evaluation of that same choice.

Undefined metrics and missing classes should be reported explicitly rather than silently converted to zero.

### 3. Controlled robustness evaluation

Models can be evaluated under configurable image perturbations, including:

- Reduced image resolution
- Gaussian noise
- Blur
- Compression artifacts
- Brightness changes
- Contrast changes

Perturbation parameters and random seeds should be recorded for each run.

Performance should be compared against the corresponding unperturbed baseline. The analysis should report both absolute performance and the change associated with each perturbation.

These experiments characterize behavior under simulated image-quality changes. They do not establish robustness to real clinical distribution shifts, different hospitals, different patient populations, or changes in acquisition equipment.

### 4. Efficiency evaluation

TinyMedVision measures computational cost alongside predictive performance.

| Measurement | Purpose |
|---|---|
| Parameter count | Quantifies model parameterization |
| Serialized model size | Measures checkpoint storage requirements |
| Inference latency | Measures time required to produce predictions |
| Throughput | Measures predictions per unit time under specified conditions |
| Memory usage | Quantifies memory consumption where measurable |
| Training time | Records the computational cost of training |

Benchmark reports should include hardware, operating system, library versions, input dimensions, batch size, warm-up iterations, and timing methodology.

Preprocessing time should be distinguished from model inference time where feasible. Results measured on different hardware should not be treated as directly equivalent.

No benchmark values should be reported until the corresponding measurements have actually been collected.

## Reproducibility

Each experiment should record:

- Dataset identity, version, and provenance
- Data-split assignments and grouping identifiers where available
- Model architecture and configuration
- Random seeds
- Training hyperparameters
- Preprocessing and augmentation settings
- Software and dependency versions
- Hardware information
- Checkpoint locations
- Evaluation and robustness metrics
- Warnings, failed runs, and skipped experiments

Configuration files should make experiments reproducible without requiring edits to the underlying source code.

Checkpoints should be reusable for evaluation without retraining. Large datasets and model artifacts should generally remain outside version control.

## Data and responsible use

TinyMedVision is intended for methodological research and educational experimentation.

- Use only datasets whose access terms and intended use have been reviewed.
- Document dataset sources, label definitions, and known limitations.
- Do not commit restricted or unnecessarily large medical-image datasets.
- Preserve patient or study identifiers when available and appropriate.
- Prevent related images or derived versions of the same source image from crossing data partitions.
- Do not claim patient-level separation when the required identifiers are unavailable.
- Treat dataset labels as labels, not infallible clinical ground truth.

**TinyMedVision is not a medical device and is not intended for diagnosis, treatment decisions, or direct clinical deployment.** Benchmark performance alone does not establish clinical validity, safety, generalizability, or clinical utility.

## Installation

TinyMedVision requires Python 3.11 or later.

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/TinyMedVision.git
cd TinyMedVision
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade pip and install the package in editable mode with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

These commands assume that the repository's `pyproject.toml` defines the package and a `dev` optional-dependency group. Confirm that the dependency configuration exists before relying on the commands.

## Quick start

Once the corresponding interfaces have been implemented, the intended workflow is:

1. Validate a dataset and its metadata.
2. Load an experiment configuration.
3. Train a selected architecture.
4. Evaluate the saved checkpoint.
5. Run controlled robustness tests.
6. Measure efficiency.
7. Export a report and run manifest.

The exact commands depend on the implemented CLI. A possible interface is:

```bash
python -m tinymedvision.cli validate-data \
  --config configs/baseline.yaml

python -m tinymedvision.cli train \
  --config configs/baseline.yaml

python -m tinymedvision.cli evaluate \
  --config configs/baseline.yaml

python -m tinymedvision.cli benchmark \
  --config configs/benchmark.yaml
```

These are proposed command examples, not a claim that the commands already exist. Update this section to match the actual CLI and verify every example before release.

## Testing

Run the unit and integration test suite:

```bash
python -m pytest tests/ -v
```

Run linting if the repository configures Ruff:

```bash
python -m ruff check .
```

Run a small end-to-end synthetic experiment:

```bash
python -m tinymedvision.cli smoke-test
```

The smoke-test command should be implemented and verified before being documented as available. Automated tests should run without downloading external datasets.

## Benchmark reporting

Every published comparison should identify:

1. Dataset and split used.
2. Architecture and model configuration.
3. Training budget and preprocessing.
4. Hardware and software environment.
5. Metrics and uncertainty estimates.
6. Perturbation settings, where applicable.
7. Deviations from the planned protocol.
8. Failed or excluded runs.

A benchmark should not omit a failed experiment simply because its result is unfavorable. If results are aggregated over multiple runs, the aggregation method and variability should be reported.

## Limitations

TinyMedVision has several important limitations:

- Simulated image perturbations do not reproduce every source of clinical distribution shift.
- Dataset labels may contain noise or reflect dataset-specific labeling procedures.
- A model may learn acquisition artifacts or other shortcuts rather than clinically meaningful image features.
- Small test sets can produce uncertain or unstable metric estimates.
- Calibration and subgroup analyses depend on sample size and metadata availability.
- CPU inference latency depends on the hardware and software environment.
- Results from a single dataset cannot establish broad clinical generalizability.
- Passing software tests does not establish clinical validity.

These limitations should be revisited as new datasets, analyses, and experiments are added.

## Contributing

Contributions should prioritize reproducibility, scientific correctness, and maintainability.

Before submitting a change:

- Add tests for new functionality and relevant edge cases.
- Preserve existing configuration compatibility where possible.
- Document methodological assumptions and changes to evaluation behavior.
- Record any new dataset or dependency requirements.
- Avoid adding benchmark claims without reproducible measurements.
- Run the relevant tests and lint checks.

## Citation

If you use TinyMedVision in research, cite the repository version and the relevant datasets and methods.

A `CITATION.cff` file should be maintained with the actual repository metadata. Do not invent a DOI, publication, or archival identifier.

## License

The project's code license is specified in the repository's `LICENSE` file. Dataset licenses and usage terms are separate and must be checked independently.

## Project status

TinyMedVision is a research software project. Its maturity should be described using verifiable milestones, such as implemented modules, passing tests, completed experiments, and documented dataset provenance.

**The central principle is simple: evaluate efficiency and reliability alongside predictive performance, and make every reported result reproducible.**
