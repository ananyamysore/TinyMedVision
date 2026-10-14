# API reference

The package is organized by responsibility under `src/tinymedvision`:

- `data.read_manifest(path, root=None)` returns validated `Sample` records;
  `validate_samples(samples, expected_classes=None)` checks file readability,
  dimensions, and observed classes. `ManifestDataset` yields grayscale tensors.
- `splitting.validate_split_groups(samples)` detects source-group leakage;
  `split_indices(labels, test_size, groups, seed)` creates seeded partitions.
- `transforms.normalize` and `transforms.corrupt` implement scaling and seeded
  corruption operations. Compression is quantization rather than JPEG.
- `models.build_model(name, num_classes=2, width=16)` returns logits models named
  `compact_cnn`, `depthwise_cnn`, or `residual_cnn`.
- `training.seed_everything`, `train_model`, `save_checkpoint`, and
  `load_checkpoint` implement seeded training and state-dict checkpoints.
- `evaluation.evaluate_predictions` computes binary predictive metrics;
  `bootstrap_auc_ci` optionally resamples independent groups.
- `calibration.calibration_metrics` returns Brier score, equal-width-bin ECE,
  and reliability curve coordinates.
- `robustness.evaluate_corruptions` compares the same examples under stress.
- `efficiency.benchmark_model` measures parameters, serialized size if supplied,
  inference time, throughput, and available runtime metadata.
- `experiments.run_experiment` orchestrates a shared-split comparison;
  `reporting.write_report` writes CSV, JSON, and a figure.

Input contracts and scientific interpretation are described in
[evaluation methodology](evaluation.md) and
[dataset provenance](dataset_provenance.md).
