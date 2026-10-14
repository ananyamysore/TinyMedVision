# Reproduce a complete run

1. Install Python 3.11 or 3.12 and TinyMedVision using the README commands.
2. Generate an offline fixture, or prepare a local CSV manifest using the data
   provenance guide. For real data, record its exact source, version, access
   terms, split policy, and any patient/source IDs.
3. Run `tinymedvision run` with explicit seed, model names, image size, epochs,
   batch size, and device. The experiment seed controls Python, NumPy, PyTorch,
   split generation, and train-loader shuffling.
4. Preserve the output directory containing `config.json`, `manifest.json`,
   `results.csv`, `metrics.png`, and model checkpoints. The manifest includes
   environment and split counts. Record OS/hardware alongside runs.
5. To benchmark architecture comparisons, use identical data, split, preprocessing,
   training budget, and timing protocol; do not compare different devices as if
   their latency numbers were directly comparable.

PyTorch kernels and parallel reductions may still vary across environments.
Exact bitwise reproducibility across CPU/GPU types is not promised.
