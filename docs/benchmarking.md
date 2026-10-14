# Benchmark methodology

`benchmarks/run_synthetic.py` measures each architecture on a shared generated
split with the same image resolution, seed, optimizer, epoch count, and batch
size. It records parameter counts, serialized checkpoint size, median and 95th
percentile inference latency, throughput, training time, host platform, Python,
PyTorch, and timing settings. Results are written at runtime to `results/` and
are not checked in as fabricated reference numbers.

Latency is measured in-process with warm-up followed by repeated forward passes.
Data loading and preprocessing are not included in model inference latency.
Small CPU timings are sensitive to scheduler load, thread configuration, and
thermal state; compare runs only with hardware and software metadata attached.
CUDA peak memory is recorded only for CUDA runs; CPU peak memory is unavailable
in this implementation. The model family comparison is controlled but parameter
count and compute are not equalized, so differences cannot be attributed to one
architectural property alone.
