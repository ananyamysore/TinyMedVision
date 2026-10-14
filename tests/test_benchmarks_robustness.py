import torch

from tinymedvision.efficiency import benchmark_model
from tinymedvision.models import build_model
from tinymedvision.robustness import evaluate_corruptions


def test_efficiency_output_schema():
    result = benchmark_model(build_model("compact_cnn", width=4), input_shape=(1, 16, 16),
                             warmup=1, iterations=3)
    assert result["parameters"] > 0
    assert result["latency_median_ms"] >= 0
    assert result["throughput_images_per_second"] > 0
    assert result["input_shape"] == [1, 16, 16]


def test_corruption_results_have_shared_conditions():
    model = build_model("compact_cnn", width=4)
    results = evaluate_corruptions(model, torch.rand(4, 1, 16, 16), torch.tensor([0, 1, 0, 1]),
                                   kinds=("noise",), severities=(0.5,), seed=8)
    assert len(results) == 1
    assert results[0]["corruption"] == "noise"
    assert results[0]["n"] == 4
