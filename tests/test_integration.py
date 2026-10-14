import json

import pandas as pd

from tinymedvision.experiments import run_experiment


def test_end_to_end_smoke(fixture_manifest, tmp_path):
    output = tmp_path / "run"
    rows = run_experiment(fixture_manifest, output, seed=5, epochs=1, batch_size=4,
                          image_size=16, models=("compact_cnn",))
    assert len(rows) == 1
    assert {"results.csv", "config.json", "manifest.json", "metrics.png"} <= {
        p.name for p in output.iterdir()}
    data = pd.read_csv(output / "results.csv")
    assert data.loc[0, "model"] == "compact_cnn"
    assert json.loads((output / "manifest.json").read_text())["seed"] == 5
