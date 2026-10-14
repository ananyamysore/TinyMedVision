"""Reproduce a measured synthetic comparison (software smoke test, not medical evidence)."""

from pathlib import Path

from tinymedvision.experiments import run_experiment
from tinymedvision.synthetic import generate_fixture


def main() -> None:
    root = Path("results/synthetic-benchmark")
    manifest = generate_fixture(root / "data", n_per_class=100, seed=17)
    result = run_experiment(manifest, root / "run", seed=17, epochs=5, batch_size=32,
                            image_size=28)
    print(f"Wrote {len(result)} measured rows under {root / 'run'}")


if __name__ == "__main__":
    main()
