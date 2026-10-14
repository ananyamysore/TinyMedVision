"""Small offline end-to-end run; outputs are generated locally and ignored by Git."""

from pathlib import Path

from tinymedvision.experiments import run_experiment
from tinymedvision.synthetic import generate_fixture


def main() -> None:
    manifest = generate_fixture(Path("results/example-data"), n_per_class=20, seed=17)
    run_experiment(manifest, "results/example-run", seed=17, epochs=2)
    print("See results/example-run/ for measured reports and checkpoints.")


if __name__ == "__main__":
    main()
