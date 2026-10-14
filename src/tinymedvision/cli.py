"""Command-line entry point for fixture generation and reproducible experiments."""

from __future__ import annotations

import argparse

from .data import read_manifest, validate_samples
from .experiments import run_experiment
from .synthetic import generate_fixture


def main() -> None:
    parser = argparse.ArgumentParser(prog="tinymedvision")
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("generate-fixture", help="create a deterministic toy image dataset")
    fixture.add_argument("output")
    fixture.add_argument("--n-per-class", type=int, default=20)
    fixture.add_argument("--seed", type=int, default=17)
    validate = sub.add_parser("validate", help="check manifest paths, images, labels, and groups")
    validate.add_argument("manifest")
    run = sub.add_parser("run", help="train and evaluate model(s) from a CSV manifest")
    run.add_argument("manifest")
    run.add_argument("--output", default="results/run")
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--epochs", type=int, default=2)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--image-size", type=int, default=28)
    run.add_argument("--device", default="cpu")
    run.add_argument("--robustness", action="store_true",
                     help="evaluate the same test images under controlled corruptions")
    run.add_argument("--models", nargs="+", default=["compact_cnn", "depthwise_cnn", "residual_cnn"])
    args = parser.parse_args()
    if args.command == "generate-fixture":
        print(generate_fixture(args.output, n_per_class=args.n_per_class, seed=args.seed))
    elif args.command == "validate":
        samples = read_manifest(args.manifest)
        validate_samples(samples)
        print(f"Validated {len(samples)} images across labels {sorted({s.label for s in samples})}")
    else:
        for row in run_experiment(args.manifest, args.output, seed=args.seed, epochs=args.epochs,
                                  batch_size=args.batch_size, image_size=args.image_size,
                                  models=tuple(args.models), device=args.device,
                                  robustness=args.robustness):
            print(row)


if __name__ == "__main__":
    main()
