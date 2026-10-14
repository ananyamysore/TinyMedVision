import pandas as pd
import pytest

from tinymedvision.data import read_manifest, validate_samples
from tinymedvision.splitting import split_indices, validate_split_groups
from tinymedvision.data import Sample
from pathlib import Path


def test_generated_manifest_validates(fixture_manifest):
    samples = read_manifest(fixture_manifest)
    validate_samples(samples, expected_classes={0, 1})
    assert len(samples) == 12


def test_missing_file_raises(tmp_path):
    sample = Sample(tmp_path / "missing.png", 0)
    with pytest.raises(FileNotFoundError):
        validate_samples([sample])


def test_bad_manifest_columns_raise(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"filename": ["x"], "target": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="path"):
        read_manifest(path)


def test_group_split_is_disjoint_and_reproducible():
    labels = [0, 1] * 10
    groups = [f"patient-{i // 2}" for i in range(20)]
    train, test = split_indices(labels, groups=groups, seed=3)
    assert set(groups[i] for i in train).isdisjoint(groups[i] for i in test)
    assert (train == split_indices(labels, groups=groups, seed=3)[0]).all()


def test_declared_group_leakage_is_rejected(tmp_path):
    samples = [Sample(tmp_path / "a", 0, "patient-1", "train"),
               Sample(tmp_path / "b", 1, "patient-1", "test")]
    with pytest.raises(ValueError, match="cross split"):
        validate_split_groups(samples)
