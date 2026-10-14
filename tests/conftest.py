from pathlib import Path

import pytest

from tinymedvision.synthetic import generate_fixture


@pytest.fixture
def fixture_manifest(tmp_path: Path) -> Path:
    return generate_fixture(tmp_path / "images", n_per_class=6, seed=23)
