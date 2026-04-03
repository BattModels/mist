from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from channel_schema import load_dataset_spec, validate_dataset_spec_units


def test_load_dataset_spec_validates_existing_units() -> None:
    spec = load_dataset_spec("conductivity")
    assert spec["channels"][1]["unit"] == "kelvin"


def test_validate_dataset_spec_units_rejects_invalid_unit() -> None:
    spec = {"channels": [{"name": "target", "unit": "definitely_not_a_unit"}]}

    with pytest.raises(ValueError, match="invalid unit"):
        validate_dataset_spec_units(spec, dataset_name="test")


def test_validate_dataset_spec_units_accepts_logit() -> None:
    spec = {"channels": [{"name": "target", "unit": "logit"}]}

    validate_dataset_spec_units(spec, dataset_name="test")
