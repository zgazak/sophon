"""Tests for Pydantic v2 schema models."""

import pytest
import yaml
from pydantic import ValidationError

from sophon.schema import BenchmarkSpec, QuestionConfig, Sample, SuiteSpec

VALID_SPEC_YAML = """\
name: cartesian_rotation
description: "2D Cartesian rotation detection"
coordinate_system: cartesian_2d
points:
  num_points: [30, 60]
  distribution: uniform
  margin: 0.1
transforms:
  rotation_deg: [-30, 30]
questions:
  - type: binary
    alignment_threshold: 0.02
  - type: categorical
  - type: quantitative
num_samples: 100
seed: 42
"""


def test_valid_spec_from_yaml():
    data = yaml.safe_load(VALID_SPEC_YAML)
    spec = BenchmarkSpec(**data)
    assert spec.name == "cartesian_rotation"
    assert spec.coordinate_system == "cartesian_2d"
    assert spec.points.num_points == (30, 60)
    assert spec.transforms["rotation_deg"] == [-30, 30]
    assert len(spec.questions) == 3
    assert spec.num_samples == 100
    assert spec.seed == 42


def test_spec_defaults():
    data = yaml.safe_load(VALID_SPEC_YAML)
    spec = BenchmarkSpec(**data)
    assert spec.rendering.image_size == 512
    assert spec.rendering.color_a == "#2563EB"
    assert spec.rendering.show_grid is True


def test_spec_missing_required_field():
    data = yaml.safe_load(VALID_SPEC_YAML)
    del data["coordinate_system"]
    with pytest.raises(ValidationError):
        BenchmarkSpec(**data)


def test_spec_invalid_coordinate_system():
    data = yaml.safe_load(VALID_SPEC_YAML)
    data["coordinate_system"] = "hyperbolic"
    with pytest.raises(ValidationError):
        BenchmarkSpec(**data)


def test_spec_invalid_question_type():
    data = yaml.safe_load(VALID_SPEC_YAML)
    data["questions"] = [{"type": "essay"}]
    with pytest.raises(ValidationError):
        BenchmarkSpec(**data)


def test_spec_extra_field_rejected():
    data = yaml.safe_load(VALID_SPEC_YAML)
    data["bogus_field"] = 123
    with pytest.raises(ValidationError):
        BenchmarkSpec(**data)


def test_sample_roundtrip():
    s = Sample(
        sample_id=0,
        spec_name="test",
        coordinate_system="cartesian_2d",
        question_type="binary",
        question_text="Are they aligned?",
        ground_truth="yes",
        transform_params={"rotation_deg": 5.0},
        transform_magnitude=5.0,
    )
    json_str = s.model_dump_json()
    restored = Sample.model_validate_json(json_str)
    assert restored == s


def test_sample_numeric_ground_truth():
    s = Sample(
        sample_id=1,
        spec_name="test",
        coordinate_system="cartesian_2d",
        question_type="quantitative",
        question_text="Estimate the angle",
        ground_truth=15.3,
        transform_params={"rotation_deg": 15.3},
        transform_magnitude=15.3,
    )
    assert s.ground_truth == 15.3


def test_suite_spec():
    data = {
        "name": "full_suite",
        "version": "0.1",
        "description": "Complete suite",
        "specs": ["cartesian_rotation.yaml", "polar_rotation.yaml"],
        "seed": 42,
    }
    suite = SuiteSpec(**data)
    assert len(suite.specs) == 2
    assert suite.version == "0.1"


def test_question_config_alignment_threshold():
    q = QuestionConfig(type="binary", alignment_threshold=0.05)
    assert q.alignment_threshold == 0.05

    q2 = QuestionConfig(type="categorical")
    assert q2.alignment_threshold is None
