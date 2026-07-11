"""Tests for per-sample metadata fields."""

import yaml

from sophon.generator import BenchmarkGenerator
from sophon.schema import BenchmarkSpec


class TestPerSampleMetadata:
    def test_num_points_populated(self):
        spec_yaml = """\
name: meta_test
coordinate_system: cartesian_2d
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
questions:
  - type: binary
num_samples: 5
seed: 42
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        for _, sample in gen.generate():
            assert sample.num_points is not None
            assert 10 <= sample.num_points <= 15

    def test_show_grid_populated(self):
        spec_yaml = """\
name: grid_meta_test
coordinate_system: cartesian_2d
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
questions:
  - type: binary
num_samples: 3
seed: 42
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        for _, sample in gen.generate():
            assert sample.show_grid is True

    def test_transform_direction_for_categorical(self):
        spec_yaml = """\
name: dir_meta_test
coordinate_system: cartesian_2d
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  rotation_deg: [-30, 30]
questions:
  - type: categorical
num_samples: 5
seed: 42
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        for _, sample in gen.generate():
            assert sample.transform_direction is not None
            assert sample.transform_direction in [
                "left",
                "right",
                "up",
                "down",
                "clockwise",
                "counterclockwise",
            ]

    def test_transform_direction_none_for_binary(self):
        spec_yaml = """\
name: dir_binary_test
coordinate_system: cartesian_2d
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
questions:
  - type: binary
num_samples: 5
seed: 42
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        for _, sample in gen.generate():
            assert sample.transform_direction is None

    def test_metadata_in_model_dump(self):
        """Verify new fields appear in model_dump() output."""
        spec_yaml = """\
name: dump_test
coordinate_system: cartesian_2d
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
questions:
  - type: binary
num_samples: 2
seed: 42
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        for _, sample in gen.generate():
            d = sample.model_dump()
            assert "num_points" in d
            assert "show_grid" in d
            assert "transform_params" in d
            assert "transform_magnitude" in d
