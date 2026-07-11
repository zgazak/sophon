"""Tests for coordinate_system: all support."""

import yaml

from sophon.generator import BenchmarkGenerator
from sophon.schema import BenchmarkSpec


class TestAllCoordinateSystem:
    def test_spec_accepts_all(self):
        spec_yaml = """\
name: all_test
coordinate_system: all
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
  translation_y: [-0.15, 0.15]
  rotation_deg: [-30, 30]
  radial_shift: [-0.2, 0.2]
  angular_rotation_deg: [-30, 30]
  azimuth_rotation_deg: [-30, 30]
  altitude_offset_deg: [-15, 15]
questions:
  - type: binary
num_samples: 8
seed: 42
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        assert spec.coordinate_system == "all"

    def test_balanced_sampling(self):
        """All 4 coordinate systems should get approximately equal samples."""
        spec_yaml = """\
name: all_balance
coordinate_system: all
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
  translation_y: [-0.15, 0.15]
  rotation_deg: [-30, 30]
  radial_shift: [-0.2, 0.2]
  angular_rotation_deg: [-30, 30]
  azimuth_rotation_deg: [-30, 30]
  altitude_offset_deg: [-15, 15]
questions:
  - type: binary
num_samples: 8
seed: 42
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        results = list(gen.generate())

        cs_counts = {}
        for _, sample in results:
            cs = sample.coordinate_system
            cs_counts[cs] = cs_counts.get(cs, 0) + 1

        # 8 samples / 3 systems = 2 each + 2 remainder to the first systems
        assert cs_counts == {"linear_1d": 3, "cartesian_2d": 3, "polar": 2}

    def test_correct_transforms_per_system(self):
        """Each coordinate system should only get its applicable transforms."""
        spec_yaml = """\
name: all_transforms
coordinate_system: all
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
  rotation_deg: [-30, 30]
  radial_shift: [-0.2, 0.2]
  angular_rotation_deg: [-30, 30]
  azimuth_rotation_deg: [-30, 30]
  altitude_offset_deg: [-15, 15]
questions:
  - type: binary
num_samples: 4
seed: 42
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        for _, sample in gen.generate():
            cs = sample.coordinate_system
            params = sample.transform_params
            if cs == "linear_1d":
                assert set(params.keys()) == {"translation_x"}
            elif cs == "cartesian_2d":
                # Only translation_x and rotation_deg are in the spec and supported
                assert "translation_x" in params
                assert "rotation_deg" in params
            elif cs == "polar":
                assert "radial_shift" in params
                assert "angular_rotation_deg" in params

    def test_deterministic_ordering(self):
        """All linear_1d first, then cartesian_2d, etc."""
        spec_yaml = """\
name: all_order
coordinate_system: all
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
  radial_shift: [-0.2, 0.2]
  angular_rotation_deg: [-30, 30]
  azimuth_rotation_deg: [-30, 30]
  altitude_offset_deg: [-15, 15]
questions:
  - type: binary
num_samples: 8
seed: 42
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        cs_order = [sample.coordinate_system for _, sample in gen.generate()]
        # Should be grouped: linear, cartesian, polar
        expected_order = (
            ["linear_1d"] * 3 + ["cartesian_2d"] * 3 + ["polar"] * 2
        )
        assert cs_order == expected_order

    def test_remainder_distribution(self):
        """Remainder samples should go to first coordinate systems."""
        spec_yaml = """\
name: all_remainder
coordinate_system: all
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
  radial_shift: [-0.2, 0.2]
  angular_rotation_deg: [-30, 30]
  azimuth_rotation_deg: [-30, 30]
  altitude_offset_deg: [-15, 15]
questions:
  - type: binary
num_samples: 5
seed: 42
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        cs_counts = {}
        for _, sample in gen.generate():
            cs = sample.coordinate_system
            cs_counts[cs] = cs_counts.get(cs, 0) + 1

        # 5 / 3 = 1 each + 2 remainder -> first two systems get 2
        assert cs_counts["linear_1d"] == 2
        assert cs_counts["cartesian_2d"] == 2
        assert cs_counts["polar"] == 1
