"""Tests for text description output mode."""

import numpy as np
import yaml

from sophon.coordinates import get_coordinate_system
from sophon.generator import BenchmarkGenerator
from sophon.schema import BenchmarkSpec


def make_rng(seed=42):
    return np.random.Generator(np.random.PCG64(seed))


class TestDescribePoints:
    """Test describe_points() on each coordinate system."""

    def test_linear(self):
        cs = get_coordinate_system("linear_1d")
        pts = np.array([[0.3], [0.1], [0.7]])
        desc = cs.describe_points(pts)
        assert "1D number line" in desc
        # Should be sorted by first coordinate
        lines = desc.strip().split("\n")
        assert "0.100" in lines[1]  # First point after header
        assert "0.300" in lines[2]
        assert "0.700" in lines[3]

    def test_cartesian(self):
        # describe_points reports the DISPLAY frame (y' = 1 - y), matching the
        # rendered image and the direction ground-truth words.
        cs = get_coordinate_system("cartesian_2d")
        pts = np.array([[0.5, 0.3], [0.1, 0.9]])
        desc = cs.describe_points(pts)
        assert "2D Cartesian" in desc
        assert "(0.100, 0.100)" in desc
        assert "(0.500, 0.700)" in desc

    def test_polar(self):
        cs = get_coordinate_system("polar")
        pts = np.array([[0.5, np.pi / 2], [0.2, 0.0]])
        desc = cs.describe_points(pts)
        assert "polar" in desc
        assert "r=0.200" in desc
        assert "r=0.500" in desc

    def test_determinism(self):
        cs = get_coordinate_system("cartesian_2d")
        pts = cs.generate_points(20, make_rng(42), "uniform", 0.1)
        desc1 = cs.describe_points(pts)
        desc2 = cs.describe_points(pts)
        assert desc1 == desc2

    def test_sorted_by_first_coord(self):
        cs = get_coordinate_system("cartesian_2d")
        pts = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
        desc = cs.describe_points(pts)
        lines = desc.strip().split("\n")
        # After header line, points should be sorted by x
        assert "0.100" in lines[1]
        assert "0.500" in lines[2]
        assert "0.900" in lines[3]

    def test_three_decimal_places(self):
        cs = get_coordinate_system("cartesian_2d")
        pts = np.array([[0.123456789, 0.987654321]])
        desc = cs.describe_points(pts)
        assert "0.123" in desc
        assert "0.012" in desc  # display frame: 1 - 0.9877 -> 0.012


class TestTextGroundTruthConsistency:
    """Regression tests for the 2026-08 text-mode direction bug: GT words are
    defined in the rendered (display) frame, so the printed coordinates must be
    display-frame too. Before the fix, trans_y and rotation categorical GTs were
    reversed relative to the printed math-frame coordinates (8/8 inverted)."""

    def _pts(self, block):
        import re
        return [(float(m.group(1)), float(m.group(2)))
                for m in re.finditer(r"\(([\d.-]+), ([\d.-]+)\)", block)]

    def _samples(self, transforms, seed):
        from sophon.generator import BenchmarkGenerator
        from sophon.schema import BenchmarkSpec, PointsConfig, QuestionConfig
        spec = BenchmarkSpec(
            name="gt_consistency", coordinate_system="cartesian_2d",
            points=PointsConfig(num_points=(5, 5)), transforms=transforms,
            questions=[QuestionConfig(type="categorical")], num_samples=8,
            seed=seed, output_mode="text_description", balance="exact")
        return list(BenchmarkGenerator(spec).generate(seed=seed))

    def test_trans_y_gt_matches_printed_coordinates(self):
        for _img, s in self._samples({"translation_y": [-0.15, 0.15]}, seed=7):
            blue, red = s.text_description.split("Red points")
            by = np.mean([p[1] for p in self._pts(blue)])
            ry = np.mean([p[1] for p in self._pts(red)])
            assert ("up" if by > ry else "down") == s.ground_truth

    def test_rotation_gt_matches_printed_coordinates(self):
        import math
        for _img, s in self._samples({"rotation_deg": [-45.0, 45.0]}, seed=11):
            blue, red = (self._pts(b) for b in s.text_description.split("Red points"))
            dth = [(math.atan2(by - 0.5, bx - 0.5) - math.atan2(ry - 0.5, rx - 0.5)
                    + math.pi) % (2 * math.pi) - math.pi
                   for (bx, by), (rx, ry) in zip(blue, red)]
            need = "counterclockwise" if sum(dth) / len(dth) > 0 else "clockwise"
            assert need == s.ground_truth

    def test_trans_x_gt_matches_printed_coordinates(self):
        for _img, s in self._samples({"translation_x": [-0.15, 0.15]}, seed=5):
            blue, red = s.text_description.split("Red points")
            bx = np.mean([p[0] for p in self._pts(blue)])
            rx = np.mean([p[0] for p in self._pts(red)])
            assert ("right" if bx > rx else "left") == s.ground_truth


class TestTextDescriptionOutputMode:
    """Test that output_mode='text_description' generates text and no image."""

    def test_text_only(self):
        spec_yaml = """\
name: text_test
coordinate_system: cartesian_2d
points:
  num_points: [5, 10]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
questions:
  - type: binary
num_samples: 3
seed: 42
output_mode: text_description
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        for img, sample in gen.generate():
            assert img is None
            assert sample.text_description is not None
            assert "Blue points" in sample.text_description
            assert "Red points" in sample.text_description

    def test_both_mode(self):
        spec_yaml = """\
name: both_test
coordinate_system: cartesian_2d
points:
  num_points: [5, 10]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
questions:
  - type: binary
num_samples: 2
seed: 42
output_mode: both
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        for img, sample in gen.generate():
            assert img is not None
            assert sample.text_description is not None

    def test_image_mode_no_text(self):
        spec_yaml = """\
name: image_test
coordinate_system: cartesian_2d
points:
  num_points: [5, 10]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
questions:
  - type: binary
num_samples: 2
seed: 42
output_mode: image
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen = BenchmarkGenerator(spec)
        for img, sample in gen.generate():
            assert img is not None
            assert sample.text_description is None

    def test_text_determinism(self):
        spec_yaml = """\
name: text_determ
coordinate_system: cartesian_2d
points:
  num_points: [10, 10]
  margin: 0.1
transforms:
  rotation_deg: [-20, 20]
questions:
  - type: categorical
num_samples: 3
seed: 42
output_mode: text_description
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        gen1 = BenchmarkGenerator(spec)
        gen2 = BenchmarkGenerator(spec)
        for (_, s1), (_, s2) in zip(gen1.generate(), gen2.generate()):
            assert s1.text_description == s2.text_description

