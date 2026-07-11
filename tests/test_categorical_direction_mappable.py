"""Regression tests for the (cartesian_2d, scale, categorical) label bug.

Pre-2026-07, categorical questions could be paired with transforms that have
no direction reading in the coordinate system's vocabulary (cartesian
"scale"); ``classify_direction`` then fabricated a constant fallback answer
("right", inverted to "left"), producing ill-posed items whose label is
independent of the image. See debug exhibit in the sophon-sae-projector repo
(`debug/cart2d_scale_left_bug/`).
"""

import collections

import pytest
import yaml

from sophon.coordinates import get_coordinate_system
from sophon.generator import BenchmarkGenerator
from sophon.questions import direction_mappable_transforms
from sophon.schema import BenchmarkSpec

CART2D_SCALE_SPEC = """\
name: test_cart2d_scale_regression
description: "cart2d with scale enabled; categorical must never draw scale"
coordinate_system: cartesian_2d
points:
  num_points: [10, 15]
  distribution: uniform
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
  translation_y: [-0.15, 0.15]
  rotation_deg: [-45, 45]
  scale: [-0.3, 0.3]
questions:
  - type: categorical
    vocab_mode: full
    vocab_order: random
num_samples: 150
seed: 7
"""

POLAR_SPEC = """\
name: test_polar_radial_scale_ok
description: "polar radial_scale legitimately maps to inward/outward"
coordinate_system: polar
points:
  num_points: [10, 15]
  distribution: uniform
  margin: 0.1
transforms:
  radial_scale: [-0.3, 0.3]
questions:
  - type: categorical
num_samples: 20
seed: 7
"""


def _generate(spec_yaml):
    spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
    gen = BenchmarkGenerator(spec)
    return [sample for _img, sample in gen.generate()]


class TestDirectionMappableTransforms:
    def test_cartesian_excludes_scale(self):
        mappable = direction_mappable_transforms("cartesian_2d")
        assert set(mappable) == {"translation_x", "translation_y", "rotation_deg"}

    def test_polar_includes_radial_scale(self):
        assert "radial_scale" in direction_mappable_transforms("polar")

    def test_linear(self):
        assert direction_mappable_transforms("linear_1d") == ["translation_x"]


class TestNoIllPosedCategorical:
    def test_scale_never_drawn_for_categorical(self):
        samples = _generate(CART2D_SCALE_SPEC)
        assert len(samples) == 150  # allow-list filtering, not skip-and-drop
        for s in samples:
            assert abs(s.transform_params.get("scale", 0.0)) <= 1e-9, (
                f"{s.sample_id}: categorical item drawn with scale transform "
                f"{s.transform_params}"
            )

    def test_no_degenerate_label_mass(self):
        samples = _generate(CART2D_SCALE_SPEC)
        counts = collections.Counter(str(s.ground_truth).lower() for s in samples)
        n = sum(counts.values())
        top_label, top_n = counts.most_common(1)[0]
        # 3 mappable transforms x 2 signs -> 6 roughly equal classes; the old
        # bug put ~25% constant mass on "left". 40% is a loose degeneracy bound.
        assert top_n / n < 0.40, f"degenerate label mass: {top_label}={top_n}/{n}"
        assert counts["left"] > 0 and counts["right"] > 0

    def test_polar_radial_scale_still_categorical(self):
        samples = _generate(POLAR_SPEC)
        assert len(samples) == 20
        answers = {str(s.ground_truth).lower() for s in samples}
        assert answers <= {"inward", "outward"} and len(answers) == 2


class TestClassifyDirectionRaises:
    def test_cartesian_pure_scale_raises(self):
        cs = get_coordinate_system("cartesian_2d")
        with pytest.raises(ValueError):
            cs.classify_direction(
                {"translation_x": 0.0, "translation_y": 0.0,
                 "rotation_deg": 0.0, "scale": 0.25}
            )
