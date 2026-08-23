"""Tests for the BenchmarkGenerator."""

import yaml
from PIL import Image

from sophon.generator import BenchmarkGenerator
from sophon.schema import BenchmarkSpec

SIMPLE_SPEC_YAML = """\
name: test_cartesian
description: "Test spec"
coordinate_system: cartesian_2d
points:
  num_points: [10, 15]
  distribution: uniform
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
  rotation_deg: [-20, 20]
questions:
  - type: binary
    alignment_threshold: 0.05
  - type: categorical
  - type: quantitative
num_samples: 5
seed: 42
"""


def _make_spec(yaml_str=SIMPLE_SPEC_YAML):
    return BenchmarkSpec(**yaml.safe_load(yaml_str))


class TestBenchmarkGenerator:
    def test_output_count(self):
        spec = _make_spec()
        gen = BenchmarkGenerator(spec)
        results = list(gen.generate())
        # 5 samples × 3 question types = 15
        assert len(results) == 15

    def test_output_types(self):
        spec = _make_spec()
        gen = BenchmarkGenerator(spec)
        for img, sample in gen.generate():
            assert isinstance(img, Image.Image)
            assert sample.spec_name == "test_cartesian"
            assert sample.coordinate_system == "cartesian_2d"

    def test_question_types_present(self):
        spec = _make_spec()
        gen = BenchmarkGenerator(spec)
        q_types = {s.question_type for _, s in gen.generate()}
        assert q_types == {"binary", "categorical", "quantitative"}

    def test_determinism(self):
        spec = _make_spec()
        gen1 = BenchmarkGenerator(spec)
        gen2 = BenchmarkGenerator(spec)
        for (_, s1), (_, s2) in zip(gen1.generate(), gen2.generate()):
            assert s1.sample_id == s2.sample_id
            assert s1.ground_truth == s2.ground_truth
            assert s1.transform_params == s2.transform_params

    def test_seed_override(self):
        spec = _make_spec()
        gen = BenchmarkGenerator(spec)
        results_42 = [(s.ground_truth, s.transform_params) for _, s in gen.generate(seed=42)]
        results_99 = [(s.ground_truth, s.transform_params) for _, s in gen.generate(seed=99)]
        # Different seeds should produce different results
        assert results_42 != results_99

    def test_binary_balance(self):
        """Binary questions should be roughly 50/50 over enough samples."""
        yaml_str = """\
name: binary_balance_test
coordinate_system: cartesian_2d
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.3, 0.3]
questions:
  - type: binary
    alignment_threshold: 0.05
num_samples: 200
seed: 42
"""
        spec = _make_spec(yaml_str)
        gen = BenchmarkGenerator(spec)
        answers = [s.ground_truth for _, s in gen.generate()]
        yes_count = answers.count("yes")
        no_count = answers.count("no")
        ratio = yes_count / max(no_count, 1)
        assert 0.6 < ratio < 1.7, f"Binary balance off: {yes_count} yes, {no_count} no"

    def test_sample_ids_sequential_and_unique(self):
        spec = _make_spec()
        gen = BenchmarkGenerator(spec)
        ids = [s.sample_id for _, s in gen.generate()]
        # IDs should be globally unique strings with sequential indices
        assert len(ids) == len(set(ids)), "sample IDs must be unique"
        # Check format: {spec_name}_seed{seed}_{zero_padded_index}
        for i, sid in enumerate(ids):
            assert isinstance(sid, str), f"sample_id should be str, got {type(sid)}"
            assert sid.endswith(f"_{i:04d}"), f"Expected index {i:04d} in {sid}"
            assert spec.name in sid, f"Expected spec name in {sid}"


class TestComparative:
    def test_comparative_output(self):
        yaml_str = """\
name: comp_test
coordinate_system: cartesian_2d
points:
  num_points: [10, 15]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
questions:
  - type: comparative
num_samples: 5
seed: 42
"""
        spec = _make_spec(yaml_str)
        gen = BenchmarkGenerator(spec)
        for img, sample in gen.generate():
            assert sample.question_type == "comparative"
            assert sample.ground_truth in ("A", "B")
            # Comparative images should be wider than tall
            assert img.size[0] > img.size[1]




class TestDegenerateSingleteonTransforms:
    """Centroid-relative transforms are the identity at n=1.

    Cartesian rotation and scale both operate about the centroid, so with a
    single point nothing moves however large the parameter. Such a sample has
    two identical point sets and a ground truth asserting a direction, so a
    model that correctly says "these are the same" is scored wrong. The
    generator must not emit them.
    """

    @staticmethod
    def _spec(transform, n, qtype="categorical"):
        import yaml

        from sophon.schema import BenchmarkSpec

        return BenchmarkSpec(**yaml.safe_load(f"""
name: degen
coordinate_system: cartesian_2d
output_mode: text_description
points:
  num_points: [{n}, {n}]
  distribution: uniform
  margin: 0.1
transforms:
  {transform}
questions:
  - {{type: {qtype}, vocab_mode: restricted}}
num_samples: 30
seed: 3
"""))

    def test_cartesian_rotation_at_n1_yields_nothing(self):
        from sophon.generator import BenchmarkGenerator

        spec = self._spec("rotation_deg: [20, 45]", 1)
        assert list(BenchmarkGenerator(spec).generate()) == []

    def test_cartesian_scale_at_n1_yields_nothing(self):
        from sophon.generator import BenchmarkGenerator

        spec = self._spec("scale: [0.15, 0.3]", 1, qtype="quantitative")
        assert list(BenchmarkGenerator(spec).generate()) == []

    def test_cartesian_translation_at_n1_is_fine(self):
        from sophon.generator import BenchmarkGenerator

        spec = self._spec("translation_x: [0.1, 0.15]", 1)
        assert len(list(BenchmarkGenerator(spec).generate())) > 0

    def test_cartesian_rotation_at_n2_is_fine(self):
        from sophon.generator import BenchmarkGenerator

        spec = self._spec("rotation_deg: [20, 45]", 2)
        assert len(list(BenchmarkGenerator(spec).generate())) > 0

    def test_polar_transforms_unaffected_at_n1(self):
        import numpy as np

        from sophon.coordinates import get_coordinate_system

        cs = get_coordinate_system("polar")
        pts = np.array([[0.6, 1.0]])
        for params in ({"radial_scale": 0.3}, {"angular_rotation_deg": 35.0},
                       {"radial_shift": 0.2}):
            assert not np.allclose(cs.apply_transform(pts, params), pts)
