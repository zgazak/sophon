"""Tests for the opt-in ``balance: exact`` spec mode (stratified label balance).

Default ("random") preserves historical uniform-draw behavior; "exact"
schedules strata per question type so realized label counts are balanced
(differing by ≤1 when n is not divisible by the stratum count).
"""

import collections

import yaml

from sophon.generator import BenchmarkGenerator
from sophon.schema import BenchmarkSpec

BASE = """\
name: balance_test
description: "balance: exact tests"
coordinate_system: cartesian_2d
output_mode: text_description
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
  - {{type: {qtype}{extra}}}
num_samples: {n}
seed: 11
balance: exact
"""


def _samples(qtype, n, extra=""):
    spec = BenchmarkSpec(**yaml.safe_load(BASE.format(qtype=qtype, n=n, extra=extra)))
    return [s for _img, s in BenchmarkGenerator(spec).generate()]


def test_default_is_random():
    spec_yaml = BASE.format(qtype="binary", n=4, extra="").replace("balance: exact\n", "")
    spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
    assert spec.balance == "random"


def test_categorical_exactly_balanced():
    # 3 mappable transforms x 2 signs = 6 labels; 60/6 = 10 each, exactly.
    samples = _samples("categorical", 60, extra=", vocab_mode: full, vocab_order: random")
    counts = collections.Counter(str(s.ground_truth).lower() for s in samples)
    assert len(samples) == 60
    assert sorted(counts.values()) == [10] * 6, counts
    for s in samples:
        assert abs(s.transform_params.get("scale", 0.0)) <= 1e-9


def test_binary_exactly_balanced():
    samples = _samples("binary", 40, extra=", alignment_threshold: 0.05")
    counts = collections.Counter(str(s.ground_truth).lower() for s in samples)
    assert counts == {"yes": 20, "no": 20}, counts


def test_comparative_exactly_balanced():
    samples = _samples("comparative", 40)
    counts = collections.Counter(str(s.ground_truth) for s in samples)
    assert counts == {"A": 20, "B": 20}, counts


def test_quantitative_even_transforms_and_signs():
    # rotation + scale active -> 20 each (vs ~99% rotation under random,
    # where the raw-unit max always favors degrees over scale fractions);
    # scale sign balance -> 10 factors > 1 and 10 < 1.
    samples = _samples("quantitative", 40)
    assert len(samples) == 40
    scale_items = [s for s in samples if abs(s.transform_params.get("scale", 0.0)) > 1e-9]
    rot_items = [s for s in samples if abs(s.transform_params.get("rotation_deg", 0.0)) > 1e-9]
    assert len(scale_items) == 20 and len(rot_items) == 20
    grow = sum(1 for s in scale_items if float(s.ground_truth) > 1.0)
    assert grow == 10, f"grow={grow}"


def test_all_coords_exact_balance():
    spec = BenchmarkSpec(**yaml.safe_load("""
name: balance_all
description: "exact balance across coordinate systems"
coordinate_system: all
output_mode: text_description
points: {num_points: [10, 15], distribution: uniform, margin: 0.1}
transforms:
  translation_x: [-0.15, 0.15]
  translation_y: [-0.15, 0.15]
  rotation_deg: [-45, 45]
  scale: [-0.3, 0.3]
  radial_shift: [-0.2, 0.2]
  angular_rotation_deg: [-45, 45]
  radial_scale: [-0.3, 0.3]
questions:
  - {type: categorical, vocab_mode: full, vocab_order: random}
num_samples: 36
seed: 11
balance: exact
"""))
    per_cs = collections.defaultdict(collections.Counter)
    for _img, s in BenchmarkGenerator(spec).generate():
        per_cs[s.coordinate_system][str(s.ground_truth).lower()] += 1
    # 36/3 = 12 per system; cart2d 6 strata -> 2 each; linear 2 -> 6 each;
    # polar 6 strata (3 transforms x 2 signs) -> 2 each label-side... polar
    # has 4 labels (in/out from 2 transforms, cw/ccw from 1): in=out=4, cw=ccw=2.
    assert sorted(per_cs["cartesian_2d"].values()) == [2] * 6
    assert sorted(per_cs["linear_1d"].values()) == [6, 6]
    assert per_cs["polar"]["clockwise"] == 2 and per_cs["polar"]["counterclockwise"] == 2
    assert per_cs["polar"]["inward"] == 4 and per_cs["polar"]["outward"] == 4


SINGLE = """\
name: balance_single
description: "signed stratum degeneracy regression"
coordinate_system: {coord}
output_mode: text_description
points:
  num_points: [10, 15]
  distribution: uniform
  margin: 0.1
transforms:
  {key}: {rng}
questions:
  - {{type: categorical, vocab_mode: restricted}}
num_samples: {n}
seed: 99
balance: exact
"""


def test_signed_sampler_never_draws_the_degenerate_band():
    """A signed draw must never land where `_generate_one` would reject it.

    Under ``balance: exact`` such a draw consumes its scheduled stratum and
    yields nothing, so the realized labels come out unbalanced — the one thing
    the mode promises not to do.
    """
    import numpy as np

    from sophon.coordinates import get_coordinate_system
    from sophon.generator import _DEGENERATE_EPS, _sample_signed_single_transform

    cs = get_coordinate_system("cartesian_2d")
    rng = np.random.Generator(np.random.PCG64(0))
    # A range tight around zero, so essentially every draw would otherwise fall
    # in the degenerate band.
    ranges = {"translation_x": [-2e-6, 2e-6]}
    for sign in (+1, -1):
        for _ in range(500):
            v = _sample_signed_single_transform(rng, cs, ranges, "translation_x", sign)
            got = v["translation_x"]
            assert abs(got) >= _DEGENERATE_EPS, f"drew into the degenerate band: {got}"
            assert got * sign > 0, f"sign {sign} stratum produced {got}"


def test_signed_sampler_survives_range_narrower_than_the_band():
    """Pathological sub-epsilon range must not invert the uniform() bounds."""
    import numpy as np

    from sophon.coordinates import get_coordinate_system
    from sophon.generator import _sample_signed_single_transform

    cs = get_coordinate_system("cartesian_2d")
    rng = np.random.Generator(np.random.PCG64(0))
    ranges = {"translation_x": [-1e-9, 1e-9]}
    for sign in (+1, -1):
        drawn = _sample_signed_single_transform(rng, cs, ranges, "translation_x", sign)
        got = drawn["translation_x"]
        assert -1e-9 <= got <= 1e-9, f"escaped the declared range: {got}"


def test_large_single_transform_runs_stay_exactly_balanced():
    """End-to-end invariant at the scale where the degeneracy bug surfaced.

    Whether any particular (n, seed, num_points) actually draws into the
    degenerate band depends on the exact RNG stream, so this is a smoke check
    over a large run rather than a deterministic reproduction — the guard that
    genuinely fails without the fix is
    `test_signed_sampler_never_draws_the_degenerate_band` above.
    """
    cases = [
        ("cartesian_2d", "translation_x", "[-0.15, 0.15]"),
        ("cartesian_2d", "translation_y", "[-0.15, 0.15]"),
        ("polar", "radial_scale", "[-0.3, 0.3]"),
    ]
    for coord, key, rng_ in cases:
        spec = BenchmarkSpec(
            **yaml.safe_load(SINGLE.format(coord=coord, key=key, rng=rng_, n=10400))
        )
        samples = [s for _img, s in BenchmarkGenerator(spec).generate()]
        counts = collections.Counter(s.ground_truth for s in samples)
        assert len(samples) == 10400, f"{key}: yielded {len(samples)} of 10400"
        assert set(counts.values()) == {5200}, f"{key}: unbalanced {dict(counts)}"
