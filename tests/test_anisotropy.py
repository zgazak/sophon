"""PointsConfig.aspect_ratio — controllable screen-aligned h/w.

The point of the knob is that the axis-aligned bounding box becomes an
independent variable. A model reading "did the box get wider or taller" as a
rotation cue has a sign flip wherever bbox width peaks, and that peak moves with
the cloud's aspect ratio; a generic 4-fold screen-locked cue does not move. So
these tests pin down both that the ratio is delivered exactly and that the
default stays a bit-for-bit no-op.
"""

import numpy as np
import pytest

from sophon.coordinates import get_coordinate_system
from sophon.schema import PointsConfig


@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_delivers_requested_aspect(ratio):
    cs = get_coordinate_system("cartesian_2d")
    rng = np.random.default_rng(0)
    measured = []
    for _ in range(200):
        pts = cs.generate_points_shaped(int(rng.integers(30, 61)), rng, "uniform", 0.1, ratio)
        w, h = np.ptp(pts, axis=0)
        measured.append(h / w)
    assert np.mean(measured) == pytest.approx(ratio, rel=0.02)


def test_default_is_a_bitwise_noop():
    """aspect_ratio=1.0 must not perturb the values or the rng stream."""
    cs = get_coordinate_system("cartesian_2d")
    plain = cs.generate_points(50, np.random.default_rng(3), "uniform", 0.1)
    shaped = cs.generate_points_shaped(50, np.random.default_rng(3), "uniform", 0.1, 1.0)
    assert np.array_equal(plain, shaped)


@pytest.mark.parametrize("distribution", ["uniform", "clustered", "grid"])
@pytest.mark.parametrize("ratio", [0.4, 2.5])
def test_margin_is_preserved(distribution, ratio):
    """Squeezing only ever compresses, so a cloud inside the margin stays inside."""
    cs = get_coordinate_system("cartesian_2d")
    rng = np.random.default_rng(1)
    for _ in range(50):
        pts = cs.generate_points_shaped(40, rng, distribution, 0.1, ratio)
        assert pts.min() >= 0.1 - 1e-9
        assert pts.max() <= 0.9 + 1e-9


@pytest.mark.parametrize("system", ["polar", "linear_1d"])
def test_systems_without_a_screen_basis_reject_it(system):
    """Polar natives are (r, theta); linear_1d has one axis. Squeezing either
    would not mean what it says, so it must fail loudly rather than quietly."""
    cs = get_coordinate_system(system)
    rng = np.random.default_rng(0)
    cs.generate_points_shaped(20, rng, "uniform", 0.1, 1.0)  # default still fine
    with pytest.raises(ValueError, match="aspect_ratio"):
        cs.generate_points_shaped(20, rng, "uniform", 0.1, 2.0)


def test_bbox_width_peak_moves_with_aspect():
    """The whole reason the knob exists: the sign-flip angle is a function of
    the aspect ratio, so it can be predicted before the model is ever run."""
    cs = get_coordinate_system("cartesian_2d")
    grid = np.arange(2.0, 90.0, 0.5)

    def peak(ratio):
        rng = np.random.default_rng(0)
        out = []
        for _ in range(120):
            pts = cs.generate_points_shaped(45, rng, "uniform", 0.1, ratio)
            c = pts.mean(0)
            widths = []
            for t in np.radians(grid):
                rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
                widths.append(np.ptp(((pts - c) @ rot)[:, 0]))
            out.append(grid[int(np.argmax(widths))])
        return float(np.mean(out))

    peaks = [peak(r) for r in (1.0, 2.0, 3.0)]
    assert peaks[0] == pytest.approx(45.0, abs=4.0)      # isotropic control
    assert peaks == sorted(peaks)                        # monotone in aspect
    assert peaks[-1] - peaks[0] > 25.0                   # and by a large margin


def test_config_validates():
    assert PointsConfig(num_points=(10, 20)).aspect_ratio == 1.0
    assert PointsConfig(num_points=(10, 20), aspect_ratio=2.0).aspect_ratio == 2.0
    with pytest.raises(ValueError, match="aspect_ratio"):
        PointsConfig(num_points=(10, 20), aspect_ratio=0.0)
    with pytest.raises(ValueError, match="aspect_ratio"):
        PointsConfig(num_points=(10, 20), aspect_ratio=-1.0)
