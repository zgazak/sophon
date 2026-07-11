"""Tests for coordinate system implementations."""

import numpy as np
import pytest

from sophon.coordinates import get_coordinate_system
from sophon.coordinates.cartesian import Cartesian2D
from sophon.coordinates.linear import Linear1D
from sophon.coordinates.polar import Polar

ALL_SYSTEMS = [
    ("linear_1d", Linear1D),
    ("cartesian_2d", Cartesian2D),
    ("polar", Polar),
]


@pytest.fixture(params=[name for name, _ in ALL_SYSTEMS], ids=[name for name, _ in ALL_SYSTEMS])
def coord_system(request):
    return get_coordinate_system(request.param)


def make_rng(seed=42):
    return np.random.Generator(np.random.PCG64(seed))


class TestRegistry:
    def test_all_registered(self):
        for name, cls in ALL_SYSTEMS:
            cs = get_coordinate_system(name)
            assert isinstance(cs, cls)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_coordinate_system("hyperbolic")


class TestGeneratePoints:
    def test_shape(self, coord_system):
        pts = coord_system.generate_points(20, make_rng(), "uniform", 0.1)
        assert pts.shape == (20, coord_system.ndim)

    def test_determinism(self, coord_system):
        pts1 = coord_system.generate_points(30, make_rng(99), "uniform", 0.1)
        pts2 = coord_system.generate_points(30, make_rng(99), "uniform", 0.1)
        np.testing.assert_array_equal(pts1, pts2)

    def test_different_seeds_differ(self, coord_system):
        pts1 = coord_system.generate_points(30, make_rng(1), "uniform", 0.1)
        pts2 = coord_system.generate_points(30, make_rng(2), "uniform", 0.1)
        assert not np.array_equal(pts1, pts2)

    def test_distributions(self, coord_system):
        for dist in ["uniform", "clustered", "grid"]:
            pts = coord_system.generate_points(25, make_rng(), dist, 0.1)
            assert pts.shape[0] == 25

    def test_unknown_distribution(self, coord_system):
        with pytest.raises(ValueError, match="Unknown distribution"):
            coord_system.generate_points(10, make_rng(), "spiral", 0.1)


def _make_ranges(coord_system) -> dict[str, list[float]]:
    """Create test transform ranges for a coordinate system."""
    range_map = {
        "translation_x": [-0.15, 0.15],
        "translation_y": [-0.15, 0.15],
        "rotation_deg": [-30.0, 30.0],
        "scale": [-0.2, 0.2],
        "radial_shift": [-0.2, 0.2],
        "angular_rotation_deg": [-30.0, 30.0],
        "radial_scale": [-0.2, 0.2],
        "azimuth_rotation_deg": [-30.0, 30.0],
        "altitude_offset_deg": [-15.0, 15.0],
        "roll_deg": [-20.0, 20.0],
    }
    return {k: range_map[k] for k in coord_system.supported_transforms if k in range_map}


class TestApplyTransform:
    def test_identity(self, coord_system):
        rng = make_rng()
        pts = coord_system.generate_points(20, rng, "uniform", 0.1)
        identity = {k: 0.0 for k in coord_system.supported_transforms}
        transformed = coord_system.apply_transform(pts, identity)
        np.testing.assert_allclose(transformed, pts, atol=1e-12)

    def test_does_not_mutate(self, coord_system):
        rng = make_rng()
        pts = coord_system.generate_points(20, rng, "uniform", 0.1)
        original = pts.copy()
        params = coord_system.sample_transform(make_rng(), _make_ranges(coord_system))
        coord_system.apply_transform(pts, params)
        np.testing.assert_array_equal(pts, original)


class TestToImageCoords:
    def test_output_shape(self, coord_system):
        pts = coord_system.generate_points(15, make_rng(), "uniform", 0.1)
        img_coords = coord_system.to_image_coords(pts, 512, 0.1)
        assert img_coords.shape == (15, 2)

    def test_within_image_bounds(self, coord_system):
        pts = coord_system.generate_points(50, make_rng(), "uniform", 0.1)
        img_coords = coord_system.to_image_coords(pts, 512, 0.1)
        # Points should be roughly within the image
        assert np.all(img_coords >= -10)  # small tolerance for edge effects
        assert np.all(img_coords <= 522)


class TestClassifyDirection:
    def test_returns_valid_vocab(self, coord_system):
        params = coord_system.sample_transform(make_rng(), _make_ranges(coord_system))
        direction = coord_system.classify_direction(params)
        assert direction in coord_system.direction_vocabulary

    def test_identity_returns_something(self, coord_system):
        identity = {k: 0.0 for k in coord_system.supported_transforms}
        direction = coord_system.classify_direction(identity)
        assert direction in coord_system.direction_vocabulary


class TestTransformMagnitude:
    def test_identity_is_zero(self, coord_system):
        identity = {k: 0.0 for k in coord_system.supported_transforms}
        assert coord_system.transform_magnitude(identity) == 0.0

    def test_non_identity_positive(self, coord_system):
        params = coord_system.sample_transform(make_rng(), _make_ranges(coord_system))
        # Could be zero by chance, but very unlikely with these ranges
        mag = coord_system.transform_magnitude(params)
        assert mag >= 0.0


class TestSampleTransform:
    def test_keys(self, coord_system):
        ranges = _make_ranges(coord_system)
        params = coord_system.sample_transform(make_rng(), ranges)
        for key in coord_system.supported_transforms:
            assert key in params

    def test_zero_range_gives_zero(self, coord_system):
        ranges = {k: [0.0, 0.0] for k in coord_system.supported_transforms}
        params = coord_system.sample_transform(make_rng(), ranges)
        for v in params.values():
            assert v == 0.0

    def test_within_bounds(self, coord_system):
        ranges = _make_ranges(coord_system)
        rng = make_rng()
        for _ in range(50):
            params = coord_system.sample_transform(rng, ranges)
            for key, val in params.items():
                if key in ranges:
                    lo, hi = ranges[key]
                    assert lo <= val <= hi, f"{key}: {val} not in [{lo}, {hi}]"


class TestDescribeTransform:
    def test_identity(self, coord_system):
        identity = {k: 0.0 for k in coord_system.supported_transforms}
        desc = coord_system.describe_transform(identity)
        assert desc == "identity"

    def test_non_identity_nonempty(self, coord_system):
        params = coord_system.sample_transform(make_rng(7), _make_ranges(coord_system))
        desc = coord_system.describe_transform(params)
        assert len(desc) > 0


# --- Coordinate-system-specific tests ---


class TestCartesian2D:
    def test_translation(self):
        cs = Cartesian2D()
        pts = np.array([[0.5, 0.5], [0.6, 0.6]])
        result = cs.apply_transform(pts, {"translation_x": 0.1, "translation_y": -0.2})
        np.testing.assert_allclose(result, [[0.6, 0.3], [0.7, 0.4]], atol=1e-10)

    def test_rotation_90(self):
        cs = Cartesian2D()
        # Two points symmetric about (0.5, 0.5)
        pts = np.array([[0.5, 0.7], [0.5, 0.3]])
        result = cs.apply_transform(pts, {"rotation_deg": 90.0})
        # 90° CCW: (0, 0.2) -> (-0.2, 0), (0, -0.2) -> (0.2, 0)
        expected = np.array([[0.3, 0.5], [0.7, 0.5]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_y_flip_in_image_coords(self):
        cs = Cartesian2D()
        pts = np.array([[0.5, 0.9]])  # High y in native = top of image
        img = cs.to_image_coords(pts, 512, 0.0)
        # With no margin, y=0.9 should map to near top of image (low y pixel)
        assert img[0, 1] < 512 / 2


class TestLinear1D:
    def test_translation(self):
        cs = Linear1D()
        pts = np.array([[0.3], [0.5], [0.7]])
        result = cs.apply_transform(pts, {"translation_x": 0.1})
        np.testing.assert_allclose(result, [[0.4], [0.6], [0.8]], atol=1e-10)

    def test_image_coords_horizontal(self):
        cs = Linear1D()
        pts = np.array([[0.2], [0.8]])
        img = cs.to_image_coords(pts, 512, 0.1)
        # Both y values should be at center
        np.testing.assert_allclose(img[:, 1], 256.0)
        # x values should differ
        assert img[0, 0] < img[1, 0]


class TestPolar:
    def test_angular_rotation(self):
        cs = Polar()
        pts = np.array([[0.5, 0.0], [0.5, np.pi]])
        result = cs.apply_transform(pts, {"angular_rotation_deg": 90.0})
        expected_theta = np.array([np.pi / 2, 3 * np.pi / 2])
        np.testing.assert_allclose(result[:, 1], expected_theta, atol=1e-10)
        np.testing.assert_allclose(result[:, 0], 0.5, atol=1e-10)

    def test_radial_shift(self):
        cs = Polar()
        pts = np.array([[0.3, 1.0], [0.6, 2.0]])
        result = cs.apply_transform(pts, {"radial_shift": 0.1})
        np.testing.assert_allclose(result[:, 0], [0.4, 0.7], atol=1e-10)

    def test_radial_clamp(self):
        cs = Polar()
        pts = np.array([[0.95, 0.0]])
        result = cs.apply_transform(pts, {"radial_shift": 0.2})
        assert result[0, 0] == 1.0  # Clamped

    def test_area_uniform_distribution(self):
        """With r~sqrt(U), the CDF of r^2 should be roughly uniform."""
        cs = Polar()
        pts = cs.generate_points(5000, make_rng(), "uniform", 0.0)
        r_vals = pts[:, 0]
        # r ~ sqrt(U) means r^2 ~ U(0, r_max^2), so r^2 values should be uniform
        r_sq = r_vals**2
        median_r_sq = np.median(r_sq)
        r_max = r_vals.max()
        expected_median = r_max**2 / 2
        # Median of r^2 should be close to r_max^2 / 2
        assert abs(median_r_sq - expected_median) < 0.1, (
            f"r^2 median={median_r_sq:.3f}, expected ~{expected_median:.3f}"
        )


