"""Tests for transform utilities."""

import pytest

from sophon.transforms import _DIRECTION_INVERSES, invert_direction, make_identity_transform


def test_invert_direction_roundtrip():
    """Every direction should round-trip: invert(invert(d)) == d."""
    for direction in _DIRECTION_INVERSES:
        assert invert_direction(invert_direction(direction)) == direction


def test_invert_direction_coverage():
    """Inverse map should be symmetric: if A->B then B->A."""
    for d, inv in _DIRECTION_INVERSES.items():
        assert _DIRECTION_INVERSES[inv] == d


def test_invert_direction_known_pairs():
    assert invert_direction("left") == "right"
    assert invert_direction("up") == "down"
    assert invert_direction("clockwise") == "counterclockwise"
    assert invert_direction("inward") == "outward"
    assert invert_direction("rotate_azimuth_cw") == "rotate_azimuth_ccw"


def test_invert_unknown_direction():
    with pytest.raises(ValueError, match="Unknown direction"):
        invert_direction("diagonal")


def test_make_identity_transform():
    identity = make_identity_transform(["translation_x", "rotation_deg", "scale"])
    assert identity == {"translation_x": 0.0, "rotation_deg": 0.0, "scale": 0.0}


def test_make_identity_empty():
    assert make_identity_transform([]) == {}


class TestShortestArcWrap:
    """Rotation ground truth must follow the shortest arc the question promises.

    Categorical rotation questions ask which way a set should move "over the
    shortest arc". Classifying on the raw angle labels a +190 deg rotation as
    the long way round, contradicting the prompt. Only |angle| > 180 is
    affected, so the usual +-45 deg training range is unchanged.
    """

    def test_wrap_angle_deg(self):
        from sophon.transforms import wrap_angle_deg

        assert wrap_angle_deg(30) == 30
        assert wrap_angle_deg(-30) == -30
        assert wrap_angle_deg(190) == pytest.approx(-170)
        assert wrap_angle_deg(350) == pytest.approx(-10)
        assert wrap_angle_deg(-190) == pytest.approx(170)
        assert wrap_angle_deg(370) == pytest.approx(10)
        # (-180, 180]: exactly-180 maps to +180, not -180
        assert wrap_angle_deg(180) == 180
        assert wrap_angle_deg(-180) == 180

    def test_cartesian_direction_follows_shortest_arc(self):
        from sophon.coordinates import get_coordinate_system

        cs = get_coordinate_system("cartesian_2d")
        # +190 deg is a 170 deg turn the other way, so it must classify opposite
        # to +170 deg.
        near = cs.classify_direction({"rotation_deg": 170.0})
        past = cs.classify_direction({"rotation_deg": 190.0})
        assert near != past, "190 deg must classify opposite to 170 deg"
        assert cs.classify_direction({"rotation_deg": 350.0}) == cs.classify_direction(
            {"rotation_deg": -10.0}
        )

    def test_polar_direction_follows_shortest_arc(self):
        from sophon.coordinates import get_coordinate_system

        cs = get_coordinate_system("polar")
        near = cs.classify_direction({"angular_rotation_deg": 170.0})
        past = cs.classify_direction({"angular_rotation_deg": 190.0})
        assert near != past
        assert cs.classify_direction({"angular_rotation_deg": 350.0}) == (
            cs.classify_direction({"angular_rotation_deg": -10.0})
        )

    def test_magnitude_uses_shortest_arc(self):
        from sophon.coordinates import get_coordinate_system

        cs = get_coordinate_system("cartesian_2d")
        # A 350 deg rotation is a 10 deg turn; magnitude must not report 1.94.
        assert cs.transform_magnitude({"rotation_deg": 350.0}) == pytest.approx(
            cs.transform_magnitude({"rotation_deg": -10.0})
        )
        assert cs.transform_magnitude({"rotation_deg": 350.0}) < 0.1
