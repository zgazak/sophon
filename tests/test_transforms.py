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
