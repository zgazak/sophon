"""Tests for question generation."""

from sophon.coordinates import get_coordinate_system
from sophon.questions import (
    generate_binary_question,
    generate_categorical_question,
    generate_comparative_question,
    generate_coord_identification_question,
    generate_quantitative_question,
)
from sophon.transforms import _DIRECTION_INVERSES


def test_binary_aligned():
    text, answer = generate_binary_question(is_aligned=True)
    assert answer == "yes"
    assert "aligned" in text.lower()
    assert "dot width" in text.lower()


def test_binary_misaligned():
    text, answer = generate_binary_question(is_aligned=False)
    assert answer == "no"


def test_categorical_returns_inverse():
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.2, "translation_y": 0.0, "rotation_deg": 0.0, "scale": 0.0}
    text, answer = generate_categorical_question(cs, params)
    assert answer == "left"  # Inverse of "right"
    assert "direction" in text.lower()


def test_categorical_valid_direction():
    for name in ["linear_1d", "cartesian_2d", "polar", "spherical"]:
        cs = get_coordinate_system(name)
        # Make a non-zero transform
        params = {k: 0.1 for k in cs.supported_transforms}
        _, answer = generate_categorical_question(cs, params)
        assert answer in _DIRECTION_INVERSES


def test_comparative():
    cs = get_coordinate_system("cartesian_2d")
    pa = {"translation_x": 0.1}
    pb = {"translation_x": 0.3}
    text, answer = generate_comparative_question(cs, pa, pb, "A")
    assert answer == "A"
    assert "A or B" in text


def test_quantitative_rotation_positive_magnitude():
    cs = get_coordinate_system("cartesian_2d")
    # Negative rotation — ground truth should be positive magnitude
    params = {"translation_x": 0.0, "translation_y": 0.0, "rotation_deg": -15.3, "scale": 0.0}
    result = generate_quantitative_question(cs, params)
    assert result is not None
    text, answer = result
    assert isinstance(answer, float)
    assert abs(answer - 15.3) < 1e-6  # Positive magnitude
    assert "rotation" in text.lower()
    assert "between" in text.lower()  # Asks for measurement, not correction


def test_quantitative_rotation_already_positive():
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.0, "translation_y": 0.0, "rotation_deg": 22.5, "scale": 0.0}
    result = generate_quantitative_question(cs, params)
    assert result is not None
    _, answer = result
    assert abs(answer - 22.5) < 1e-6


def test_quantitative_scale():
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.0, "translation_y": 0.0, "rotation_deg": 0.0, "scale": 0.5}
    result = generate_quantitative_question(cs, params)
    assert result is not None
    text, answer = result
    # Scale ground truth is 1 + scale_param = 1.5x
    assert abs(answer - 1.5) < 1e-6
    assert "factor" in text.lower()
    assert "relative" in text.lower()  # Asks for measurement, not correction


def test_quantitative_translation_only_returns_none():
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.2, "translation_y": 0.1, "rotation_deg": 0.0, "scale": 0.0}
    result = generate_quantitative_question(cs, params)
    assert result is None


def test_coord_identification():
    for name, expected in [
        ("linear_1d", "1D linear"),
        ("cartesian_2d", "2D Cartesian"),
        ("polar", "circular"),
        ("spherical", "circular"),
    ]:
        cs = get_coordinate_system(name)
        text, answer = generate_coord_identification_question(cs)
        assert answer == expected
        assert "coordinate system" in text.lower()
    # Template should offer 3-way choice
    cs = get_coordinate_system("cartesian_2d")
    text, _ = generate_coord_identification_question(cs)
    assert "circular" in text
