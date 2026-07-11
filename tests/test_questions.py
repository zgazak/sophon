"""Tests for question generation."""

from sophon.coordinates import get_coordinate_system
from sophon.questions import (
    generate_binary_question,
    generate_categorical_question,
    generate_comparative_question,
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


def test_binary_linear_1d_offset_phrasing():
    """Linear-1D binary template asks about horizontal *offset* rather than
    *alignment*, to sidestep the axis-reading ambiguity (parallel rows always
    'lie along horizontal axes' so 'aligned' degenerates)."""
    text, _ = generate_binary_question(is_aligned=True, coord_system_name="linear_1d")
    assert "offset" in text.lower()
    assert "horizontal" in text.lower()
    # Other coord systems still use the default template.
    text_default, _ = generate_binary_question(is_aligned=True, coord_system_name="cartesian_2d")
    assert "offset" not in text_default.lower()


def test_categorical_returns_inverse():
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.2, "translation_y": 0.0, "rotation_deg": 0.0, "scale": 0.0}
    text, answer, vocab = generate_categorical_question(cs, params)
    assert answer == "left"  # Inverse of "right"
    assert "direction" in text.lower()


def test_categorical_restricted_vocab():
    """Restricted vocab only includes the relevant direction pair."""
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.2, "translation_y": 0.0, "rotation_deg": 0.0, "scale": 0.0}
    text, answer, vocab = generate_categorical_question(cs, params, vocab_mode="restricted")
    assert vocab == ["left", "right"]
    assert "left, right" in text
    assert "clockwise" not in text
    assert "shortest arc" not in text  # translation → no arc language


def test_categorical_rotation_uses_arc_template():
    """Rotation questions use 'shortest arc' phrasing."""
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.0, "translation_y": 0.0, "rotation_deg": 15.0, "scale": 0.0}
    text, _, _ = generate_categorical_question(cs, params)
    assert "shortest arc" in text


def test_categorical_full_vocab():
    """Full vocab includes all direction words for the coordinate system."""
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.2, "translation_y": 0.0, "rotation_deg": 0.0, "scale": 0.0}
    text, answer, vocab = generate_categorical_question(cs, params, vocab_mode="full")
    assert vocab == cs.direction_vocabulary
    assert "clockwise" in text


def test_categorical_y_matches_visual_direction():
    cs = get_coordinate_system("cartesian_2d")
    # Renderer: pixel_y = image_size - native_y, drawn on matplotlib y-up axis.
    # Net: positive ty (higher native y) appears visually DOWN, so the
    # correction (red -> blue) should be UP.
    pos = {"translation_x": 0.0, "translation_y": 0.2, "rotation_deg": 0.0, "scale": 0.0}
    _, answer_pos, _ = generate_categorical_question(cs, pos)
    assert answer_pos == "up"

    neg = {"translation_x": 0.0, "translation_y": -0.2, "rotation_deg": 0.0, "scale": 0.0}
    _, answer_neg, _ = generate_categorical_question(cs, neg)
    assert answer_neg == "down"


def test_categorical_valid_direction():
    for name in ["linear_1d", "cartesian_2d", "polar"]:
        cs = get_coordinate_system(name)
        # Make a non-zero transform
        params = {k: 0.1 for k in cs.supported_transforms}
        _, answer, _ = generate_categorical_question(cs, params)
        assert answer in _DIRECTION_INVERSES


def test_categorical_restricted_vocab_all_systems():
    """Restricted vocab works for all single-transform specs across coord systems."""
    from sophon.questions import RESTRICTED_VOCAB

    for (cs_name, transform_key), expected_vocab in RESTRICTED_VOCAB.items():
        cs = get_coordinate_system(cs_name)
        params = {k: 0.0 for k in cs.supported_transforms}
        params[transform_key] = 0.2  # positive nonzero
        _, answer, vocab = generate_categorical_question(cs, params, vocab_mode="restricted")
        assert vocab == expected_vocab, f"({cs_name}, {transform_key}): {vocab} != {expected_vocab}"
        assert answer in vocab, f"GT '{answer}' not in vocab {vocab} for ({cs_name}, {transform_key})"


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


def test_quantitative_no_range_hint_by_default():
    """Default (range_hint=False) produces no range info in question text."""
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.0, "translation_y": 0.0, "rotation_deg": -15.3, "scale": 0.0}
    result = generate_quantitative_question(cs, params)
    assert result is not None
    text, _ = result
    assert "between 0 and" not in text


def test_quantitative_range_hint_rotation():
    """range_hint=True appends the rotation range to the question."""
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.0, "translation_y": 0.0, "rotation_deg": -15.3, "scale": 0.0}
    ranges = {"rotation_deg": [-45, 45]}
    result = generate_quantitative_question(cs, params, range_hint=True, transform_ranges=ranges)
    assert result is not None
    text, answer = result
    assert "between 0 and 45 degrees" in text
    assert abs(answer - 15.3) < 1e-6  # Ground truth unchanged


def test_quantitative_range_hint_scale():
    """range_hint=True appends the scale range to the question."""
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.0, "translation_y": 0.0, "rotation_deg": 0.0, "scale": 0.2}
    ranges = {"scale": [-0.3, 0.3]}
    result = generate_quantitative_question(cs, params, range_hint=True, transform_ranges=ranges)
    assert result is not None
    text, answer = result
    assert "between 0.70 and 1.30" in text
    assert abs(answer - 1.2) < 1e-6  # Ground truth unchanged


def test_quantitative_range_hint_false_no_change():
    """range_hint=False with ranges provided still produces no hint."""
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.0, "translation_y": 0.0, "rotation_deg": 15.0, "scale": 0.0}
    ranges = {"rotation_deg": [-45, 45]}
    result_no_hint = generate_quantitative_question(cs, params, range_hint=False, transform_ranges=ranges)
    result_default = generate_quantitative_question(cs, params)
    assert result_no_hint is not None and result_default is not None
    assert result_no_hint[0] == result_default[0]  # Same question text


def test_quantitative_translation_only_returns_none():
    cs = get_coordinate_system("cartesian_2d")
    params = {"translation_x": 0.2, "translation_y": 0.1, "rotation_deg": 0.0, "scale": 0.0}
    result = generate_quantitative_question(cs, params)
    assert result is None


