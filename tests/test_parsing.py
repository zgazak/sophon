"""Tests for the VLM response parser."""

import pytest

from sophon.parsing import (
    ParseResult,
    parse_binary,
    parse_categorical,
    parse_comparative,
    parse_coord_identification,
    parse_quantitative,
)


# =========================================================================
# Binary
# =========================================================================

class TestParseBinary:
    def test_simple_yes(self):
        assert parse_binary("Yes").parsed_answer == "yes"

    def test_simple_no(self):
        assert parse_binary("No").parsed_answer == "no"

    def test_verbose_yes(self):
        assert parse_binary("Yes, the points appear aligned").parsed_answer == "yes"

    def test_verbose_no(self):
        r = parse_binary("Based on my analysis, I believe they are not aligned")
        assert r.parsed_answer == "no"

    def test_last_keyword_wins_no_then_yes(self):
        # Model restates then concludes — last keyword wins
        r = parse_binary("No, wait. Looking more carefully, yes they are aligned.")
        assert r.parsed_answer == "yes"

    def test_last_keyword_wins_yes_then_no(self):
        r = parse_binary("The question asks if they are aligned. No, they are not aligned.")
        assert r.parsed_answer == "no"

    def test_restatement_then_answer(self):
        r = parse_binary(
            "The question asks whether they are aligned. "
            "Looking at the image, no, they are not aligned."
        )
        assert r.parsed_answer == "no"

    def test_not_keyword(self):
        r = parse_binary("They are definitely not aligned")
        assert r.parsed_answer == "no"

    def test_unparseable(self):
        r = parse_binary("I cannot determine this")
        assert r.parse_success is False
        assert r.parsed_answer is None

    def test_unparseable_hedging(self):
        r = parse_binary("It's hard to tell from this image")
        assert r.parse_success is False


# =========================================================================
# Categorical
# =========================================================================

class TestParseCategorical:
    def test_direct(self):
        vocab = ["left", "right", "up", "down"]
        assert parse_categorical("left", vocab).parsed_answer == "left"

    def test_embedded(self):
        vocab = ["left", "right", "up", "down"]
        r = parse_categorical("The red points should move to the left", vocab)
        assert r.parsed_answer == "left"

    def test_last_keyword_wins(self):
        vocab = ["left", "right"]
        r = parse_categorical(
            "The red points are shifted right. To align, they should move left.",
            vocab,
        )
        assert r.parsed_answer == "left"

    def test_last_keyword_wins_clockwise(self):
        vocab = ["clockwise", "counterclockwise"]
        r = parse_categorical(
            "The rotation appears counterclockwise. "
            "So the correction should be clockwise.",
            vocab,
        )
        assert r.parsed_answer == "clockwise"

    def test_no_match(self):
        vocab = ["left", "right"]
        r = parse_categorical("I'm not sure", vocab)
        assert r.parse_success is False

    def test_case_insensitive(self):
        vocab = ["inward", "outward"]
        r = parse_categorical("They should move OUTWARD", vocab)
        assert r.parsed_answer == "outward"

    def test_multiple_same_word(self):
        vocab = ["left", "right"]
        r = parse_categorical("Move left. Definitely left.", vocab)
        assert r.parsed_answer == "left"


# =========================================================================
# Comparative
# =========================================================================

class TestParseComparative:
    def test_simple_a(self):
        assert parse_comparative("A").parsed_answer == "A"

    def test_simple_b(self):
        assert parse_comparative("B").parsed_answer == "B"

    def test_image_b(self):
        assert parse_comparative("Image B shows better alignment").parsed_answer == "B"

    def test_last_keyword_wins(self):
        r = parse_comparative("Image A is close but Image B is better aligned.")
        assert r.parsed_answer == "B"

    def test_last_keyword_wins_reversed(self):
        r = parse_comparative("B has some alignment but A is clearly better.")
        assert r.parsed_answer == "A"

    def test_unparseable(self):
        r = parse_comparative("The first image")
        assert r.parse_success is False


# =========================================================================
# Quantitative
# =========================================================================

class TestParseQuantitative:
    # --- Priority 1: unit-adjacent ---

    def test_degrees_suffix(self):
        r = parse_quantitative("approximately 30.5 degrees")
        assert r.parsed_answer == pytest.approx(30.5)
        assert r.parse_success is True

    def test_degree_symbol(self):
        r = parse_quantitative("~30.5°")
        assert r.parsed_answer == pytest.approx(30.5)

    def test_deg_abbreviation(self):
        r = parse_quantitative("about 15 deg")
        assert r.parsed_answer == pytest.approx(15.0)

    def test_scale_factor_of(self):
        r = parse_quantitative("factor of 2.3")
        assert r.parsed_answer == pytest.approx(2.3)

    def test_scale_factor_colon(self):
        r = parse_quantitative("factor: 1.8")
        assert r.parsed_answer == pytest.approx(1.8)

    def test_scale_factor_is(self):
        r = parse_quantitative("The scale factor is 0.85")
        assert r.parsed_answer == pytest.approx(0.85)

    def test_scale_x_suffix(self):
        r = parse_quantitative("2.3x")
        assert r.parsed_answer == pytest.approx(2.3)

    def test_negative_degrees(self):
        r = parse_quantitative("-15.2 degrees")
        assert r.parsed_answer == pytest.approx(-15.2)

    def test_negative_with_abs(self):
        r = parse_quantitative("-34 degrees", take_abs=True)
        assert r.parsed_answer == pytest.approx(34.0)

    # --- Priority 2: last non-marker number ---

    def test_simple_number(self):
        r = parse_quantitative("30.5")
        assert r.parsed_answer == pytest.approx(30.5)
        assert r.parse_success is True

    def test_integer(self):
        r = parse_quantitative("15")
        assert r.parsed_answer == pytest.approx(15.0)

    def test_last_number_preferred(self):
        r = parse_quantitative(
            "The first set has 46 points. The rotation is 30."
        )
        assert r.parsed_answer == pytest.approx(30.0)

    def test_answer_at_end(self):
        r = parse_quantitative(
            "Based on my analysis of the 46 blue and 46 red points, "
            "I estimate the rotation to be approximately 25.3"
        )
        assert r.parsed_answer == pytest.approx(25.3)

    # --- Tutorial / list marker rejection ---

    def test_tutorial_response_fails(self):
        """Model writes a tutorial instead of answering — list markers only."""
        response = (
            "To estimate the rotation angle:\n"
            "1. Normalize the Point Sets\n"
            "2. Calculate the CCF\n"
            "3. Find the Peak"
        )
        r = parse_quantitative(response)
        assert r.parse_success is False

    def test_step_prefix_markers(self):
        """'Step N' markers should be rejected."""
        response = "Step 1: look at blue. Step 2: look at red. Step 3: compare."
        r = parse_quantitative(response)
        assert r.parse_success is False

    def test_answer_after_tutorial(self):
        """Model writes procedure then answers at the end."""
        response = (
            "1. First normalize the points\n"
            "2. Then calculate rotation\n"
            "The rotation angle is approximately 25.3 degrees."
        )
        r = parse_quantitative(response)
        assert r.parsed_answer == pytest.approx(25.3)
        assert r.parse_success is True

    def test_paren_list_markers(self):
        """'1)' style list markers should be rejected."""
        response = (
            "1) Identify point sets\n"
            "2) Compare positions\n"
            "3) Estimate angle"
        )
        r = parse_quantitative(response)
        assert r.parse_success is False

    def test_real_llava_tutorial_response(self):
        """Actual LLaVA-NeXT response that triggered the bug."""
        response = (
            " The image you've provided appears to be a scatter plot with two "
            "sets of points, one in blue and one in red. To estimate the rotation "
            "angle between these two sets, we can use the cross-correlation "
            "function (CCF) between the two sets of points.\n\n"
            "1. **Normalize the Point Sets**: Ensure that both sets of points are "
            "normalized to the same scale.\n\n"
            "2. **Calculate the CCF**: The CCF is calculated by sliding one set "
            "of points over the other.\n\n"
            "3. **Find the Peak of the CCF**: The peak corresponds to the rotation angle."
        )
        r = parse_quantitative(response)
        assert r.parse_success is False

    # --- No numbers ---

    def test_no_numbers(self):
        r = parse_quantitative("I cannot determine the rotation.")
        assert r.parse_success is False
        assert r.parsed_answer is None

    def test_text_only_refusal(self):
        r = parse_quantitative("This is impossible to estimate from the image.")
        assert r.parse_success is False

    # --- Edge cases ---

    def test_tilde_degrees(self):
        r = parse_quantitative("~15°")
        assert r.parsed_answer == pytest.approx(15.0)

    def test_multiple_unit_numbers_takes_last(self):
        r = parse_quantitative("Maybe 10 degrees, actually more like 25 degrees")
        assert r.parsed_answer == pytest.approx(25.0)

    def test_number_in_context_with_point_count(self):
        """Number like '46 points' should not be preferred over later answer."""
        r = parse_quantitative("There are 46 points. Rotation: 30 degrees.")
        assert r.parsed_answer == pytest.approx(30.0)

    def test_abs_with_unit_pattern(self):
        r = parse_quantitative("The angle is -22.5 degrees", take_abs=True)
        assert r.parsed_answer == pytest.approx(22.5)


# =========================================================================
# Coordinate identification
# =========================================================================

class TestParseCoordIdentification:
    def test_cartesian(self):
        r = parse_coord_identification("This is a 2D Cartesian coordinate system")
        assert r.parsed_answer == "2D Cartesian"

    def test_polar(self):
        r = parse_coord_identification("polar")
        assert r.parsed_answer == "circular"

    def test_spherical(self):
        r = parse_coord_identification("spherical")
        assert r.parsed_answer == "circular"

    def test_circular(self):
        r = parse_coord_identification("circular")
        assert r.parsed_answer == "circular"

    def test_fisheye_alias(self):
        r = parse_coord_identification("It looks like a fisheye projection")
        assert r.parsed_answer == "circular"

    def test_unparseable(self):
        r = parse_coord_identification("I don't know")
        assert r.parse_success is False

    def test_last_keyword_wins(self):
        r = parse_coord_identification(
            "At first it looks linear but actually this is 2D Cartesian"
        )
        assert r.parsed_answer == "2D Cartesian"

    def test_linear(self):
        r = parse_coord_identification("1D linear")
        assert r.parsed_answer == "1D linear"
