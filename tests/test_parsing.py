"""Tests for the VLM response parser."""

from sophon.parsing import (
    parse_binary,
    parse_categorical,
    parse_comparative,
    parse_coord_identification,
    parse_quantitative,
)


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

    def test_both_takes_first(self):
        r = parse_binary("No, they are not aligned. Yes might be wrong.")
        assert r.parsed_answer == "no"

    def test_unparseable(self):
        r = parse_binary("I cannot determine this")
        assert r.parse_success is False
        assert r.parsed_answer is None


class TestParseCategorical:
    def test_direct(self):
        vocab = ["left", "right", "up", "down"]
        assert parse_categorical("left", vocab).parsed_answer == "left"

    def test_embedded(self):
        vocab = ["left", "right", "up", "down"]
        r = parse_categorical("The red points should move to the left", vocab)
        assert r.parsed_answer == "left"

    def test_no_match(self):
        vocab = ["left", "right"]
        r = parse_categorical("I'm not sure", vocab)
        assert r.parse_success is False


class TestParseComparative:
    def test_simple_a(self):
        assert parse_comparative("A").parsed_answer == "A"

    def test_image_b(self):
        assert parse_comparative("Image B shows better alignment").parsed_answer == "B"

    def test_unparseable(self):
        r = parse_comparative("The first image")
        assert r.parse_success is False


class TestParseQuantitative:
    def test_integer(self):
        assert parse_quantitative("15").parsed_answer == 15.0

    def test_float(self):
        assert parse_quantitative("15.3").parsed_answer == 15.3

    def test_with_units(self):
        r = parse_quantitative("approximately 15 degrees")
        assert r.parsed_answer == 15.0

    def test_negative(self):
        assert parse_quantitative("-7.5").parsed_answer == -7.5

    def test_negative_with_abs(self):
        assert parse_quantitative("-34 degrees", take_abs=True).parsed_answer == 34.0

    def test_tilde(self):
        r = parse_quantitative("~15°")
        assert r.parsed_answer == 15.0

    def test_no_number(self):
        r = parse_quantitative("I cannot estimate this")
        assert r.parse_success is False


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
