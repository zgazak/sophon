"""Parse free-form VLM text responses into structured answers."""

from __future__ import annotations

import re

from pydantic import BaseModel


class ParseResult(BaseModel):
    """Result of parsing a VLM response."""

    raw_response: str
    parsed_answer: str | float | None
    parse_success: bool


def parse_binary(response: str) -> ParseResult:
    """Extract yes/no from a VLM response."""
    text = response.lower()
    yes_match = re.search(r"\byes\b", text)
    no_match = re.search(r"\b(no|not)\b", text)
    if yes_match and no_match:
        # Take whichever appears first
        answer = "yes" if yes_match.start() < no_match.start() else "no"
    elif yes_match:
        answer = "yes"
    elif no_match:
        answer = "no"
    else:
        return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)
    return ParseResult(raw_response=response, parsed_answer=answer, parse_success=True)


def parse_categorical(response: str, vocabulary: list[str]) -> ParseResult:
    """Extract a direction word from the response."""
    text = response.lower()
    for word in vocabulary:
        if re.search(r"\b" + re.escape(word.lower()) + r"\b", text):
            return ParseResult(raw_response=response, parsed_answer=word, parse_success=True)
    return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)


def parse_comparative(response: str) -> ParseResult:
    """Extract A or B from the response."""
    # Look for standalone A or B, or "Image A", "Image B"
    text = response.strip()
    a_match = re.search(r"\b[Aa]\b", text)
    b_match = re.search(r"\b[Bb]\b", text)
    if a_match and b_match:
        answer = "A" if a_match.start() < b_match.start() else "B"
    elif a_match:
        answer = "A"
    elif b_match:
        answer = "B"
    else:
        return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)
    return ParseResult(raw_response=response, parsed_answer=answer, parse_success=True)


def parse_quantitative(response: str, take_abs: bool = False) -> ParseResult:
    """Extract a numeric value from the response.

    If take_abs is True (used for rotation), takes absolute value of the
    extracted number — the model might say "-34 degrees" or "34 degrees
    clockwise" and both should yield 34.
    """
    # Find first number (int or float, possibly negative)
    match = re.search(r"[-+]?\d*\.?\d+", response)
    if match:
        value = float(match.group())
        if take_abs:
            value = abs(value)
        return ParseResult(raw_response=response, parsed_answer=value, parse_success=True)
    return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)


def parse_coord_identification(response: str) -> ParseResult:
    """Match response against coordinate system names (3-way classification)."""
    text = response.lower()
    aliases = {
        "1d linear": "1D linear",
        "linear": "1D linear",
        "1d": "1D linear",
        "2d cartesian": "2D Cartesian",
        "cartesian": "2D Cartesian",
        "2d": "2D Cartesian",
        "circular": "circular",
        "polar": "circular",
        "spherical": "circular",
        "fisheye": "circular",
    }
    for alias, canonical in aliases.items():
        if alias in text:
            return ParseResult(
                raw_response=response, parsed_answer=canonical, parse_success=True
            )
    return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)
