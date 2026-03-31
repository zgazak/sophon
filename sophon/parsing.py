"""Parse free-form VLM text responses into structured answers.

Design principles:
- **Last-keyword-wins**: models often restate the question before answering.
  "The question asks if they are aligned. No, they are not." — answer is "no".
- **Reject garbage extractions**: a number scraped from "1. Normalize the
  Point Sets" is not an answer. parse_success=False means "no genuine answer
  attempt detected", not "model got it wrong".
- **Priority extraction for quantitative**: unit-adjacent numbers first,
  then last non-list-marker number, then reject.
"""

from __future__ import annotations

import re

from pydantic import BaseModel


class ParseResult(BaseModel):
    """Result of parsing a VLM response."""

    raw_response: str
    parsed_answer: str | float | None
    parse_success: bool


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------

def parse_binary(response: str) -> ParseResult:
    """Extract yes/no from a VLM response.

    Uses last-keyword-wins: if both "yes" and "no" appear, the one that
    appears later is taken as the model's conclusion.
    """
    text = response.lower()

    # Find ALL occurrences, take the last
    yes_positions = [m.start() for m in re.finditer(r"\byes\b", text)]
    no_positions = [m.start() for m in re.finditer(r"\b(?:no|not)\b", text)]

    last_yes = yes_positions[-1] if yes_positions else -1
    last_no = no_positions[-1] if no_positions else -1

    if last_yes < 0 and last_no < 0:
        return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)

    answer = "yes" if last_yes > last_no else "no"
    return ParseResult(raw_response=response, parsed_answer=answer, parse_success=True)


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

def parse_categorical(response: str, vocabulary: list[str]) -> ParseResult:
    """Extract a direction word from the response.

    Uses last-keyword-wins: if the model says "shifted right ... move left",
    the answer is "left" (the last direction mentioned).
    """
    text = response.lower()
    last_word = None
    last_pos = -1

    for word in vocabulary:
        for m in re.finditer(r"\b" + re.escape(word.lower()) + r"\b", text):
            if m.start() > last_pos:
                last_pos = m.start()
                last_word = word

    if last_word is None:
        return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)
    return ParseResult(raw_response=response, parsed_answer=last_word, parse_success=True)


# ---------------------------------------------------------------------------
# Comparative
# ---------------------------------------------------------------------------

def parse_comparative(response: str) -> ParseResult:
    """Extract A or B from the response.

    Uses last-keyword-wins.  Matches standalone A/B as well as
    "Image A", "Panel B", etc.
    """
    text = response.strip()

    a_positions = [m.start() for m in re.finditer(r"\b[Aa]\b", text)]
    b_positions = [m.start() for m in re.finditer(r"\b[Bb]\b", text)]

    last_a = a_positions[-1] if a_positions else -1
    last_b = b_positions[-1] if b_positions else -1

    if last_a < 0 and last_b < 0:
        return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)

    answer = "A" if last_a > last_b else "B"
    return ParseResult(raw_response=response, parsed_answer=answer, parse_success=True)


# ---------------------------------------------------------------------------
# Quantitative
# ---------------------------------------------------------------------------

# Patterns for numbers adjacent to unit keywords (Priority 1)
_UNIT_PATTERNS = [
    # Rotation: "30 degrees", "30.5 deg", "30°", "approximately 30 degrees"
    r"([-+]?\d*\.?\d+)\s*(?:degrees?|deg|°)",
    # Scale: number then unit — "2.3x", "2.3×"
    r"([-+]?\d*\.?\d+)\s*(?:×|x)\b",
    # Scale: "factor of 2.3", "factor: 2.3", "factor is 2.3"
    r"(?:factor\s*(?:of|:|is)\s*)([-+]?\d*\.?\d+)",
]

# General number pattern
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def _is_list_marker(match: re.Match, text: str) -> bool:
    """Check if a number match is a list/step marker, not an answer.

    Detects patterns like:
    - "1. Normalize" (numbered list)
    - "1. **Normalize" (markdown numbered list)
    - "2) Calculate"  (paren list)
    - "Step 1", "Item 2" (prefixed)
    """
    start, end = match.span()
    after = text[end:end + 10]

    # "1. N" or "2. C" — digit, period, space, then uppercase letter
    if re.match(r"\.\s+[A-Z]", after):
        return True
    # "1. **Bold" — digit, period, space, markdown bold
    if re.match(r"\.\s+\*\*", after):
        return True
    # "1.\n" — digit, period, end of line (but NOT "30." at end of sentence
    # preceded by a word char — that's an answer ending with a period)
    if re.match(r"\.\s*\n", after) or re.match(r"\.\s*$", after):
        # Only treat as marker if the number is a bare integer at line start
        # or after a newline, not mid-sentence
        line_start = text.rfind("\n", 0, start) + 1
        prefix = text[line_start:start].strip()
        if prefix == "" or prefix.endswith(":"):
            return True
    # "1)" or "2) " — digit, close-paren
    if after.startswith(")"):
        return True
    # Step/item prefix: "Step 1", "Item 2", "step 1"
    before = text[max(0, start - 6):start].lower()
    if re.search(r"(?:step|item)\s*$", before):
        return True

    return False


def parse_quantitative(response: str, take_abs: bool = False) -> ParseResult:
    """Extract a numeric value from the response.

    Priority order:
    1. Number adjacent to a unit keyword ("30 degrees", "factor of 2.3")
    2. Last non-list-marker number in the response
    3. If only list markers found → parse_success=False

    If take_abs is True (used for rotation), takes absolute value.
    """
    text = response.strip()

    # Priority 1: unit-adjacent number
    for pattern in _UNIT_PATTERNS:
        # Search the whole text; take the LAST match if multiple
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            value = float(matches[-1].group(1))
            if take_abs:
                value = abs(value)
            return ParseResult(raw_response=response, parsed_answer=value, parse_success=True)

    # Priority 2: last non-list-marker number
    all_matches = list(_NUMBER_RE.finditer(text))
    non_markers = [m for m in all_matches if not _is_list_marker(m, text)]

    if non_markers:
        value = float(non_markers[-1].group())
        if take_abs:
            value = abs(value)
        return ParseResult(raw_response=response, parsed_answer=value, parse_success=True)

    # Priority 3: only list markers found, or no numbers at all
    return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)


# ---------------------------------------------------------------------------
# Coordinate identification
# ---------------------------------------------------------------------------

def parse_coord_identification(response: str) -> ParseResult:
    """Match response against coordinate system names (3-way classification).

    Uses last-keyword-wins for consistency with other parsers.
    """
    text = response.lower()

    # Ordered from most specific to least specific to avoid
    # "2d cartesian" matching "2d" before the full phrase.
    aliases = [
        ("1d linear", "1D linear"),
        ("linear", "1D linear"),
        ("1d", "1D linear"),
        ("2d cartesian", "2D Cartesian"),
        ("cartesian", "2D Cartesian"),
        ("2d", "2D Cartesian"),
        ("circular", "circular"),
        ("polar", "circular"),
        ("spherical", "circular"),
        ("fisheye", "circular"),
    ]

    last_answer = None
    last_pos = -1

    for alias, canonical in aliases:
        # Find all occurrences of this alias
        for m in re.finditer(re.escape(alias), text):
            if m.start() > last_pos:
                last_pos = m.start()
                last_answer = canonical

    if last_answer is None:
        return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)
    return ParseResult(raw_response=response, parsed_answer=last_answer, parse_success=True)
