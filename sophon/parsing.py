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

_ANSWER_MARKER_RE = re.compile(
    r"\b(?:final\s+)?answer(?:\s+is)?\s*[:=\-]?[\s*]*([AB])\b",
    re.IGNORECASE,
)


def parse_comparative(response: str) -> ParseResult:
    """Extract A or B from the response.

    Strategy (in priority order):

    0. **Explicit answer marker (last-wins)**: ``Final Answer: B``,
       ``**Answer: B**``, ``answer is A``. When the model uses an explicit
       conclusion marker, the *last* such marker wins (handles
       ``initially I thought A, but on reflection: B``). This is needed for
       structured-response models (e.g. Qwen2.5-VL-32B) that write
       ``**Image A:** ... **Image B:** ... **Final Answer: B**`` — without
       this rule, the first-keyword-wins fallback grabs the "A" from the
       ``**Image A**`` header.
    1. **Bolded answer**: ``**A**`` or ``**B**`` — models often bold their
       answer. Uses first match (the direct answer, not a mention in
       explanation).
    2. **First-keyword-wins**: the first standalone A/B in the response.
       Verbose models (e.g. Opus) state their answer first, then discuss
       both options.  Last-keyword-wins would pick up the *other* label
       from the contrastive explanation.
    """
    text = response.strip()

    # Priority 0: explicit "answer" marker (last-wins)
    markers = list(_ANSWER_MARKER_RE.finditer(text))
    if markers:
        return ParseResult(
            raw_response=response,
            parsed_answer=markers[-1].group(1).upper(),
            parse_success=True,
        )

    # Priority 1: bolded answer  **A** or **B**
    bold = re.search(r"\*\*([AaBb])\*\*", text)
    if bold:
        return ParseResult(
            raw_response=response,
            parsed_answer=bold.group(1).upper(),
            parse_success=True,
        )

    # Priority 2: first standalone A or B
    a_positions = [m.start() for m in re.finditer(r"\b[Aa]\b", text)]
    b_positions = [m.start() for m in re.finditer(r"\b[Bb]\b", text)]

    first_a = a_positions[0] if a_positions else float("inf")
    first_b = b_positions[0] if b_positions else float("inf")

    if first_a == float("inf") and first_b == float("inf"):
        return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)

    answer = "A" if first_a < first_b else "B"
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


def _is_percentage(match: re.Match, text: str) -> bool:
    """Check if a number match is followed by '%', making it a percentage."""
    end = match.end()
    after = text[end:end + 2].lstrip()
    return after.startswith("%")


def _is_range_hyphen_component(match: re.Match, text: str) -> bool:
    """Check if a negative-looking number is actually the second half of a
    range like "100-120" or "0.5-1.5", where `_NUMBER_RE`'s optional sign
    greedily grabs the hyphen as a minus sign.
    """
    start = match.span()[0]
    group = match.group()
    if not group.startswith(("-", "+")):
        return False
    # Look just before the sign char
    before = text[max(0, start - 3):start]
    # Match "DIGIT-" with optional whitespace around the hyphen
    if re.search(r"\d\s*$", before):
        return True
    return False


def _is_coordinate_pair(match: re.Match, text: str) -> bool:
    """Check if a number is part of a coordinate pair like (380, 255).

    VLMs often describe pixel/image coordinates during their reasoning
    ("center at (380, 255)", "point [120, 340]", "coordinates 380,255") —
    these are not answers and should be excluded from the fallback number
    extractor.

    Triggers on:
      - Number inside parentheses/brackets with ≥2 numbers total
      - Number immediately followed by ", " and another number
      - Number immediately preceded by another number + ", "
    """
    start, end = match.span()

    # Scan for enclosing brackets within ~30 chars
    left_window = text[max(0, start - 40):start]
    right_window = text[end:end + 40]

    for open_ch, close_ch in [("(", ")"), ("[", "]")]:
        open_pos = left_window.rfind(open_ch)
        close_pos = right_window.find(close_ch)
        if open_pos == -1 or close_pos == -1:
            continue
        # Check no closer of the same type intervenes (stay inside a single bracket)
        inner_close = left_window[open_pos + 1:].find(close_ch)
        if inner_close != -1:
            continue
        inner_open = right_window[:close_pos].find(open_ch)
        if inner_open != -1:
            continue
        bracket_text = left_window[open_pos:] + match.group() + right_window[:close_pos + 1]
        if len(list(_NUMBER_RE.finditer(bracket_text))) >= 2:
            return True

    # "NUM, NUM" pattern without brackets — e.g., "at 380, 255"
    if re.match(r"\s*,\s*[-+]?\d*\.?\d+", right_window):
        return True
    if re.search(r"[-+]?\d*\.?\d+\s*,\s*$", left_window):
        return True

    return False


def parse_quantitative(response: str, take_abs: bool = False) -> ParseResult:
    """Extract a numeric value from the response.

    Priority order:
    1. Bolded number (``**0.95**``) — models often bold their answer;
       this is the strongest signal of intent.
    2. Number adjacent to a unit keyword ("30 degrees", "factor of 2.3")
    3. Last non-list-marker, non-percentage number in the response
    4. If only list markers / percentages found → parse_success=False

    If take_abs is True (used for rotation), takes absolute value.
    """
    text = response.strip()

    # Priority 1: bolded number  **1.23** or **~1.23**
    bold_matches = list(re.finditer(r"\*\*~?([-+]?\d*\.?\d+)\*\*", text))
    if bold_matches:
        # Take the last bolded number (final answer, not intermediate)
        value = float(bold_matches[-1].group(1))
        if take_abs:
            value = abs(value)
        return ParseResult(raw_response=response, parsed_answer=value, parse_success=True)

    # Priority 2: unit-adjacent number
    for pattern in _UNIT_PATTERNS:
        # Search the whole text; take the LAST match if multiple
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            value = float(matches[-1].group(1))
            if take_abs:
                value = abs(value)
            return ParseResult(raw_response=response, parsed_answer=value, parse_success=True)

    # Priority 3: last non-list-marker, non-percentage, non-coordinate-pair,
    # non-range-component number
    all_matches = list(_NUMBER_RE.finditer(text))
    non_markers = [
        m for m in all_matches
        if not _is_list_marker(m, text)
        and not _is_percentage(m, text)
        and not _is_coordinate_pair(m, text)
        and not _is_range_hyphen_component(m, text)
    ]

    if non_markers:
        value = float(non_markers[-1].group())
        if take_abs:
            value = abs(value)
        return ParseResult(raw_response=response, parsed_answer=value, parse_success=True)

    # Priority 4: only list markers / percentages found, or no numbers at all
    return ParseResult(raw_response=response, parsed_answer=None, parse_success=False)


