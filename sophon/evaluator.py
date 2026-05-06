"""Evaluation and scoring infrastructure.

Provides:
- OpenAI-compatible VLM adapter (works with OpenRouter, vllm, direct OpenAI)
- Scoring pipeline: run VLM against generated samples, produce predictions + results
- API key resolution with clear error messages
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import Image

from sophon import __version__
from sophon.parsing import (
    ParseResult,
    parse_binary,
    parse_categorical,
    parse_comparative,
    parse_quantitative,
)
from sophon.questions import _QUANTITATIVE_ROTATION_KEYS, QUANTITATIVE_KEYS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def _is_local(api_base: str | None) -> bool:
    """Return True if api_base points to a local server (no auth needed)."""
    if api_base is None:
        return False
    return api_base.startswith("http://")


def resolve_api_key(api_base: str | None) -> str | None:
    """Resolve the API key from environment variables.

    Resolution order:
    - If api_base points to localhost (http://), return None (no key needed)
    - If api_base contains 'openrouter.ai', try OPENROUTER_API_KEY
    - Try OPENAI_API_KEY
    - Try SOPHON_API_KEY
    - Return None if nothing found
    """
    # Try loading .env if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if _is_local(api_base):
        return None

    if api_base and "openrouter.ai" in api_base:
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            return key

    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    key = os.environ.get("SOPHON_API_KEY")
    if key:
        return key

    return None


def _api_key_error_message(api_base: str | None) -> str:
    """Generate a helpful error message when no API key is found."""
    if api_base and "openrouter.ai" in api_base:
        return (
            "Error: No API key found for OpenRouter.\n\n"
            "Set your API key as an environment variable:\n"
            "  export OPENROUTER_API_KEY=sk-or-...\n\n"
            "Get a key at: https://openrouter.ai/settings/keys\n\n"
            "Tip: OpenRouter provides unified access to GPT, Claude, Gemini, and\n"
            "200+ other models with a single API key."
        )
    return (
        "Error: No API key found.\n\n"
        "Set one of these environment variables:\n"
        "  export OPENAI_API_KEY=sk-...\n"
        "  export SOPHON_API_KEY=...\n\n"
        "For OpenRouter (recommended for frontier models):\n"
        "  export OPENROUTER_API_KEY=sk-or-...\n"
        "  Then use --api-base https://openrouter.ai/api/v1"
    )


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def _image_to_base64(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 data URL.

    Converts RGBA to RGB (white background) to avoid alpha channel
    issues with some vision API providers.
    """
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# OpenAI-compatible VLM adapter
# ---------------------------------------------------------------------------


def create_openai_vlm_fn(
    model_name: str,
    api_base: str | None = None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    extra_headers: dict[str, str] | None = None,
    max_requests_per_minute: int | None = None,
) -> Callable[[Image.Image | None, str, str | None], tuple[str, dict]]:
    """Create a VLM function using the OpenAI chat/completions API.

    Returns a callable (image_or_None, question, text_description) -> (response_text, usage_dict).
    Works with OpenRouter, vllm, direct OpenAI, and any OpenAI-compatible endpoint.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for evaluation. "
            "Install it with: pip install sophon-bench[eval]"
        )

    client_kwargs: dict = {}
    if api_base:
        client_kwargs["base_url"] = api_base
    if api_key:
        client_kwargs["api_key"] = api_key
    elif not _is_local(api_base):
        # For non-local endpoints, we need a key even if it's a dummy
        resolved = resolve_api_key(api_base)
        if resolved:
            client_kwargs["api_key"] = resolved
        else:
            raise ValueError(_api_key_error_message(api_base))
    else:
        # Local endpoint: use a dummy key
        client_kwargs["api_key"] = "not-needed"

    if extra_headers:
        client_kwargs["default_headers"] = extra_headers

    client = OpenAI(**client_kwargs)

    # Rate limiting state
    _request_times: list[float] = []

    def vlm_fn(
        image: Image.Image | None,
        question: str,
        text_description: str | None = None,
    ) -> tuple[str, dict]:
        # Rate limiting
        if max_requests_per_minute:
            now = time.monotonic()
            _request_times[:] = [t for t in _request_times if now - t < 60]
            if len(_request_times) >= max_requests_per_minute:
                wait = 60 - (now - _request_times[0])
                if wait > 0:
                    time.sleep(wait)
            _request_times.append(time.monotonic())

        # Build messages
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user content
        user_content: list[dict] = []
        if image is not None:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_base64(image),
                        "detail": "high",
                    },
                }
            )
        if text_description:
            user_content.append(
                {
                    "type": "text",
                    "text": text_description,
                }
            )
        user_content.append(
            {
                "type": "text",
                "text": question,
            }
        )

        messages.append({"role": "user", "content": user_content})

        # Retry with exponential backoff
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                break
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"Rate limited, retrying in {wait}s (attempt {attempt + 1})")
                    time.sleep(wait)
                    continue
                raise

        text = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens
            usage["completion_tokens"] = response.usage.completion_tokens
            usage["total_tokens"] = response.usage.total_tokens
        # OpenRouter returns cost in usage — field is "cost" (not "total_cost")
        if response.usage:
            raw_usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else {}
            # Try "cost" first (OpenRouter's actual field name), then "total_cost"
            for key in ("cost", "total_cost"):
                if key in raw_usage:
                    usage["total_cost"] = float(raw_usage[key])
                    break

        return text, usage

    return vlm_fn


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _parse_response(
    question_type: str,
    raw_response: str,
    direction_vocabulary: list[str] | None = None,
) -> ParseResult:
    """Parse a VLM response for the given question type."""
    if question_type == "binary":
        return parse_binary(raw_response)
    elif question_type == "categorical":
        return parse_categorical(raw_response, direction_vocabulary or [])
    elif question_type == "comparative":
        return parse_comparative(raw_response)
    elif question_type == "quantitative":
        return parse_quantitative(raw_response, take_abs=True)
    else:
        return ParseResult(raw_response=raw_response, parsed_answer=None, parse_success=False)


def _is_correct(
    question_type: str,
    parsed_answer: str | float | None,
    ground_truth: str | float,
) -> bool | None:
    """Check if a parsed answer is correct. Returns None if unparseable."""
    if parsed_answer is None:
        return None
    if question_type == "quantitative":
        return None  # Quantitative uses error, not correct/incorrect
    # String comparison (case-insensitive)
    return str(parsed_answer).lower() == str(ground_truth).lower()


def _quantitative_error(
    parsed_answer: str | float | None,
    ground_truth: float,
) -> float | None:
    """Compute absolute error for quantitative answers."""
    if parsed_answer is None:
        return None
    try:
        return abs(float(parsed_answer) - float(ground_truth))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_samples(
    vlm_fn: Callable,
    samples: list[tuple[Image.Image | None, dict]],
    direction_vocabularies: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Run VLM on samples and produce per-sample prediction records.

    Args:
        vlm_fn: Callable (image, question, text_description) -> (response_text, usage)
        samples: List of (image_or_None, sample_dict) where sample_dict is a Sample.model_dump()
        direction_vocabularies: Map from coordinate_system name -> direction vocabulary list

    Returns:
        List of prediction dicts (one per sample).
    """
    predictions = []
    for image, sample in samples:
        text_desc = sample.get("text_description")
        q_text = sample["question_text"]
        q_type = sample["question_type"]
        gt = sample["ground_truth"]
        cs_name = sample["coordinate_system"]

        raw_response, usage = vlm_fn(image, q_text, text_desc)

        # Prefer sample-level restricted vocab when available
        vocab = sample.get("valid_directions")
        if vocab is None and direction_vocabularies and cs_name in direction_vocabularies:
            vocab = direction_vocabularies[cs_name]

        parsed = _parse_response(q_type, raw_response, vocab)
        correct = _is_correct(q_type, parsed.parsed_answer, gt)
        error = None
        if q_type == "quantitative":
            error = _quantitative_error(parsed.parsed_answer, gt)

        pred = {
            "sample_id": sample.get("sample_id"),
            "coordinate_system": cs_name,
            "question_type": q_type,
            "transform_params": sample.get("transform_params"),
            "transform_magnitude": sample.get("transform_magnitude"),
            "transform_direction": sample.get("transform_direction"),
            "num_points": sample.get("num_points"),
            "show_grid": sample.get("show_grid"),
            "ground_truth": gt,
            "raw_response": raw_response,
            "parsed_answer": parsed.parsed_answer,
            "parse_success": parsed.parse_success,
            "correct": correct,
            "error": error,
        }
        if usage:
            pred["usage"] = usage

        predictions.append(pred)

    return predictions


# ---------------------------------------------------------------------------
# Aggregated results
# ---------------------------------------------------------------------------

# Chance baselines per coordinate system for categorical questions
_CATEGORICAL_CHANCES: dict[str, float] = {
    "linear_1d": 1.0 / 2,  # 2 directions
    "cartesian_2d": 1.0 / 6,  # 6 directions (but typically 4 used for single-axis)
    "polar": 1.0 / 4,  # 4 directions
}


def aggregate_results(
    predictions: list[dict],
    spec_name: str,
    model_name: str,
    seed: int,
    num_samples: int,
) -> dict:
    """Aggregate per-sample predictions into summary results.

    Returns a results dict matching the schema in the design doc.
    """
    # Group predictions by (coordinate_system, question_type)
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for pred in predictions:
        key = (pred["coordinate_system"], pred["question_type"])
        groups[key].append(pred)

    # Count parse successes
    total = len(predictions)
    parsed = sum(1 for p in predictions if p["parse_success"])
    parse_rate = parsed / total if total > 0 else 0.0

    results: dict[str, dict] = {}
    for (cs_name, q_type), preds in groups.items():
        if cs_name not in results:
            results[cs_name] = {}

        n = len(preds)
        if q_type == "quantitative":
            errors = [p["error"] for p in preds if p["error"] is not None]
            mae = sum(errors) / len(errors) if errors else None

            # Binned accuracy for rotation
            has_rotation = any(
                any(
                    abs(p.get("transform_params", {}).get(k, 0.0)) > 1e-6
                    for k in _QUANTITATIVE_ROTATION_KEYS
                )
                for p in preds
            )
            binned = {}
            if has_rotation and errors:
                for threshold_name, threshold in [("5deg", 5), ("10deg", 10), ("25deg", 25)]:
                    binned[threshold_name] = sum(1 for e in errors if e <= threshold) / len(errors)

            # Scale binned accuracy
            has_scale = any(
                any(
                    abs(p.get("transform_params", {}).get(k, 0.0)) > 1e-6
                    for k in QUANTITATIVE_KEYS - _QUANTITATIVE_ROTATION_KEYS
                )
                for p in preds
            )
            if has_scale and errors:
                for threshold_name, threshold in [("0.1x", 0.1), ("0.25x", 0.25)]:
                    binned[threshold_name] = sum(1 for e in errors if e <= threshold) / len(errors)

            entry = {"mae": mae, "n": n}
            if binned:
                entry["binned_accuracy"] = binned
            results[cs_name][q_type] = entry
        else:
            correct_preds = [p for p in preds if p["correct"] is True]
            parseable = [p for p in preds if p["correct"] is not None]
            accuracy = len(correct_preds) / len(parseable) if parseable else None

            chance = 0.5
            if q_type == "categorical":
                chance = _CATEGORICAL_CHANCES.get(cs_name, 0.25)
            elif q_type in ("binary", "comparative"):
                chance = 0.5

            results[cs_name][q_type] = {
                "accuracy": accuracy,
                "n": n,
                "chance": chance,
            }

    # Build chance baselines
    chance_baselines = {
        "binary": 0.50,
        "categorical": {cs: _CATEGORICAL_CHANCES.get(cs, 0.25) for cs in _CATEGORICAL_CHANCES},
        "comparative": 0.50,
    }

    return {
        "metadata": {
            "spec_name": spec_name,
            "model_name": model_name,
            "seed": seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sophon_version": __version__,
            "num_samples": num_samples,
            "parse_rate": round(parse_rate, 4),
        },
        "results": results,
        "chance_baselines": chance_baselines,
    }


def score_predictions(
    predictions_path: Path,
    ground_truth_path: Path,
) -> dict:
    """Score a predictions file against ground truth.

    predictions_path: JSONL file with per-sample predictions
    ground_truth_path: JSON file with {sample_id: {question_type, ground_truth, coordinate_system}}

    Returns aggregated results dict.
    """
    with open(ground_truth_path) as f:
        gt = json.load(f)

    predictions = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    # Enrich predictions with ground truth if needed
    for pred in predictions:
        sid = str(pred.get("sample_id", ""))
        if sid in gt:
            if "ground_truth" not in pred:
                pred["ground_truth"] = gt[sid]["ground_truth"]
            if "question_type" not in pred:
                pred["question_type"] = gt[sid]["question_type"]
            if "coordinate_system" not in pred:
                pred["coordinate_system"] = gt[sid]["coordinate_system"]

    return aggregate_results(
        predictions,
        spec_name="scored",
        model_name="unknown",
        seed=0,
        num_samples=len(predictions),
    )
