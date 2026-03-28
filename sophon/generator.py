"""Core generator: spec + seed → deterministic (image, question, answer) triples."""

from __future__ import annotations

from typing import Iterator

import numpy as np
from PIL import Image

from sophon.coordinates import get_coordinate_system
from sophon.coordinates.base import CoordinateSystem
from sophon.questions import (
    generate_binary_question,
    generate_categorical_question,
    generate_comparative_question,
    generate_coord_identification_question,
    generate_quantitative_question,
    has_quantitative_transform,
)
from sophon.renderer import render_comparative, render_sample
from sophon.schema import BenchmarkSpec, QuestionConfig, RenderingConfig, Sample
from sophon.transforms import make_identity_transform

_ALL_COORD_SYSTEMS = ["linear_1d", "cartesian_2d", "polar"]

# Minimum misalignment in native coords, calibrated to ~3-4 dot diameters.
# At 512px, point_radius=6, margin=0.1: 1 native unit ≈ 410px usable,
# so 1 dot diameter (12px) ≈ 0.029 native. 3.5 diameters ≈ 0.10 native.
# For angular transforms (degrees), 3.5 diameters at mid-radius ≈ 8°.
_MIN_MISALIGNMENT: dict[str, float] = {
    "translation_x": 0.10,
    "translation_y": 0.10,
    "rotation_deg": 8.0,
    "angular_rotation_deg": 8.0,
    "azimuth_rotation_deg": 8.0,
    "roll_deg": 8.0,
    "scale": 0.10,
    "radial_scale": 0.10,
    "radial_shift": 0.10,
    "altitude_offset_deg": 8.0,
}


def _sample_single_transform(
    rng: np.random.Generator,
    cs: CoordinateSystem,
    transform_ranges: dict[str, list[float]],
) -> dict[str, float]:
    """Sample exactly one transform type, zeroing all others.

    For categorical questions: ensures the direction answer is unambiguous.
    Picks one active key (with nonzero range) at random.
    """
    active_keys = [
        k for k in cs.supported_transforms
        if k in transform_ranges and transform_ranges[k] != [0.0, 0.0]
    ]
    if not active_keys:
        return make_identity_transform(cs.supported_transforms)

    chosen = active_keys[int(rng.integers(0, len(active_keys)))]

    lo, hi = transform_ranges[chosen]
    params = make_identity_transform(cs.supported_transforms)
    params[chosen] = float(rng.uniform(lo, hi))
    return params


def _sample_aligned_transform(
    rng: np.random.Generator,
    cs: CoordinateSystem,
    transform_ranges: dict[str, list[float]],
    rendering_config: RenderingConfig,
) -> dict[str, float]:
    """Sample a small coherent transform for 'aligned' binary samples.

    Produces 0.25-0.6 dot diameters of offset in image space — dots
    always partially overlap with both colors peeking through. Clearly
    "nearly aligned." Picks one random active transform key.
    """
    dot_diameter_native = _dot_diameter_native(rendering_config)
    lo = 0.25 * dot_diameter_native
    hi = 0.6 * dot_diameter_native

    active_keys = [
        k for k in cs.supported_transforms
        if k in transform_ranges and transform_ranges[k] != [0.0, 0.0]
    ]
    if not active_keys:
        active_keys = list(cs.supported_transforms)

    chosen = active_keys[int(rng.integers(0, len(active_keys)))]

    # For angular transforms, convert dot-diameter in native coords to degrees
    if chosen in _ANGULAR_TRANSFORMS:
        # At mid-radius (~0.5), one native unit ≈ image_size/2 pixels,
        # so 1 dot diameter ≈ 2 * dot_diameter_native in degrees
        lo_val = lo * (180.0 / np.pi) * 2
        hi_val = hi * (180.0 / np.pi) * 2
    else:
        lo_val = lo
        hi_val = hi

    sign = 1.0 if rng.random() > 0.5 else -1.0
    magnitude = float(rng.uniform(lo_val, hi_val))

    params = make_identity_transform(cs.supported_transforms)
    params[chosen] = sign * magnitude
    return params


def _dot_diameter_native(rendering_config: RenderingConfig) -> float:
    """Convert dot diameter from pixels to native [0,1] coordinate units."""
    dot_diameter_px = rendering_config.point_radius * 2
    usable_px = rendering_config.image_size * 0.8  # approx with margin=0.1
    return dot_diameter_px / usable_px


_ANGULAR_TRANSFORMS = {
    "rotation_deg", "angular_rotation_deg", "azimuth_rotation_deg",
    "roll_deg", "altitude_offset_deg",
}


def _sample_misaligned_transform(
    rng: np.random.Generator,
    cs: CoordinateSystem,
    transform_ranges: dict[str, list[float]],
    rendering_config: RenderingConfig,
) -> dict[str, float]:
    """Sample a transform enforcing minimum misalignment.

    Resamples (up to 20 tries) until at least one transform parameter
    exceeds the minimum misalignment threshold (calibrated to ~3-4 dot
    diameters in image space). This ensures "no" binary samples are
    clearly visually misaligned with no borderline cases.
    """
    # Scale min-misalignment by actual dot size relative to default
    dot_scale = rendering_config.point_radius / 6.0

    for _ in range(20):
        params = cs.sample_transform(rng, transform_ranges)
        # Check if any parameter exceeds its minimum
        for key, val in params.items():
            min_mag = _MIN_MISALIGNMENT.get(key, 0.10) * dot_scale
            if abs(val) >= min_mag:
                return params
    # Fallback: return whatever we got (shouldn't happen with reasonable ranges)
    return params


class BenchmarkGenerator:
    """Generate (image, sample) pairs from a benchmark spec.

    RNG discipline: a single numpy Generator created from PCG64(seed) is used
    sequentially. The draw order per sample is:
      1. coordinate system selection (if coord_identification)
      2. num_points (randint)
      3. reference points (generate_points)
      4. binary coin flip (if binary question)
      5. transform parameters (sample_transform / jitter RNG draws)
      6. transform B parameters (if comparative, sample_transform)
      7. A/B assignment coin flip (if comparative)

    This ordering is fixed and must not change to preserve determinism.
    """

    def __init__(self, spec: BenchmarkSpec):
        self.spec = spec
        self.coord_system = get_coordinate_system(spec.coordinate_system)

    def generate(self, seed: int | None = None) -> Iterator[tuple[Image.Image, Sample]]:
        """Yield (image, sample_metadata) tuples for all samples × question types."""
        seed = seed if seed is not None else self.spec.seed
        rng = np.random.Generator(np.random.PCG64(seed))

        sample_id = 0
        for _sample_idx in range(self.spec.num_samples):
            for question_config in self.spec.questions:
                result = self._generate_one(rng, sample_id, question_config)
                if result is not None:
                    yield result
                    sample_id += 1

    def _generate_one(
        self,
        rng: np.random.Generator,
        sample_id: int,
        question_config: QuestionConfig,
    ) -> tuple[Image.Image, Sample] | None:
        spec = self.spec
        q_type = question_config.type

        # 1. For coord_identification, cycle through all coordinate systems
        if q_type == "coord_identification":
            cs_name = _ALL_COORD_SYSTEMS[sample_id % len(_ALL_COORD_SYSTEMS)]
            cs = get_coordinate_system(cs_name)
        else:
            cs = self.coord_system

        # 2. Sample number of points
        n_lo, n_hi = spec.points.num_points
        n_points = int(rng.integers(n_lo, n_hi + 1))

        # 3. Generate reference points
        distribution = (
            "coord_uniform" if q_type == "coord_identification"
            else spec.points.distribution
        )
        ref_points = cs.generate_points(
            n_points, rng, distribution, spec.points.margin
        )

        if q_type == "coord_identification":
            return self._gen_coord_id(cs, ref_points, sample_id, spec)

        if q_type == "comparative":
            return self._gen_comparative(
                cs, ref_points, rng, sample_id, spec, question_config
            )

        # Standard single-transform questions (binary, categorical, quantitative)
        # 4. Binary coin flip
        is_aligned = False
        if q_type == "binary":
            is_aligned = bool(rng.random() < 0.5)

        # 5. Sample transform
        if is_aligned:
            # Aligned binary: small coherent transform (0.25-1.0 dot diameters
            # in image space). Both colors visible, clearly the same point
            # constellation with a tiny offset. Never identity (which hides blue).
            transform_params = _sample_aligned_transform(
                rng, cs, spec.transforms, spec.rendering
            )
        elif q_type == "binary":
            # Misaligned binary: enforce minimum magnitude (≥3 dot diameters)
            transform_params = _sample_misaligned_transform(
                rng, cs, spec.transforms, spec.rendering
            )
        elif q_type == "categorical":
            transform_params = _sample_single_transform(rng, cs, spec.transforms)
        else:
            transform_params = cs.sample_transform(rng, spec.transforms)

        # Quantitative only applies to rotation/scale — skip if only translation
        if q_type == "quantitative" and not has_quantitative_transform(transform_params):
            return None

        transformed = cs.apply_transform(ref_points, transform_params)

        # Render
        img = render_sample(
            cs, ref_points, transformed, transform_params,
            spec.rendering, spec.points.margin,
        )

        # Generate question and answer
        if q_type == "binary":
            q_text, answer = generate_binary_question(is_aligned)
        elif q_type == "categorical":
            q_text, answer = generate_categorical_question(cs, transform_params)
        elif q_type == "quantitative":
            result = generate_quantitative_question(cs, transform_params)
            if result is None:
                return None
            q_text, answer = result
        else:
            raise ValueError(f"Unknown question type: {q_type!r}")

        magnitude = cs.transform_magnitude(transform_params)
        direction = (
            cs.classify_direction(transform_params) if q_type == "categorical" else None
        )

        sample = Sample(
            sample_id=sample_id,
            spec_name=spec.name,
            coordinate_system=cs.name,
            question_type=q_type,
            question_text=q_text,
            ground_truth=answer,
            transform_params=transform_params,
            transform_magnitude=magnitude,
            transform_direction=direction,
        )
        return img, sample

    def _gen_coord_id(
        self,
        cs: CoordinateSystem,
        ref_points: np.ndarray,
        sample_id: int,
        spec: BenchmarkSpec,
    ) -> tuple[Image.Image, Sample]:
        """Generate a coordinate-identification sample."""
        identity = make_identity_transform(cs.supported_transforms)
        render_config = spec.rendering.model_copy(update={"show_grid": False})
        img = render_sample(
            cs, ref_points, None, identity, render_config, spec.points.margin
        )
        q_text, answer = generate_coord_identification_question(cs)
        sample = Sample(
            sample_id=sample_id,
            spec_name=spec.name,
            coordinate_system=cs.name,
            question_type="coord_identification",
            question_text=q_text,
            ground_truth=answer,
            transform_params=identity,
            transform_magnitude=0.0,
        )
        return img, sample

    def _gen_comparative(
        self,
        cs: CoordinateSystem,
        ref_points: np.ndarray,
        rng: np.random.Generator,
        sample_id: int,
        spec: BenchmarkSpec,
        question_config: QuestionConfig,
    ) -> tuple[Image.Image, Sample]:
        """Generate a comparative A|B sample with a clear winner.

        Two tiers (50/50 split):
        - Easy: one panel aligned (small jitter), other clearly misaligned
        - Hard: both misaligned, but worse panel has 2-4x larger magnitude

        The "better" panel is randomly assigned to A or B.
        """
        is_easy = bool(rng.random() < 0.5)

        if is_easy:
            # Easy: one aligned, one misaligned
            params_better = _sample_aligned_transform(
                rng, cs, spec.transforms, spec.rendering
            )
            params_worse = _sample_misaligned_transform(
                rng, cs, spec.transforms, spec.rendering
            )
        else:
            # Hard: both misaligned, worse is 2-4x larger
            params_better = _sample_misaligned_transform(
                rng, cs, spec.transforms, spec.rendering
            )
            # Scale up: multiply each nonzero param by 2-4x
            multiplier = float(rng.uniform(2.0, 4.0))
            params_worse = {
                k: v * multiplier for k, v in params_better.items()
            }

        # Randomly assign to A or B
        better_is_a = bool(rng.random() < 0.5)
        if better_is_a:
            params_a, params_b = params_better, params_worse
        else:
            params_a, params_b = params_worse, params_better
        better_label = "A" if better_is_a else "B"

        trans_a = cs.apply_transform(ref_points, params_a)
        trans_b = cs.apply_transform(ref_points, params_b)

        img = render_comparative(
            cs, ref_points, trans_a, trans_b,
            params_a, params_b, spec.rendering, spec.points.margin,
        )

        q_text, answer = generate_comparative_question(
            cs, params_a, params_b, better_label
        )

        mag_better = cs.transform_magnitude(params_better)
        sample = Sample(
            sample_id=sample_id,
            spec_name=spec.name,
            coordinate_system=cs.name,
            question_type="comparative",
            question_text=q_text,
            ground_truth=answer,
            transform_params=params_better,
            transform_magnitude=mag_better,
        )
        return img, sample
