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

_ALL_COORD_SYSTEMS = ["linear_1d", "cartesian_2d", "polar", "spherical"]
_COORD_ID_SYSTEMS = ["linear_1d", "cartesian_2d", "polar"]

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
        k
        for k in cs.supported_transforms
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
        k
        for k in cs.supported_transforms
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
    "rotation_deg",
    "angular_rotation_deg",
    "azimuth_rotation_deg",
    "roll_deg",
    "altitude_offset_deg",
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


def _build_text_description(
    cs: CoordinateSystem,
    ref_points: np.ndarray,
    transformed_points: np.ndarray | None,
) -> str:
    """Build a text description of point positions in native coordinate space."""
    parts = [
        "Blue points (reference set):",
        cs.describe_points(ref_points),
    ]
    if transformed_points is not None:
        parts.extend(
            [
                "",
                "Red points (transformed set):",
                cs.describe_points(transformed_points),
            ]
        )
    parts.extend(
        [
            "",
            f"The coordinate system is {cs.name}.",
        ]
    )
    return "\n".join(parts)


def _filter_transforms_for_cs(
    cs: CoordinateSystem, all_transforms: dict[str, list[float]]
) -> dict[str, list[float]]:
    """Filter spec transforms to only those supported by the given coordinate system."""
    return {k: v for k, v in all_transforms.items() if k in cs.supported_transforms}


class BenchmarkGenerator:
    """Generate (image, sample) pairs from a benchmark spec.

    RNG discipline: a single numpy Generator created from PCG64(seed) is used
    sequentially. The draw order per sample is:
      1. coordinate system selection (if coord_identification or all)
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
        self._is_all = spec.coordinate_system == "all"
        if not self._is_all:
            self.coord_system = get_coordinate_system(spec.coordinate_system)
        else:
            self.coord_system = None  # resolved per-sample

    def generate(self, seed: int | None = None) -> Iterator[tuple[Image.Image | None, Sample]]:
        """Yield (image_or_None, sample_metadata) tuples for all samples x question types.

        When output_mode is "text_description", image is None.
        When output_mode is "image" or "both", image is a PIL Image.
        """
        seed = seed if seed is not None else self.spec.seed
        self._active_seed = seed
        rng = np.random.Generator(np.random.PCG64(seed))

        if self._is_all:
            yield from self._generate_all_coords(rng)
        else:
            yield from self._generate_single_coord(rng)

    def _make_sample_id(self, index: int) -> str:
        """Build a globally unique sample ID: {spec_name}_seed{seed}_{zero_padded_index}."""
        return f"{self.spec.name}_seed{self._active_seed}_{index:04d}"

    def _generate_single_coord(
        self, rng: np.random.Generator
    ) -> Iterator[tuple[Image.Image | None, Sample]]:
        sample_idx = 0
        for _sample_idx in range(self.spec.num_samples):
            for question_config in self.spec.questions:
                sid = self._make_sample_id(sample_idx)
                result = self._generate_one(rng, sid, question_config)
                if result is not None:
                    yield result
                    sample_idx += 1

    def _generate_all_coords(
        self, rng: np.random.Generator
    ) -> Iterator[tuple[Image.Image | None, Sample]]:
        """Generate balanced samples across all coordinate systems."""
        per_cs = self.spec.num_samples // len(_ALL_COORD_SYSTEMS)
        remainder = self.spec.num_samples % len(_ALL_COORD_SYSTEMS)

        sample_idx = 0
        for cs_idx, cs_name in enumerate(_ALL_COORD_SYSTEMS):
            cs = get_coordinate_system(cs_name)
            n_samples = per_cs + (1 if cs_idx < remainder else 0)
            transforms = _filter_transforms_for_cs(cs, self.spec.transforms)

            for _sample_idx in range(n_samples):
                for question_config in self.spec.questions:
                    sid = self._make_sample_id(sample_idx)
                    result = self._generate_one(
                        rng,
                        sid,
                        question_config,
                        cs_override=cs,
                        transforms_override=transforms,
                    )
                    if result is not None:
                        yield result
                        sample_idx += 1

    def _generate_one(
        self,
        rng: np.random.Generator,
        sample_id: int | str,
        question_config: QuestionConfig,
        cs_override: CoordinateSystem | None = None,
        transforms_override: dict[str, list[float]] | None = None,
    ) -> tuple[Image.Image | None, Sample] | None:
        spec = self.spec
        q_type = question_config.type
        transforms = transforms_override if transforms_override is not None else spec.transforms

        # 1. For coord_identification, cycle through coordinate systems
        #    Use the numeric index from the ID for cycling (works for both
        #    legacy int IDs and new string IDs like "spec_seed42_0003").
        if q_type == "coord_identification":
            if isinstance(sample_id, str):
                idx = int(sample_id.rsplit("_", 1)[-1])
            else:
                idx = sample_id
            cs_name = _COORD_ID_SYSTEMS[idx % len(_COORD_ID_SYSTEMS)]
            cs = get_coordinate_system(cs_name)
        elif cs_override is not None:
            cs = cs_override
        else:
            cs = self.coord_system

        # 2. Sample number of points
        n_lo, n_hi = spec.points.num_points
        n_points = int(rng.integers(n_lo, n_hi + 1))

        # 3. Generate reference points
        distribution = (
            "coord_uniform" if q_type == "coord_identification" else spec.points.distribution
        )
        ref_points = cs.generate_points(n_points, rng, distribution, spec.points.margin)

        if q_type == "coord_identification":
            return self._gen_coord_id(cs, ref_points, n_points, sample_id, spec)

        if q_type == "comparative":
            return self._gen_comparative(
                cs,
                ref_points,
                n_points,
                rng,
                sample_id,
                spec,
                question_config,
                transforms,
            )

        # Standard single-transform questions (binary, categorical, quantitative)
        # 4. Binary coin flip
        is_aligned = False
        if q_type == "binary":
            is_aligned = bool(rng.random() < 0.5)

        # 5. Sample transform
        if is_aligned:
            transform_params = _sample_aligned_transform(rng, cs, transforms, spec.rendering)
        elif q_type == "binary":
            transform_params = _sample_misaligned_transform(rng, cs, transforms, spec.rendering)
        elif q_type == "categorical":
            transform_params = _sample_single_transform(rng, cs, transforms)
        else:
            transform_params = cs.sample_transform(rng, transforms)

        # Quantitative only applies to rotation/scale — skip if only translation
        if q_type == "quantitative" and not has_quantitative_transform(transform_params):
            return None

        transformed = cs.apply_transform(ref_points, transform_params)

        # Render image if needed
        want_image = spec.output_mode in ("image", "both")
        want_text = spec.output_mode in ("text_description", "both")

        img = None
        if want_image:
            img = render_sample(
                cs,
                ref_points,
                transformed,
                transform_params,
                spec.rendering,
                spec.points.margin,
            )

        text_desc = None
        if want_text:
            text_desc = _build_text_description(cs, ref_points, transformed)

        # Generate question and answer
        valid_dirs: list[str] | None = None
        if q_type == "binary":
            q_text, answer = generate_binary_question(is_aligned)
        elif q_type == "categorical":
            vocab_mode = question_config.vocab_mode
            q_text, answer, valid_dirs = generate_categorical_question(
                cs, transform_params, vocab_mode=vocab_mode,
            )
        elif q_type == "quantitative":
            result = generate_quantitative_question(
                cs, transform_params,
                range_hint=question_config.range_hint,
                transform_ranges=transforms,
            )
            if result is None:
                return None
            q_text, answer = result
        else:
            raise ValueError(f"Unknown question type: {q_type!r}")

        magnitude = cs.transform_magnitude(transform_params)
        direction = cs.classify_direction(transform_params) if q_type == "categorical" else None

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
            valid_directions=valid_dirs,
            text_description=text_desc,
            num_points=n_points,
            show_grid=spec.rendering.show_grid,
        )
        return img, sample

    def _gen_coord_id(
        self,
        cs: CoordinateSystem,
        ref_points: np.ndarray,
        n_points: int,
        sample_id: int | str,
        spec: BenchmarkSpec,
    ) -> tuple[Image.Image | None, Sample]:
        """Generate a coordinate-identification sample."""
        identity = make_identity_transform(cs.supported_transforms)
        render_config = spec.rendering.model_copy(update={"show_grid": False})

        want_image = spec.output_mode in ("image", "both")
        want_text = spec.output_mode in ("text_description", "both")

        img = None
        if want_image:
            img = render_sample(cs, ref_points, None, identity, render_config, spec.points.margin)

        text_desc = None
        if want_text:
            text_desc = _build_text_description(cs, ref_points, None)

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
            text_description=text_desc,
            num_points=n_points,
            show_grid=False,
        )
        return img, sample

    def _gen_comparative(
        self,
        cs: CoordinateSystem,
        ref_points: np.ndarray,
        n_points: int,
        rng: np.random.Generator,
        sample_id: int,
        spec: BenchmarkSpec,
        question_config: QuestionConfig,
        transforms: dict[str, list[float]],
    ) -> tuple[Image.Image | None, Sample]:
        """Generate a comparative A|B sample with a clear winner.

        Two tiers (50/50 split):
        - Easy: one panel aligned (small jitter), other clearly misaligned
        - Hard: both misaligned, but worse panel has 2-4x larger magnitude

        The "better" panel is randomly assigned to A or B.
        """
        is_easy = bool(rng.random() < 0.5)

        if is_easy:
            params_better = _sample_aligned_transform(rng, cs, transforms, spec.rendering)
            params_worse = _sample_misaligned_transform(rng, cs, transforms, spec.rendering)
        else:
            params_better = _sample_misaligned_transform(rng, cs, transforms, spec.rendering)
            multiplier = float(rng.uniform(2.0, 4.0))
            params_worse = {k: v * multiplier for k, v in params_better.items()}

        # Randomly assign to A or B
        better_is_a = bool(rng.random() < 0.5)
        if better_is_a:
            params_a, params_b = params_better, params_worse
        else:
            params_a, params_b = params_worse, params_better
        better_label = "A" if better_is_a else "B"

        trans_a = cs.apply_transform(ref_points, params_a)
        trans_b = cs.apply_transform(ref_points, params_b)

        want_image = spec.output_mode in ("image", "both")
        want_text = spec.output_mode in ("text_description", "both")

        img = None
        if want_image:
            img = render_comparative(
                cs,
                ref_points,
                trans_a,
                trans_b,
                params_a,
                params_b,
                spec.rendering,
                spec.points.margin,
            )

        text_desc = None
        if want_text:
            text_a = _build_text_description(cs, ref_points, trans_a)
            text_b = _build_text_description(cs, ref_points, trans_b)
            text_desc = f"Image A:\n{text_a}\n\nImage B:\n{text_b}"

        q_text, answer = generate_comparative_question(cs, params_a, params_b, better_label)

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
            text_description=text_desc,
            num_points=n_points,
            show_grid=spec.rendering.show_grid,
        )
        return img, sample
