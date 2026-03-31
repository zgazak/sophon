"""Pydantic v2 models for benchmark specs and samples."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VALID_COORDINATE_SYSTEMS = ["linear_1d", "cartesian_2d", "polar", "spherical", "all"]


class RenderingConfig(BaseModel):
    """Visual rendering parameters."""

    model_config = ConfigDict(extra="forbid")

    image_size: int = 512
    background_color: str = "#F8F8F8"
    color_a: str = "#2563EB"  # Blue (reference)
    color_b: str = "#DC2626"  # Red (transformed)
    point_radius: float = 6.0
    show_grid: bool = True
    grid_color: str = "#E0E0E0"
    grid_alpha: float = 0.5


class PointsConfig(BaseModel):
    """Point generation parameters."""

    model_config = ConfigDict(extra="forbid")

    num_points: tuple[int, int]  # [min, max] uniform random int
    distribution: Literal["uniform", "clustered", "grid"] = "uniform"
    margin: float = 0.1


class QuestionConfig(BaseModel):
    """Configuration for a single question type."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["binary", "categorical", "comparative", "quantitative", "coord_identification"]
    alignment_threshold: float | None = None  # Only used for binary
    vocab_mode: Literal["restricted", "full"] = "restricted"  # Only used for categorical
    range_hint: bool = False  # Only used for quantitative


class BenchmarkSpec(BaseModel):
    """Top-level benchmark specification loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    coordinate_system: Literal["linear_1d", "cartesian_2d", "polar", "spherical", "all"]
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    points: PointsConfig
    transforms: dict[str, list[float]]  # field_name -> [min, max]
    questions: list[QuestionConfig]
    num_samples: int = 1000
    seed: int = 42
    output_mode: Literal["image", "text_description", "both"] = "image"
    output_format: Literal["standard", "conversation"] = "standard"
    system_prompt: str | None = None


class SuiteSpec(BaseModel):
    """Meta-spec referencing multiple benchmark specs."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str = ""
    specs: list[str]  # Filenames of individual specs
    seed: int = 42


class Sample(BaseModel):
    """One generated (image, question, answer) triple with metadata."""

    model_config = ConfigDict(extra="forbid")

    sample_id: int | str
    spec_name: str
    coordinate_system: str
    question_type: str
    question_text: str
    ground_truth: str | float  # str for categorical/binary/comparative, float for quantitative
    image_path: str | None = None  # Set when saved to disk
    transform_params: dict[str, float]
    transform_magnitude: float
    transform_direction: str | None = None  # For categorical questions
    valid_directions: list[str] | None = None  # For categorical: the vocabulary used in the question
    text_description: str | None = None  # Populated when output_mode != "image"
    num_points: int | None = None
    show_grid: bool | None = None
