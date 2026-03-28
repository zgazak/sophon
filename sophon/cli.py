"""Sophon CLI: generate, preview, evaluate, score."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

import click
import yaml

from sophon.schema import BenchmarkSpec, SuiteSpec


def _resolve_spec_path(spec_path: str) -> Path:
    """Resolve a spec path, supporting built-in spec names.

    Supports:
      - Direct file path: /path/to/spec.yaml
      - Built-in name: cartesian_rotation (looks up sophon/specs/{name}.yaml)
    """
    p = Path(spec_path)
    if p.exists():
        return p
    # Try as a built-in spec name
    name = spec_path.removesuffix(".yaml")
    specs_dir = importlib.resources.files("sophon.specs")
    builtin = specs_dir / f"{name}.yaml"
    if builtin.is_file():
        return Path(str(builtin))
    raise click.BadParameter(f"Spec not found: {spec_path}")


def _load_spec(spec_path: str) -> BenchmarkSpec:
    """Load and validate a BenchmarkSpec from a YAML file."""
    path = _resolve_spec_path(spec_path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return BenchmarkSpec(**data)


def _load_suite_specs(spec_path: str) -> list[BenchmarkSpec]:
    """Load a suite YAML and return all referenced BenchmarkSpecs."""
    path = _resolve_spec_path(spec_path)
    with open(path) as f:
        data = yaml.safe_load(f)

    # Check if this is a suite (has 'specs' key) or a single spec
    if "specs" in data and "coordinate_system" not in data:
        suite = SuiteSpec(**data)
        specs = []
        spec_dir = path.parent
        for ref in suite.specs:
            ref_path = spec_dir / ref
            with open(ref_path) as f:
                spec_data = yaml.safe_load(f)
            specs.append(BenchmarkSpec(**spec_data))
        return specs
    else:
        return [BenchmarkSpec(**data)]


@click.group()
@click.version_option(package_name="sophon-bench")
def cli():
    """Sophon: Generative benchmark for spatial reasoning in VLMs."""
    pass


@cli.command()
@click.option("--spec", required=True, help="Path to YAML spec file or built-in spec name")
@click.option("--seed", type=int, default=None, help="Override seed from spec")
@click.option("--output-dir", required=True, type=click.Path(), help="Output directory")
@click.option("--num-samples", type=int, default=None, help="Override num_samples from spec")
def generate(spec, seed, output_dir, num_samples):
    """Generate a full dataset to disk."""
    from sophon.generator import BenchmarkGenerator

    specs = _load_suite_specs(spec)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / "images"
    images_dir.mkdir(exist_ok=True)

    all_metadata = []
    global_id = 0

    for bench_spec in specs:
        if num_samples is not None:
            bench_spec = bench_spec.model_copy(update={"num_samples": num_samples})

        gen = BenchmarkGenerator(bench_spec)
        click.echo(f"Generating {bench_spec.name} ({bench_spec.num_samples} samples)...")

        for img, sample in gen.generate(seed=seed):
            img_path = images_dir / f"{global_id:06d}.png"
            img.save(img_path)
            sample_with_path = sample.model_copy(
                update={"sample_id": global_id, "image_path": str(img_path)}
            )
            all_metadata.append(sample_with_path.model_dump())
            global_id += 1

    # Write metadata
    metadata_path = out / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for m in all_metadata:
            f.write(json.dumps(m, default=str) + "\n")

    # Write ground truth
    gt_path = out / "ground_truth.json"
    gt = {
        m["sample_id"]: {
            "question_type": m["question_type"],
            "ground_truth": m["ground_truth"],
            "coordinate_system": m["coordinate_system"],
        }
        for m in all_metadata
    }
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2, default=str)

    click.echo(f"Generated {global_id} samples to {out}")


@cli.command()
@click.option("--spec", required=True, help="Path to YAML spec file or built-in spec name")
@click.option("--num-samples", default=5, type=int, help="Number of samples to preview")
@click.option("--seed", type=int, default=None, help="Override seed from spec")
@click.option("--output-dir", default="preview/", type=click.Path(), help="Output directory")
def preview(spec, num_samples, seed, output_dir):
    """Quick visual inspection of a few samples with question/answer overlaid."""
    from sophon.generator import BenchmarkGenerator
    from sophon.renderer import render_preview

    bench_spec = _load_spec(spec)
    bench_spec = bench_spec.model_copy(update={"num_samples": num_samples})

    gen = BenchmarkGenerator(bench_spec)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img, sample in gen.generate(seed=seed):
        preview_img = render_preview(
            img, sample.question_text, sample.ground_truth,
            sample.sample_id, sample.question_type,
        )
        filename = f"{count:03d}_{sample.question_type}.png"
        preview_img.save(out / filename)
        click.echo(
            f"  [{count}] {sample.question_type}: "
            f"Q={sample.question_text[:60]}... "
            f"A={sample.ground_truth}"
        )
        count += 1

    click.echo(f"Saved {count} preview images to {out}")


@cli.command()
@click.option("--predictions", required=True, type=click.Path(exists=True))
@click.option("--ground-truth", required=True, type=click.Path(exists=True))
@click.option("--output", default=None, type=click.Path(), help="Output results JSON path")
def score(predictions, ground_truth, output):
    """Score existing predictions against ground truth."""
    click.echo("Scoring not yet implemented. Coming in v0.2.")
