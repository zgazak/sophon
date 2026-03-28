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


def _load_specs_from_dir(specs_dir: str) -> list[BenchmarkSpec]:
    """Load all YAML specs from a directory."""
    d = Path(specs_dir)
    if not d.is_dir():
        raise click.BadParameter(f"Not a directory: {specs_dir}")
    specs = []
    for yaml_path in sorted(d.glob("*.yaml")):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if data and "coordinate_system" in data:
            specs.append(BenchmarkSpec(**data))
    if not specs:
        raise click.BadParameter(f"No valid spec files found in: {specs_dir}")
    return specs


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
            sample_with_path = sample.model_copy(update={"sample_id": global_id})
            if img is not None:
                img_path = images_dir / f"{global_id:06d}.png"
                img.save(img_path)
                sample_with_path = sample_with_path.model_copy(update={"image_path": str(img_path)})
            all_metadata.append(sample_with_path.model_dump())
            global_id += 1

    # Write output based on format
    if specs and specs[0].output_format == "conversation":
        _write_conversation_format(out, all_metadata)
    else:
        _write_standard_format(out, all_metadata)

    click.echo(f"Generated {global_id} samples to {out}")


def _write_standard_format(out: Path, all_metadata: list[dict]) -> None:
    """Write standard metadata.jsonl and ground_truth.json."""
    metadata_path = out / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for m in all_metadata:
            f.write(json.dumps(m, default=str) + "\n")

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


def _write_conversation_format(out: Path, all_metadata: list[dict]) -> None:
    """Write ShareGPT-style conversation JSONL for fine-tuning."""
    conversations_path = out / "conversations.jsonl"
    with open(conversations_path, "w") as f:
        for m in all_metadata:
            # Build user content
            user_content = "<image>\n" + m["question_text"]
            if m.get("text_description"):
                user_content = m["text_description"] + "\n\n" + m["question_text"]

            # Terse assistant response
            gt = m["ground_truth"]
            if m["question_type"] == "binary":
                assistant_content = "Yes" if str(gt).lower() == "yes" else "No"
            elif m["question_type"] == "categorical":
                assistant_content = str(gt).capitalize()
            elif m["question_type"] == "comparative":
                assistant_content = str(gt)
            elif m["question_type"] == "quantitative":
                if isinstance(gt, float):
                    assistant_content = f"{gt:.1f} degrees"
                else:
                    assistant_content = str(gt)
            elif m["question_type"] == "coord_identification":
                assistant_content = str(gt)
            else:
                assistant_content = str(gt)

            spec_name = m.get("spec_name", "sample")
            sample_id = m.get("sample_id", 0)
            entry = {
                "id": f"{spec_name}_sample_{sample_id:04d}",
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
            }
            if m.get("image_path"):
                entry["image"] = m["image_path"]
            f.write(json.dumps(entry) + "\n")

    # Also write standard format alongside
    _write_standard_format(out, all_metadata)


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
        if img is not None:
            preview_img = render_preview(
                img,
                sample.question_text,
                sample.ground_truth,
                sample.sample_id,
                sample.question_type,
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


EVALUATE_EPILOG = """
Authentication:
  API keys are read from environment variables (never CLI args).

  For frontier models via OpenRouter (recommended):
    export OPENROUTER_API_KEY=sk-or-...
    sophon evaluate --api-base https://openrouter.ai/api/v1 --model-name openai/gpt-4o ...

  For local models via vllm (no key needed):
    sophon evaluate --api-base http://localhost:8000/v1 --model-name llava-v1.6-7b ...

  For direct OpenAI API:
    export OPENAI_API_KEY=sk-...
    sophon evaluate --model-name gpt-4o ...
"""


@cli.command(epilog=EVALUATE_EPILOG)
@click.option("--spec", default=None, help="Path to YAML spec file or built-in spec name")
@click.option(
    "--specs-dir",
    default=None,
    type=click.Path(),
    help="Directory of YAML specs to evaluate (alternative to --spec)",
)
@click.option("--seed", type=int, default=None, help="Override seed from spec (single seed)")
@click.option(
    "--seeds", default=None, help="Comma-separated seeds for multiple runs (e.g. 42,43,44)"
)
@click.option(
    "--backend",
    default="openai",
    type=click.Choice(["openai"]),
    help="VLM backend (currently only openai-compatible)",
)
@click.option("--model-name", required=True, help="Model name (e.g. gpt-4o, llava-v1.6-7b)")
@click.option("--api-base", default=None, help="API base URL")
@click.option("--system-prompt", default=None, help="System prompt to prepend to questions")
@click.option("--output-dir", required=True, type=click.Path(), help="Output directory")
@click.option("--num-samples", type=int, default=None, help="Override num_samples from spec")
@click.option("--max-rpm", type=int, default=None, help="Max requests per minute (rate limit)")
def evaluate(
    spec,
    specs_dir,
    seed,
    seeds,
    backend,
    model_name,
    api_base,
    system_prompt,
    output_dir,
    num_samples,
    max_rpm,
):
    """Evaluate a VLM against Sophon specs.

    Generates samples from specs and queries the VLM for each, producing
    per-sample predictions (JSONL) and aggregated results (JSON).
    """
    from sophon.coordinates import get_coordinate_system
    from sophon.evaluator import (
        aggregate_results,
        create_openai_vlm_fn,
        evaluate_samples,
    )
    from sophon.generator import BenchmarkGenerator

    # Resolve specs
    if spec and specs_dir:
        raise click.UsageError("Use --spec or --specs-dir, not both")
    if not spec and not specs_dir:
        raise click.UsageError("Provide --spec or --specs-dir")

    if specs_dir:
        all_specs = _load_specs_from_dir(specs_dir)
    else:
        all_specs = _load_suite_specs(spec)

    # Resolve seeds
    seed_list: list[int] = []
    if seeds:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    elif seed is not None:
        seed_list = [seed]
    else:
        seed_list = []  # Will use spec default

    # Create VLM function
    vlm_fn = create_openai_vlm_fn(
        model_name=model_name,
        api_base=api_base,
        system_prompt=system_prompt,
        max_requests_per_minute=max_rpm,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build direction vocabularies for parsing
    dir_vocabs = {}
    for cs_name in ["linear_1d", "cartesian_2d", "polar", "spherical"]:
        cs = get_coordinate_system(cs_name)
        dir_vocabs[cs_name] = cs.direction_vocabulary

    for bench_spec in all_specs:
        if num_samples is not None:
            bench_spec = bench_spec.model_copy(update={"num_samples": num_samples})

        effective_seeds = seed_list if seed_list else [bench_spec.seed]

        for eval_seed in effective_seeds:
            click.echo(
                f"Evaluating {bench_spec.name} (seed={eval_seed}, "
                f"n={bench_spec.num_samples}) with {model_name}..."
            )

            gen = BenchmarkGenerator(bench_spec)

            # Collect samples
            samples = []
            for img, sample in gen.generate(seed=eval_seed):
                samples.append((img, sample.model_dump()))

            # Run evaluation
            predictions = evaluate_samples(
                vlm_fn,
                samples,
                direction_vocabularies=dir_vocabs,
            )

            # Determine output file names
            seed_suffix = f"_seed{eval_seed}" if len(effective_seeds) > 1 else ""
            pred_path = out / f"{bench_spec.name}{seed_suffix}_predictions.jsonl"
            results_path = out / f"{bench_spec.name}{seed_suffix}_results.json"

            # Write predictions
            with open(pred_path, "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred, default=str) + "\n")

            # Write aggregated results
            results = aggregate_results(
                predictions,
                spec_name=bench_spec.name,
                model_name=model_name,
                seed=eval_seed,
                num_samples=len(predictions),
            )
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            click.echo(f"  Predictions: {pred_path}")
            click.echo(f"  Results: {results_path}")

    click.echo("Evaluation complete.")


@cli.command()
@click.option(
    "--predictions",
    required=True,
    type=click.Path(exists=True),
    help="Path to predictions JSONL file",
)
@click.option(
    "--ground-truth",
    required=True,
    type=click.Path(exists=True),
    help="Path to ground truth JSON file",
)
@click.option("--output", default=None, type=click.Path(), help="Output results JSON path")
def score(predictions, ground_truth, output):
    """Score existing predictions against ground truth."""
    from sophon.evaluator import score_predictions

    results = score_predictions(Path(predictions), Path(ground_truth))

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        click.echo(f"Results written to {output}")
    else:
        click.echo(json.dumps(results, indent=2, default=str))
