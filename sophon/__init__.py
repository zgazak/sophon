"""Sophon: Generative benchmark for probing spatial reasoning in VLMs."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth is pyproject.toml; read it back from the installed
    # package metadata so __version__ can never drift from the released version.
    __version__ = version("sophon-bench")
except PackageNotFoundError:  # pragma: no cover - running from a bare source tree
    __version__ = "unknown"
