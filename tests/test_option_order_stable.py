"""Presented option order must be reproducible across processes.

`generate_categorical_question(vocab_order="random")` shuffles the answer options
to defeat position bias. The shuffle is seeded per sample, and the seed used to
come from builtin `hash()`, which Python salts per process via PYTHONHASHSEED.
The comment called it deterministic; it was deterministic only within one run.

The consequence was not a wrong answer -- ground truth never depended on it --
but it meant no categorical dataset could be regenerated identically, which is
why evaluation batteries have to be cached rather than rebuilt. This pins the
fix so the non-determinism cannot come back.
"""

import subprocess
import sys

import yaml

from sophon.generator import BenchmarkGenerator, _stable_seed
from sophon.schema import BenchmarkSpec

_SNIPPET = """
import json, yaml, pathlib
from sophon.generator import BenchmarkGenerator
from sophon.schema import BenchmarkSpec
spec = BenchmarkSpec(**yaml.safe_load(
    pathlib.Path("sophon/specs/cartesian_rotation.yaml").read_text()))
spec.num_samples = 12
for q in spec.questions:
    if q.type == "categorical":
        q.vocab_order = "random"
print(json.dumps([s.question_text
                  for _, s in BenchmarkGenerator(spec).generate(seed=5)]))
"""


def test_stable_seed_is_process_independent():
    """The helper itself, under two different hash salts."""
    outs = []
    for salt in ("0", "12345"):
        r = subprocess.run(
            [sys.executable, "-c",
             "from sophon.generator import _stable_seed; print(_stable_seed('abc'))"],
            capture_output=True, text=True, check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": salt, "PYTHONPATH": "."},
        )
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], f"seed varies with PYTHONHASHSEED: {outs}"
    assert _stable_seed("abc") == int(outs[0])


def test_question_option_order_reproduces_across_processes():
    """The property that actually matters: identical question text, two runs."""
    outs = []
    for salt in ("0", "999"):
        r = subprocess.run(
            [sys.executable, "-c", _SNIPPET], capture_output=True, text=True,
            check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": salt, "PYTHONPATH": "."},
        )
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], "option order still depends on PYTHONHASHSEED"


def test_shuffle_is_still_actually_random():
    """Stability must not have collapsed into a constant order."""
    spec = BenchmarkSpec(**yaml.safe_load(
        open("sophon/specs/cartesian_rotation.yaml").read()))
    spec.num_samples = 40
    for q in spec.questions:
        if q.type == "categorical":
            q.vocab_order = "random"
    orders = {s.question_text.split("one of:")[-1].strip()
              for _, s in BenchmarkGenerator(spec).generate(seed=5)}
    assert len(orders) > 1, f"option order is constant: {orders}"
