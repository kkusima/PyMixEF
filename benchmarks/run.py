#!/usr/bin/env python3
"""Reduced public benchmark harness producing JSON plus an environment manifest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pymixef
from pymixef.provenance import environment_snapshot


def synthetic_lmm() -> dict[str, object]:
    rng = np.random.default_rng(20260722)
    groups = np.repeat(np.arange(20), 5)
    x = np.tile(np.linspace(0.0, 1.0, 5), 20)
    random_intercept = rng.normal(0.0, 0.8, 20)
    y = 2.0 + 1.5 * x + random_intercept[groups] + rng.normal(0.0, 0.4, x.size)
    started = time.perf_counter()
    result = pymixef.fit(
        "y ~ x + (1 | group)",
        data={"y": y, "x": x, "group": groups},
        method="reml",
    )
    elapsed = time.perf_counter() - started
    return {
        "id": "reduced-synthetic-lmm",
        "elapsed_seconds": elapsed,
        "objective": result.objective,
        "parameters": dict(result.parameters),
        "convergence": result.convergence.to_dict(),
        "n_observations": len(y),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = {
        "schema_version": "1.0.0",
        "environment": environment_snapshot(redact_host=False),
        "benchmarks": [synthetic_lmm()],
    }
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
