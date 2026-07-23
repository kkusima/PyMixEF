from __future__ import annotations

from pathlib import Path

import numpy as np

from pymixef.inference import bootstrap
from pymixef.random import RandomStreamManager

from .test_results import _result


def test_named_streams_are_order_independent() -> None:
    manager = RandomStreamManager(123)
    first = manager.generator("simulation", replicate=2).normal(size=5)
    manager.generator("other", replicate=99).normal(size=100)
    second = manager.generator("simulation", replicate=2).normal(size=5)
    np.testing.assert_array_equal(first, second)


def test_restartable_bootstrap_failure_accounting(tmp_path: Path) -> None:
    def fit_function(data):
        result = _result()
        value = float(np.mean(data["y"]))
        result.parameters = {
            "beta[Intercept]": value,
            "residual_sd": result.parameters["residual_sd"],
        }
        return result

    checkpoint = tmp_path / "checkpoint.json"
    result = bootstrap(
        fit_function,
        {"y": np.arange(8.0), "cluster": np.repeat([0, 1], 4)},
        n_replicates=5,
        seed=42,
        cluster="cluster",
        checkpoint=checkpoint,
    )
    assert result.successful_replicates == 5
    assert result.failed_replicates == 0
    resumed = bootstrap(
        fit_function,
        {"y": np.arange(8.0), "cluster": np.repeat([0, 1], 4)},
        n_replicates=5,
        seed=42,
        cluster="cluster",
        checkpoint=checkpoint,
    )
    np.testing.assert_array_equal(
        result.draws.columns["beta[Intercept]"],
        resumed.draws.columns["beta[Intercept]"],
    )
