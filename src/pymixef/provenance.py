"""Run provenance, environment capture, and deterministic fingerprints."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ._contracts import ReproducibilityClass
from ._serialization import stable_hash
from ._version import __version__

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_snapshot(*, redact_host: bool = True) -> dict[str, Any]:
    """Capture calculation-relevant environment metadata without telemetry."""

    packages = {
        name: version
        for name in (
            "pymixef",
            "numpy",
            "scipy",
            "pandas",
            "polars",
            "pyarrow",
            "xarray",
        )
        if (version := _package_version(name)) is not None
    }
    numpy_config: dict[str, Any] = {}
    try:
        config = np.__config__.show(mode="dicts")
        if isinstance(config, dict):
            numpy_config = config
    except (TypeError, AttributeError):
        numpy_config = {"capture": "unavailable"}
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": "<redacted>" if redact_host else sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": "<redacted>" if redact_host else socket.gethostname(),
        },
        "packages": packages,
        "thread_settings": {key: os.environ.get(key, "unset") for key in _THREAD_VARIABLES},
        "numpy_build": numpy_config,
    }


def fingerprint_data(data: Any) -> str:
    """Create a deterministic data fingerprint for common columnar inputs."""

    if hasattr(data, "to_dict"):
        try:
            data = data.to_dict(orient="list")
        except TypeError:
            data = data.to_dict()
    if isinstance(data, Mapping):
        normalized = {
            str(name): np.asarray(values)
            for name, values in sorted(data.items(), key=lambda item: str(item[0]))
        }
        lengths = {array.shape[0] for array in normalized.values() if array.ndim}
        if len(lengths) > 1:
            raise ValueError("Columns have inconsistent lengths and cannot be fingerprinted.")
        return stable_hash(normalized)
    if isinstance(data, np.ndarray):
        return stable_hash(data)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return stable_hash(list(data))
    raise TypeError(f"Unsupported data container for fingerprinting: {type(data).__name__}.")


def fingerprint_model_ir(model_ir: Any) -> str:
    """Hash the exact semantic payload of a model IR object or document.

    Archived IR documents are hashed as stored rather than migrated first. This
    keeps legacy documents verifiable against the manifest produced when they
    were captured while ensuring a :class:`ModelIR` object and its ``to_dict()``
    payload have identical fingerprints.
    """

    if isinstance(model_ir, Mapping):
        payload = model_ir
    else:
        to_dict = getattr(model_ir, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(
                "Model IR fingerprinting requires a mapping or an object with to_dict()."
            )
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError("Model IR to_dict() must return a mapping.")
    return stable_hash(payload)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Complete, serializable description of a PyMixEF execution."""

    manifest_schema_version: str
    package_version: str
    created_at_utc: str
    model_ir_hash: str
    data_hash: str
    engine: str
    method: str
    settings: Mapping[str, Any]
    seeds: Mapping[str, int]
    reproducibility_class: str
    environment: Mapping[str, Any]
    elapsed_seconds: float | None = None
    output_hashes: Mapping[str, str] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)
    convergence: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def capture(
        cls,
        *,
        model_ir: Any,
        data: Any,
        engine: str,
        method: str,
        settings: Mapping[str, Any] | None = None,
        seeds: Mapping[str, int] | None = None,
        reproducibility_class: ReproducibilityClass | str = (
            ReproducibilityClass.DETERMINISTIC_TOLERANCE
        ),
        elapsed_seconds: float | None = None,
        source: Mapping[str, Any] | None = None,
        convergence: Mapping[str, Any] | None = None,
        warnings: Sequence[Mapping[str, Any]] = (),
    ) -> RunManifest:
        value = (
            reproducibility_class.value
            if isinstance(reproducibility_class, ReproducibilityClass)
            else str(reproducibility_class)
        )
        return cls(
            manifest_schema_version="1.0.0",
            package_version=__version__,
            created_at_utc=datetime.now(UTC).isoformat(),
            model_ir_hash=fingerprint_model_ir(model_ir),
            data_hash=fingerprint_data(data),
            engine=engine,
            method=method,
            settings=dict(settings or {}),
            seeds=dict(seeds or {}),
            reproducibility_class=value,
            environment=environment_snapshot(),
            elapsed_seconds=elapsed_seconds,
            source={
                "build_id": os.environ.get("PYMIXEF_BUILD_ID", "unrecorded"),
                "git_commit": os.environ.get("GITHUB_SHA", "unrecorded"),
                **dict(source or {}),
            },
            convergence=dict(convergence or {}),
            warnings=tuple(dict(item) for item in warnings),
        )

    def with_outputs(self, outputs: Mapping[str, Any]) -> RunManifest:
        """Return an immutable copy carrying hashes of named result components."""

        return RunManifest(
            manifest_schema_version=self.manifest_schema_version,
            package_version=self.package_version,
            created_at_utc=self.created_at_utc,
            model_ir_hash=self.model_ir_hash,
            data_hash=self.data_hash,
            engine=self.engine,
            method=self.method,
            settings=self.settings,
            seeds=self.seeds,
            reproducibility_class=self.reproducibility_class,
            environment=self.environment,
            elapsed_seconds=self.elapsed_seconds,
            output_hashes={name: stable_hash(value) for name, value in outputs.items()},
            source=self.source,
            convergence=self.convergence,
            warnings=self.warnings,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "package_version": self.package_version,
            "created_at_utc": self.created_at_utc,
            "model_ir_hash": self.model_ir_hash,
            "data_hash": self.data_hash,
            "engine": self.engine,
            "method": self.method,
            "settings": dict(self.settings),
            "seeds": dict(self.seeds),
            "reproducibility_class": self.reproducibility_class,
            "environment": dict(self.environment),
            "elapsed_seconds": self.elapsed_seconds,
            "output_hashes": dict(self.output_hashes),
            "source": dict(self.source),
            "convergence": dict(self.convergence),
            "warnings": [dict(item) for item in self.warnings],
        }


class RunTimer:
    """Monotonic context timer used when constructing run manifests."""

    def __enter__(self) -> RunTimer:
        self._start = time.perf_counter()
        self.elapsed_seconds = None
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start
