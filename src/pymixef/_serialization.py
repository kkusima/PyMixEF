"""Deterministic, non-pickle serialization helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert supported scientific Python values to plain JSON values.

    The conversion is intentionally explicit and does not fall back to ``repr``;
    silently serializing an opaque object would make a run impossible to audit.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            return {"__float__": str(value)}
        return value
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": to_jsonable(value.tolist()),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        if hasattr(value, "to_dict"):
            return to_jsonable(value.to_dict())
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((to_jsonable(item) for item in value), key=canonical_json)
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    raise TypeError(
        f"Object of type {type(value).__name__!r} is not supported by the "
        "PyMixEF archival JSON format."
    )


def from_jsonable(value: Any) -> Any:
    """Restore arrays and special floats from :func:`to_jsonable` output."""

    if isinstance(value, list):
        return [from_jsonable(item) for item in value]
    if isinstance(value, dict):
        if "__ndarray__" in value:
            array = np.asarray(from_jsonable(value["__ndarray__"]), dtype=value.get("dtype"))
            shape = tuple(value.get("shape", array.shape))
            return array.reshape(shape)
        if "__float__" in value:
            return float(value["__float__"])
        return {key: from_jsonable(item) for key, item in value.items()}
    return value


def canonical_json(value: Any) -> str:
    """Return canonical UTF-8 JSON text suitable for hashing and diffing."""

    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: Any, *, algorithm: str = "sha256") -> str:
    """Hash a serializable value using its canonical JSON representation."""

    digest = hashlib.new(algorithm)
    digest.update(canonical_json(value).encode("utf-8"))
    return f"{algorithm}:{digest.hexdigest()}"


def write_json(path: str | os.PathLike[str], value: Any, *, indent: int = 2) -> Path:
    """Atomically write archival JSON and return its path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        indent=indent,
        ensure_ascii=True,
        allow_nan=False,
    )
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    return destination


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read archival JSON created by :func:`write_json`."""

    with Path(path).open("r", encoding="utf-8") as stream:
        return from_jsonable(json.load(stream))
