"""Typed, deterministic registries for PyMixEF extension points."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from importlib import metadata
from threading import RLock
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from .errors import PluginError

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PluginInfo:
    """Metadata retained for one registered implementation."""

    name: str
    implementation: Any
    version: str | None = None
    source: str = "runtime"


class Registry(Generic[T]):
    """Thread-safe registry with normalized names and explicit replacement."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, PluginInfo] = {}
        self._lock = RLock()

    @staticmethod
    def normalize(name: str) -> str:
        key = name.strip().lower().replace("_", "-")
        if not key:
            raise PluginError("Plugin names cannot be empty.", code="PLUGIN-NAME-001")
        return key

    def register(
        self,
        name: str,
        implementation: T | None = None,
        *,
        replace: bool = False,
        version: str | None = None,
        source: str = "runtime",
    ) -> T | Callable[[T], T]:
        """Register an implementation, directly or as a decorator."""

        def add(value: T) -> T:
            key = self.normalize(name)
            with self._lock:
                if key in self._entries and not replace:
                    raise PluginError(
                        f"{self.kind.title()} plugin {key!r} is already registered.",
                        code="PLUGIN-DUPLICATE-001",
                        details={"kind": self.kind, "name": key},
                    )
                self._entries[key] = PluginInfo(key, value, version, source)
            return value

        return add if implementation is None else add(implementation)

    def unregister(self, name: str) -> T:
        """Remove and return a runtime registration."""

        key = self.normalize(name)
        with self._lock:
            try:
                return cast(T, self._entries.pop(key).implementation)
            except KeyError as exc:
                raise PluginError(
                    f"Unknown {self.kind} plugin {key!r}.",
                    code="PLUGIN-NOT-FOUND-001",
                ) from exc

    def get(self, name: str) -> T:
        """Resolve an implementation by normalized name."""

        key = self.normalize(name)
        with self._lock:
            try:
                return cast(T, self._entries[key].implementation)
            except KeyError as exc:
                raise PluginError(
                    f"Unknown {self.kind} plugin {key!r}.",
                    code="PLUGIN-NOT-FOUND-001",
                    details={"available": list(self.names())},
                ) from exc

    def info(self, name: str) -> PluginInfo:
        """Return immutable registration metadata."""

        key = self.normalize(name)
        with self._lock:
            try:
                return self._entries[key]
            except KeyError as exc:
                raise PluginError(
                    f"Unknown {self.kind} plugin {key!r}.",
                    code="PLUGIN-NOT-FOUND-001",
                ) from exc

    def names(self) -> tuple[str, ...]:
        """Return deterministic registered names."""

        with self._lock:
            return tuple(sorted(self._entries))

    def snapshot(self) -> Mapping[str, PluginInfo]:
        """Return a detached, sorted registry snapshot."""

        with self._lock:
            return {name: self._entries[name] for name in sorted(self._entries)}

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.normalize(name) in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


@runtime_checkable
class CovariancePlugin(Protocol):
    """Minimal covariance plugin contract."""

    def covariance(self, parameters: Any, **context: Any) -> Any:
        """Construct a covariance matrix."""

    def validate(self, matrix: Any, **context: Any) -> Any:
        """Validate a covariance matrix."""

    def simulate(self, parameters: Any, **context: Any) -> Any:
        """Simulate a zero-mean draw."""


@runtime_checkable
class EstimatorPlugin(Protocol):
    """Minimal estimator extension contract."""

    def fit(self, model: Any, data: Any, **controls: Any) -> Any:
        """Fit a compiled model."""


FAMILY_REGISTRY: Registry[Any] = Registry("family")
LINK_REGISTRY: Registry[Any] = Registry("link")
COVARIANCE_REGISTRY: Registry[Any] = Registry("covariance")
ESTIMATOR_REGISTRY: Registry[Any] = Registry("estimator")
DIAGNOSTIC_REGISTRY: Registry[Any] = Registry("diagnostic")
EXPORTER_REGISTRY: Registry[Any] = Registry("exporter")
ODE_SOLVER_REGISTRY: Registry[Any] = Registry("ode-solver")


def discover_plugins(group: str = "pymixef.plugins") -> tuple[str, ...]:
    """Load package entry points in deterministic name order.

    An entry point may be a zero-argument registration function or an import
    whose module-level code performs registration.
    """

    loaded: list[str] = []
    selected = metadata.entry_points().select(group=group)
    for entry_point in sorted(selected, key=lambda item: item.name):
        try:
            target = entry_point.load()
            if callable(target):
                target()
        except Exception as exc:
            raise PluginError(
                f"Failed to load plugin entry point {entry_point.name!r}.",
                code="PLUGIN-LOAD-001",
                details={"entry_point": entry_point.name, "group": group},
            ) from exc
        loaded.append(entry_point.name)
    return tuple(loaded)


__all__ = [
    "COVARIANCE_REGISTRY",
    "DIAGNOSTIC_REGISTRY",
    "ESTIMATOR_REGISTRY",
    "EXPORTER_REGISTRY",
    "FAMILY_REGISTRY",
    "LINK_REGISTRY",
    "ODE_SOLVER_REGISTRY",
    "CovariancePlugin",
    "EstimatorPlugin",
    "PluginInfo",
    "Registry",
    "discover_plugins",
]
