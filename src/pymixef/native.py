"""Optional compiled-core discovery without changing scientific defaults."""

from __future__ import annotations

import ctypes
import importlib.resources
import platform
from pathlib import Path

from .errors import UnsupportedCapabilityError


def _library_names() -> tuple[str, ...]:
    system = platform.system()
    if system == "Windows":
        return ("pymixef_core.dll",)
    if system == "Darwin":
        return ("libpymixef_core.dylib", "pymixef_core.dylib")
    return ("libpymixef_core.so", "pymixef_core.so")


def library_path() -> Path | None:
    """Return the packaged native library path, if this distribution carries it."""

    root = importlib.resources.files("pymixef")
    for name in _library_names():
        candidate = root.joinpath("_native", name)
        if candidate.is_file():
            return Path(str(candidate))
    return None


def native_available() -> bool:
    """Return whether this installation contains a packaged native library.

    The check only discovers a platform-specific library file; it neither loads
    that library nor changes PyMixEF's selected computational backend.
    """

    return library_path() is not None


def core_version() -> str:
    """Query the native ABI version or raise a stable capability error."""

    path = library_path()
    if path is None:
        raise UnsupportedCapabilityError(
            "This wheel does not contain the optional compiled PyMixEF core.",
            code="NATIVE-UNAVAILABLE-001",
            remediation="Use the NumPy/SciPy reference backends or install a native wheel.",
        )
    library = ctypes.CDLL(str(path))
    function = library.pymixef_core_version
    function.restype = ctypes.c_char_p
    value = function()
    if value is None:
        raise RuntimeError("Native core returned a null version string.")
    return value.decode("ascii")
