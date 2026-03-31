"""Compatibility namespace for RI basis modules."""

from importlib import import_module
import sys

_base = import_module("ri_basis")

# Expose package contents.
__all__ = getattr(_base, "__all__", [])
__path__ = _base.__path__
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

# Ensure namespaced submodules resolve to the same module objects.
for _submodule in (
    "core",
    "loader",
    "gaussian",
    "real_spher_harmonic",
    "realspherharmonic",
    "types",
):
    sys.modules[f"{__name__}.{_submodule}"] = import_module(f"ri_basis.{_submodule}")
