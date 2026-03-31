"""Compatibility namespace for projection modules."""

from importlib import import_module
import sys

_base = import_module("projections")

__all__ = getattr(_base, "__all__", [])
__path__ = _base.__path__
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

for _submodule in (
    "core",
    "overlap",
    "utils",
):
    sys.modules[f"{__name__}.{_submodule}"] = import_module(f"projections.{_submodule}")
