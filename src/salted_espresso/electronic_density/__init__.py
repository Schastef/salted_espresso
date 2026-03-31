"""Compatibility namespace for electronic density modules."""

from importlib import import_module
import sys

_base = import_module("electronic_density")

__all__ = getattr(_base, "__all__", [])
__path__ = _base.__path__
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

for _submodule in (
    "cube2rho",
    "loader",
    "types",
):
    sys.modules[f"{__name__}.{_submodule}"] = import_module(f"electronic_density.{_submodule}")
