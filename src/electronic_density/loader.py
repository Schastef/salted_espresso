from pathlib import Path

from .cube2rho import load_rho_from_cube
from .types import DensityFunction
from typing import Callable, Dict

LOADERS: Dict[str, Callable[..., DensityFunction]] = {}

def register_loader(name: str, loader: Callable[..., DensityFunction]) -> Callable[..., DensityFunction]:
    LOADERS[name] = loader


def load_rho(path: str | Path) -> DensityFunction:
    """Loads a callable rho function from a specified data file.

    Supported file formats:
        - .cube files

    Parameters:
        path (str | Path): The file path to the data file.

    Returns:
        A callable function rho(r) that returns the electronic density at any point r in space.
    """

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in LOADERS:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats are: {list(LOADERS.keys())}")

    return LOADERS[suffix](path)

