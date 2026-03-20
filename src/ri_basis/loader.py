from .types import RIBasis
from typing import Callable, Dict

LOADERS: Dict[str, Callable[..., RIBasis]] = {}

def register_loader(name: str, loader: Callable[..., DensityFunction]) -> Callable[..., DensityFunction]:
    LOADERS[name] = loader

def load(method: str, **kwargs) -> RIBasis:
    """
    Returns an RIBasis, a mapping from RIKeys to callable RI functions.

    Parameters:
        method (str): Method to construct RI Basis
        **kwargs: Key-word arugments for specific method
    """

    return LOADERS[method](**kwargs)
