from .types import RIBasis
from typing import Callable, Dict

LOADERS: Dict[str, Callable[..., RIBasis]] = {}

def register_loader(name: str, loader: Callable[..., RIBasis]) -> Callable[..., RIBasis]:
    LOADERS[name] = loader

def load(method: str, **kwargs) -> RIBasis:
    """
    Defines a method that returns an RI Basis.

    Parameters:
        method (str): Method to construct RI Basis
        **kwargs: Key-word arugments for specific method
    """

    return LOADERS[method](**kwargs)
