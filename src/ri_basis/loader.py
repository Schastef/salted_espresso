from typing import Type, Any, Dict, Optional
from .core import RIBasis, RadialFunctions, AngularFunctions
from .gaussian import PrimitiveGaussianRadials
from .real_spher_harmonic import RealSphericalHarmonics

RADIAL_REGISTRY: Dict[str, Type[RadialFunctions]] = {}

ANGULAR_REGISTRY: Dict[str, Type[AngularFunctions]] = {}

def register_radial(name: str, cls: Type[RadialFunctions]):
    """Register a new radial function implementation."""
    RADIAL_REGISTRY[name] = cls


def register_angular(name: str, cls: Type[AngularFunctions]):
    """Register a new angular function implementation."""
    ANGULAR_REGISTRY[name] = cls


def load_basis(species: str,
               origin: tuple[float, float, float],
               n_max: int,
               l_max: int,
               radial_method: str = "gaussian",
               angular_method: str = "real_spherical",
               radial_params: Optional[Dict[str, Any]] = None,
               angular_params: Optional[Dict[str, Any]] = None) -> RIBasis:
    """
    Load and initialize an RI Basis set based on the specified methods and parameters.

    Parameters
    ----------
    species : str
        Chemical species label (e.g. 'O', 'H').
    origin : tuple[float, float, float]
        Center of the basis functions.
    n_max : int
        Maximum radial quantum number (exclusive, i.e., number of radial functions).
    l_max : int
        Maximum angular momentum quantum number (inclusive).
    radial_method : str, optional
        Name of the radial function method to use (default: 'gaussian').
    angular_method : str, optional
        Name of the angular function method to use (default: 'spherical_harmonics').
    radial_params : dict, optional
        Additional parameters to pass to the radial function constructor (e.g. {'alphas': [...]}).
    angular_params : dict, optional
        Additional parameters to pass to the angular function constructor.

    Returns
    -------
    RIBasis
        The constructed RI Basis object.

    Raises
    ------
    ValueError
        If the specified radial or angular method is not registered.
    """

    radial_cls = RADIAL_REGISTRY.get(radial_method)
    if not radial_cls:
        raise ValueError(f"Unknown radial method: '{radial_method}'. Available: {list(RADIAL_REGISTRY.keys())}")

    angular_cls = ANGULAR_REGISTRY.get(angular_method)
    if not angular_cls:
        raise ValueError(f"Unknown angular method: '{angular_method}'. Available: {list(ANGULAR_REGISTRY.keys())}")

    return RIBasis(species, origin, n_max, l_max,
                   radial_cls, angular_cls,
                   radial_kwargs=radial_params,
                   angular_kwargs=angular_params)


def load_basis_set(structure_file: str,
                   specifications: dict | str,
                   order_by_species: bool = False) -> "RIBasisSet":
    """
    Load a complete RI basis set for a given structure.

    This function acts as a factory for creating an RIBasisSet object, injecting the
    necessary loader function to construct individual RIBasis instances.

    Parameters
    ----------
    structure_file : str
        Path to a structure file (e.g., .xyz, .cif).
    specifications : dict or str
        A dictionary or path to a JSON file with basis specifications for each species.
    order_by_species : bool, optional
        If True, order the basis functions by species (default is False).

    Returns
    -------
    RIBasisSet
        The constructed RI basis set.
    """
    from .core import RIBasisSet  # Local import to avoid circular dependency
    return RIBasisSet(structure_file, specifications, load_basis, order_by_species)
