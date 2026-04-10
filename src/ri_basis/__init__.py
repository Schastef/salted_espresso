from .loader import load_basis, load_basis_set, register_angular, register_radial
from .gaussian import PrimitiveGaussianRadials
from .multiply_wfcs import Atomic_Rad
from .real_spher_harmonic import RealSphericalHarmonics


register_radial("gaussian", PrimitiveGaussianRadials)
register_radial("atomic", Atomic_Rad)

register_angular("spherical", RealSphericalHarmonics)
register_angular("real_spherical", RealSphericalHarmonics)

__all__ = [
    "load_basis",
    "load_basis_set"
]