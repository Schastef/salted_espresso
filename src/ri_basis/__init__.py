from .loader import load_basis, register_angular, register_radial
from .gaussian import PrimitiveGaussianRadials
from .realspherharmonic import RealSphericalHarmonics


register_radial("gaussian", PrimitiveGaussianRadials)

register_angular("spherical", RealSphericalHarmonics)
register_angular("real_spherical", RealSphericalHarmonics)

__all__ = ["load_basis"]