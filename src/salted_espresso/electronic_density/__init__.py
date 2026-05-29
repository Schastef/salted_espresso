from .loader import load_rho, register_loader
from .cube2rho import load_rho_from_cube

register_loader(".cube", load_rho_from_cube)

__all__ = ["load_rho"]