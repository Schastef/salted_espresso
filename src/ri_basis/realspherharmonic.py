from .core import RadialFunctions
from .types import ScalarOrArray, ArrayF
import numpy as np
import sphericart as sc


class RealSphericalHarmonics(RadialFunctions):

    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int, l_max: int):
        super().__init__(species, origin, n_max, l_max)
        self.sph_harm = sc.SphericalHarmonics(l_max)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r.shape}")

        sph_vals = self.sph_harm.compute(r)
        sph = np.asarray(sph_vals)

        return np.tile(sph, (1, self.n_max))
