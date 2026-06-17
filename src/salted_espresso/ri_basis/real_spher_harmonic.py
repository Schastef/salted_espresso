from .core import AngularFunctions
import numpy as np
import sphericart as sc


class RealSphericalHarmonics(AngularFunctions):

    def __init__(self, species: str, origin: tuple[float, float, float], l_max: int):
        super().__init__(species, origin, l_max)
        self.sph_harm = sc.SphericalHarmonics(l_max)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r.shape}")

        sph_vals = self.sph_harm.compute(r)
        sph = np.asarray(sph_vals)

        return sph

    def evaluate_fourier_angular(
            self,
            g_vectors: np.ndarray,
            l: int,
    ) -> np.ndarray:
        """Evaluate real spherical harmonics Y_l^m(Ghat) for m=-l..l."""
        g_vectors = np.asarray(g_vectors, dtype=float)

        if g_vectors.ndim != 2 or g_vectors.shape[1] != 3:
            raise ValueError(
                f"g_vectors must have shape (n_G, 3), got {g_vectors.shape}"
            )

        g_norm = np.linalg.norm(g_vectors, axis=1)

        g_hat = np.zeros_like(g_vectors)
        nonzero = g_norm > 0.0
        g_hat[nonzero] = g_vectors[nonzero] / g_norm[nonzero, None]

        # Direction at G=0 is undefined. Choose z-axis as harmless placeholder.
        # For l>0, the radial Bessel transform already gives G^l -> 0.
        g_hat[~nonzero] = np.array([0.0, 0.0, 1.0])

        sph = np.asarray(self.sph_harm.compute(g_hat))

        start = l * l
        stop = (l + 1) * (l + 1)

        return sph[:, start:stop]
