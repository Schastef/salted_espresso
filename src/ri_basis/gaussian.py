from typing import Dict, Tuple
from .core import RadialFunctions
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


class PrimitiveGaussianRadials(RadialFunctions):

    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int, l_max, alphas: list[float] | Dict[Tuple[int, int], float]):
        super().__init__(species, origin, n_max, l_max)

        if isinstance(alphas, list):
            if len(alphas) != len(self):
                raise ValueError(f"Expected {len(self)} alphas for n_max={self.n_max} and l_max={self.l_max}, but got {len(alphas)}.")
            self.alphas = {self.running_to_lexographic_index(idx): alpha for idx, alpha in enumerate(alphas)}
        elif isinstance(alphas, dict):
            expected_keys = {self.running_to_lexographic_index(idx) for idx in range(len(self))}
            if set(alphas.keys()) != expected_keys:
                raise ValueError(f"Expected keys {expected_keys} for alphas dict, but got {set(alphas.keys())}.")
            self.alphas = alphas

        self.radials = []
        for (n, l) in self.alphas:
            alpha = self.alphas[(n, l)]
            radial_func = PrimitiveGaussian(alpha, l)
            self.radials.append(radial_func)



class PrimitiveGaussian:
    """Primitive Gaussian radial function implementing the ``RadialFunction`` protocol.

    Represents a radial function of the form

        R(r) = A * r**l * exp(-alpha * r**2),

    where ``alpha`` is the Gaussian exponent, ``l`` is the angular momentum,
    and ``A`` is a normalization constant chosen such that

        integral_0^infinity |R(r)|^2 r^2 dr = 1.

    Parameters
    ----------
    alpha : float
        Gaussian exponent controlling the radial decay.
    l : int
        Angular momentum quantum number. Determines the polynomial prefactor
        ``r**l``.
    """
    def __init__(self, alpha: float, l: int):
        self.alpha = alpha
        self.l = l
        self.amplitude = self._compute_norm_amplitude("analytical")


    def __call__(self, r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r)

        result = self.amplitude * np.exp(-self.alpha * r_arr**2) * r_arr**self.l

        return result

    def _compute_norm_amplitude(self, method: str = "analytical") -> float:
        match method:
            case "analytical":
                k = (3.0 + 2.0 * self.l) / 2.0
                I = 0.5 * (2.0 * self.alpha) ** (-k) * gamma(k)
                amplitude = float(np.sqrt(1.0 / I))
            case "numerical":
                def integrand(r):
                    return r**2 * self.__call__(r)**2
                integral, _ = quad(integrand, 0, np.inf)
                amplitude = np.sqrt(1.0 / integral)
            case _:
                raise ValueError(f"Unknown normalization method: {method}")
        return amplitude




