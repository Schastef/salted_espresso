from .types import *
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


class GaussianRadial:

    def __init__(self, alpha: float, l: int):
        self.alpha = alpha
        self.l = l
        self.amplitude = self.compute_norm_amplitude("analytical")


    def __call__(self, r: ScalarOrArray) -> ScalarOrArray:
        return self.amplitude * np.exp(-self.alpha * r**2) * r**self.l


    def compute_norm_amplitude(self, method: str = "analytical") -> float:
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




