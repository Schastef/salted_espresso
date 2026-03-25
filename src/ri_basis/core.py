from .types import *
import numpy as np


class RIFunctions:
    """Base class for RI basis functions, providing common attributes and methods for radial, angular and combined functions.
    """

    def __init__(self, species: str, origin: tuple[float, float, float]):
        self.species = species
        self.origin = np.array(origin)

    def compute(self, r: np.ndarray) -> np.ndarray:
        r_arr = np.ascontiguousarray(r)
        if r_arr.ndim != 2 or r_arr.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r_arr.shape}")
        return self.__call__(r_arr - self.origin)


    def __call__(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def __len__(self) -> int:
        raise NotImplementedError


    def lexographic_to_running_index(self, index: tuple) -> int:
        raise NotImplementedError


    def running_to_lexographic_index(self, idx: int) -> tuple:
        raise NotImplementedError


class RadialFunctions(RIFunctions):

    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int, l_max: int):
        super().__init__(species, origin)
        self.n_max = n_max
        self.l_max = l_max
        self.radials = []  # Expected to be populated by subclasses


    def __call__(self, r: np.ndarray) -> np.ndarray:
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r.shape}")

        radii = np.linalg.norm(r, axis=1)

        # Evaluate each radial function for all radii
        # Each radial function returns shape (N,)
        # We want final shape (N, M)
        results = [func(radii) for func in self.radials]

        if not results:
             return np.zeros((r.shape[0], 0))

        return np.stack(results, axis=1)


    def __len__(self):
        return self.n_max * (self.l_max + 1)


    def lexographic_to_running_index(self, index: tuple) -> int:
        n, l = index
        return n * (self.l_max + 1) + l


    def running_to_lexographic_index(self, idx: int) -> tuple:
        n = idx // (self.l_max + 1)
        l = idx % (self.l_max + 1)
        return n, l


class AngularFunctions(RIFunctions):

    def __init__(self, species: str, origin: tuple[float, float, float], l_max: int):
        super().__init__(species, origin)
        self.l_max = l_max

    def __call__(self, r: np.ndarray) -> np.ndarray:
        pass


    def __len__(self):
        return sum(2 * l + 1 for l in range(self.l_max + 1))


    def lexographic_to_running_index(self, index: tuple) -> int:
        l, m = index
        return sum(2 * l_ + 1 for l_ in range(l)) + (m + l)


    def running_to_lexographic_index(self, idx: int) -> tuple:
        l = 0
        count = 0
        while count + (2 * l + 1) <= idx:
            count += 2 * l + 1
            l += 1
        m = idx - count - l
        return l, m - l


class RIBasis(RIFunctions):

    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int, l_max: int,
                 radial_funcs: RadialFunctions, angular_funcs: AngularFunctions):
        super().__init__(species, origin)
        self.n_max = n_max
        self.l_max = l_max
        self.radial_funcs = radial_funcs
        self.angular_funcs = angular_funcs


    def __call__(self, r: np.ndarray) -> np.ndarray:
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r.shape}")

        # Components expect coordinates relative to their center (passed 'r' is assumed relative)
        rad_vals = self.radial_funcs(r)  # Shape (N, n_rad_pairs)
        ang_vals = self.angular_funcs(r) # Shape (N, n_ang_funcs)

        raise NotImplementedError




    def __len__(self):
        return self.n_max * sum(2 * l + 1 for l in range(self.l_max + 1))


    def lexographic_to_running_index(self, index: tuple) -> int:
        n, l, m = index
        return n * sum(2 * l_ + 1 for l_ in range(self.l_max + 1)) + sum(2 * l_ + 1 for l_ in range(l)) + (m + l)


    def running_to_lexographic_index(self, idx: int) -> tuple:
        n = 0
        count = 0
        while count + sum(2 * l_ + 1 for l_ in range(self.l_max + 1)) <= idx:
            count += sum(2 * l_ + 1 for l_ in range(self.l_max + 1))
            n += 1

        idx -= count
        l = 0
        count = 0
        while count + (2 * l + 1) <= idx:
            count += 2 * l + 1
            l += 1

        m = idx - count - l
        return n, l, m - l






















