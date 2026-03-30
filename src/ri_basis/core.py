import numpy as np
from ase.io import read
import json

from typing import List, Tuple, Callable, Iterator


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
    """Base class for all radial functions

    RadialFunctions represents a set of radial functions, {R_{n,l}}, which are identified by their major and minor quantum numbers, n and l.

    An implementation of RadialFunctions has to populate set.radials with a list of callable objects that represent R_{n,l} in lexographic order,
    where n runs slow and l runs fast.

    Calling RadialFunctions with a cartesian point R will give a list of floats, representing the evaulations R_{n,l}(r) in lexographic order.
    """

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
    """Base class for all angular functions

    AnbgularFunctions represents a set of angular functions, {Y_l^m}}, which are identified by their minor and magnetic quantum numbers, l and m.
    """

    def __init__(self, species: str, origin: tuple[float, float, float], l_max: int):
        super().__init__(species, origin)
        self.l_max = l_max

    def __call__(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError


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
        m = idx - count
        return l, m - l


class RIBasis(RIFunctions):
    """Class representing a basis for the resolution of the identity (RI).

    The class represents a set of functions X_{nl}^m, where each such function is a product of a radial component, R_{nl}, and
    a angular component, Y_l^m. This set is centered around an origin, R, and associated with a specific chemical element.

    When calling an RIBasis object with a set of N cartesian points {r}, it will return a (N, n_basis) numpy array, where n_basis is
    the number of basis functions in the RI basis. The returned array contains the numerical evaluations of X_{nl}^m(r) in lexographic
    order (n,l,m), where n runs slowest and m fastest. 

    Example:
    If n_max=2 and l_max=2, the RIBasis is a set of 18 X-functions. Calling ribasis([x,y,z]) will return a np.array containing the results
    of [[X_{10}^0, X_{11}^-1, X_{11}^0, X_{11}^1, X_{12}^-2, X_{12}^-1 ... X_{12}^2, X_{20}^0, X_{21}^-1, ... , X_{22}^2]]

    Parameters:
    -----------
        species (str): Chemical species associated with the basis
        origin (tuple[float, float, float]): Origin of all basis functions
        n_max (int): Maximum major quantum number for basis functions
        l_max (int): Maximum minor quantum number for basis functions

        radial_cls: Class of RadialFunctions, implementing the radial part of the basis
        angular_cls: Class of AngularFunctions, implementing the angular part of the basis
        radial_kwargs: Key-word arguments passed to RadialFunctions
        angular_kwards: Key-word arguments passed to AngularFunction
    """

    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int, l_max: int,
                 radial_cls: type[RadialFunctions], angular_cls: type[AngularFunctions],
                 radial_kwargs: dict = None, angular_kwargs: dict = None):
        super().__init__(species, origin)
        self.n_max = n_max
        self.l_max = l_max

        self.radial_kwargs = radial_kwargs or {}
        self.angular_kwargs = angular_kwargs or {}

        # Instantiate radial and angular components
        # We pass common parameters (species, origin, n_max/l_max) automatically
        # Note: RadialFunctions expects (species, origin, n_max, l_max) + extra args
        # AngularFunctions expects (species, origin, l_max) + extra args

        self.radial_funcs = radial_cls(species, origin, n_max, l_max, **self.radial_kwargs)
        self.angular_funcs = angular_cls(species, origin, l_max, **self.angular_kwargs)


    def __call__(self, r: np.ndarray) -> np.ndarray:
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r.shape}")

        rad_vals = self.radial_funcs(r)  # Shape (N, n_rad_pairs)
        ang_vals = self.angular_funcs(r) # Shape (N, n_ang_funcs)

        repeats = [2 * l + 1 for l in range(self.l_max + 1)] * self.n_max
        rad_vals_expanded = np.repeat(rad_vals, repeats, axis=1)

        ang_vals_expanded = np.tile(ang_vals, (1, self.n_max))

        return rad_vals_expanded * ang_vals_expanded


    def __len__(self):
        return self.n_max * (self.l_max+1)**2


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

        m = idx - count
        return n, l, m - l


class RIBasisSet():
    """Class representing a complete RI basis for a structure, that is, a list of RIBasis objects for every atom

    Parameters
    ----------

    structure_file: str
        Path to a structure file (e.g. .xyz, .cif) containing the structural information. Read with ase.io.read
    specifications: dict | str
        A dictionary mapping chemical species to their basis specifications. The basis specifications correspond
        directly to the parameters of RIBasis, except for species and origin which are determined from the structure file
        For example:

        {
            "O": {
                "n_max": 2,
                "l_max": 2,
                "radial_method": "gaussian",
                "angular_method": "real_spherical",
                "radial_kwargs": {"alphas": [0.5, 1.0]},
                "angular_kwargs": {}
            },
        }

    The dictionary can be provided as a path to as .json file as well.

    ribasis_loader: Callable
        A function that takes the parameters (species, origin, n_max, l_max, radial_method, angular_method, radial_kwargs,
        angular_kwargs) and returns an RIBasis object.

    order_by_species: bool
        Whether to order the RIBasis objects in the final list by species. If False, the ordering will be the same as in
        the structure file.

    Methods:
    --------

    __call__(r: np.ndarray) -> np.ndarray
        Calling the RIBasisSet with a set of cartesian points will return a block diagonal array containing the
        individual evaluations of the RIBasis functions for each atom.

    """

    def __init__(self, structure_file: str, specifications: dict | str,
                 ribasis_loader: Callable, order_by_species: bool = False):
        if isinstance(specifications, str):
            with open(specifications, 'r') as f:
                specifications = json.load(f)
        self.specifications = specifications
        self.species_and_positions = self._load_structure(structure_file, return_ordered=order_by_species)
        self.ribases = []
        self.loader_func = ribasis_loader

        for species, position in self.species_and_positions:
            if species not in specifications:
                raise ValueError(f"Species '{species}' found in structure file but not in specifications.")
            ribasis = self._load_ribasis(species, position)
            self.ribases.append(ribasis)


    def __call__(self, r: np.ndarray) -> np.ndarray:
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r.shape}")

        results = [ribasis(r) for ribasis in self.ribases]
        return np.hstack(results)


    def __len__(self) -> int:
        return len(self.ribases)


    def __iter__(self) -> Iterator[RIBasis]:
        return iter(self.ribases)


    def __getitem__(self, item):
        return self.ribases[item]


    @staticmethod
    def _load_structure(structure_file: str, return_ordered: bool) \
            -> List[Tuple[str, Tuple[float, float, float]]]:
        atoms = read(str(structure_file))
        species_list: List[Tuple[str, Tuple[float, float, float]]] = [(str(atom.symbol), tuple(atom.position)) for atom in atoms]

        if return_ordered:
            species_list.sort(key=lambda x: (x[0], x[1][2], x[1][0], x[1][1]))

        return species_list


    def _load_ribasis(self, species: str, origin: tuple[float, float, float]) -> RIBasis:
        specs = self.specifications[species]
        return self.loader_func(
            species=species,
            origin=origin,
            **specs
        )
