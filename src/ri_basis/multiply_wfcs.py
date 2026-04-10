from unittest import result

import numpy as np
import scipy
from sympy.physics.quantum.cg import CG
from collections.abc import Callable
import scipy.special as sp
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline
import types
import linecache

from .core import RadialFunctions
#import sympy

##### Helper functions: multiply spherical harmonics

def Clebsch_Gordan_coeff(l1 : int, m1 : int, l2 : int, m2: int) -> dict:

    """
    Prepares Clebsch Gordan coefficients for multiplication of spherical hermonics, and returns them as a dictionary with the l and m as keys.
    Arguments:
    l1, m1: l and m of the first spherical harmonic
    l2, m2: l and m of the second spherical harmonic
    Returns:
    A dictionary with the Clebsch Gordan coefficients for the multiplication of the two spherical harmonics, with the keys being the resulting l and m values of the product spherical harmonic. The values are the corresponding Clebsch Gordan coefficients.
    """
    coeffs = {}
    for L in range(np.abs(l1 - l2), l1 + l2 + 1):
        print(L)
        M = m1 + m2
        coeff = np.sqrt((2 * l1 + 1) * (2 * l2 + 1)/ (4 * np.pi * (2 * L + 1))) * CG(l1, 0, l2, 0, L, 0) * CG(l1, m1, l2, m2, L, M)
        coeffs[(L, M)] = coeff.doit() #convert from sympy expression to a numerical value

    
    return coeffs

def multiply_spherical_harmonics(l1 : int, m1 : int, l2 : int, m2: int) -> Callable:

    """
    Multiplies two spherical harmonics and returns the resulting function as a callable.
    Arguments:
    l1, m1: l and m of the first spherical harmonic
    l2, m2: l and m of the second spherical harmonic
    Returns:
    A callable function that takes theta and phi as arguments and returns the value of the product of the two spherical harmonics at those angles.
    """
    coeffs = Clebsch_Gordan_coeff(l1, m1, l2, m2)
    
    def product(theta, phi):
        result = 0
        for (L, M), coeff in coeffs.items():
            Y_LM = sp.sph_harm_y(L, M, theta, phi)
            result += coeff * Y_LM
        return result
    
    return product

###### Helper Functions: compute radial overlap between two radial functions

def compute_radial_overlap(r1: Callable, r2: Callable, r_max: float, n_points: int) -> float:
    r = np.linspace(0, r_max, n_points)
    integrand = r1(r) * r2(r) * r**2
    return np.trapz(integrand, r)


def select_l1l2(l_prime: int) -> list[tuple[int, int]]:
    """
    Selects the allowed pairs of l1 and l2 values for a given l' based on the triangle inequality.
    Arguments:
    l_prime: The resulting l value of the product spherical harmonic
    Returns:
    A list of tuples, where each tuple contains a pair of l1 and l2 values that satisfy the triangle inequality for the given l'.
    """
    allowed_pairs = []
    for l1 in range(0, l_prime + 1):
        for l2 in range(l1, l_prime + 1):
            if np.abs(l1 - l2) <= l_prime <= (l1 + l2):
                allowed_pairs.append((l1, l2))
    return allowed_pairs

def multiply_radial_fcts(l_list: np.ndarray, rfunc_list: list[Callable], lprime_max: int) -> tuple[np.ndarray[Callable], list[int]]:

    """
    Multiplies radial functions corresponding to different l values and returns the resulting functions. There are multiple functions for each l'.
    Arguments:
    l_list: An array of l values corresponding to the radial functions
    rfunc_list: A list of callable radial functions corresponding to the l values in l_list
    lprime_max: The maximum l' value for which to compute the product radial functions
    Returns:
    A list of callable functions, where each function corresponds to a specific l' value and represents the product of the radial functions for that l'.
    """
    rfunc_products = []
    lprime_list = []
    for l_prime in range(lprime_max + 1):
        allowed_pairs = select_l1l2(l_prime)

        for (l1, l2) in allowed_pairs:

            l1ind = np.where(l_list == l1)[0]
            l2ind = np.where(l_list == l2)[0]
            for ind in l1ind:
                for ind2 in l2ind:
                    f1 = rfunc_list[ind]
                    f2 = rfunc_list[ind2]

                    def product_rfunc(r, f1=f1, f2=f2):
                        return f1(r) * f2(r)
                    rfunc_products.append(product_rfunc)
                    lprime_list.append(l_prime)
    
    return np.array(rfunc_products), lprime_list


##Helper function to generate linear combinations of radial functions based on the eigenvectors of the overlap matrix
def make_rfunc(vec, funcs):
    def new_rfunc(r):
        return sum(v * f(r) for v, f in zip(vec, funcs))
    return new_rfunc


def compress_basis(radial_basis: list[Callable], lprime_list : list[int], overlap_ninds: int, rmax: float = 10.0, n_points: int = 1000) -> tuple[list[Callable], list[int]]:

    """
    Compresses a basis set by computing the overlap matrix of all radial functions with the same lprime. We then calculate the eigenvalues and eigenvectors of the overlap matrix, and discard the eigenvectors corresponding to eigenvalues below a certain cutoff.
    The remaining eigenvectors are used to construct a new set of radial functions that form the compressed basis.
    Arguments:
    
    radial_basis: A list of tuples, where each tuple contains a callable radial function and its corresponding l' value
    lprime_list: A list of l' values corresponding to the radial functions in radial_basis
    overlap_ninds: An integer specifying the number of eigenvectors to keep based on the cumulative overlap. For example, if overlap_ninds=10, we keep the eigenvectors corresponding to the largest 10 eigenvalues that contribute to the cumulative overlap.
    rmax: The maximum radius up to which the radial functions are defined.
    n_points: The number of points to use in the numerical integration for computing radial overlaps.
    Returns:
    A list of tuples, where each tuple contains a callable radial function and its corresponding l' value, representing the compressed basis.
    """
    compressed_basis = []
    lprime_list_new = []
    lprime_values = set(lprime_list)
    for l_prime in lprime_values:
        rfuncs_for_lprime = radial_basis[np.array(lprime_list) == l_prime]
        n_funcs = len(rfuncs_for_lprime)
        #print(f"Compressing basis for l'={l_prime} with {n_funcs} functions")
        overlap_matrix = np.zeros((n_funcs, n_funcs))
        for i in range(n_funcs):
            for j in range(n_funcs):
                overlap_matrix[i, j] = compute_radial_overlap(rfuncs_for_lprime[i], rfuncs_for_lprime[j], r_max=rmax, n_points=n_points)

        eigenvalues, eigenvectors = np.linalg.eigh(overlap_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        #sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        #cumulative_overlap = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)

        for i in range(overlap_ninds[l_prime]):
            vec = sorted_eigenvectors[:, i]
            new_rfunc = make_rfunc(vec, rfuncs_for_lprime)
            compressed_basis.append(new_rfunc)
            lprime_list_new.append(l_prime)

    return compressed_basis, lprime_list

def read_from_file(filename: str) -> tuple[np.ndarray, list[Callable]]:
    """
    Reads radial functions from a file and returns them as a list of callables along with their corresponding l values.
    Arguments:
    filename: The name of the file containing the radial functions. The file is expected to have a specific format where the first line contains the l values, and the subsequent lines contain the radial function values at different radii.
    Returns:
    A tuple containing an array of l values and a list of callable radial functions corresponding to those l values.
    """
    data = np.genfromtxt(filename, dtype = np.float64, skip_header=2)
    #print(data)
    R_grid = data[:,1]
    radial_wfcs = data[:,2:]
    radial_wfcs = (radial_wfcs.T/R_grid).T
    l_vals = linecache.getline(filename, 2)
    l_vals = np.array(list(map(int, l_vals.split())))
    rfuncs = [
        make_interp_spline(R_grid, radial_wfcs[:, i], k=3)
        for i in range(radial_wfcs.shape[1])
    ]
    return l_vals, rfuncs
    

class Atomic_Rad(RadialFunctions):
    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int, l_max: int, filename: str):
        super().__init__(species, origin, n_max, l_max)
        l_list, rfunc_list = read_from_file(filename)
        wfcs_sq, lprime_list = multiply_radial_fcts(l_list, rfunc_list, lprime_max=self.l_max)
        self.radials, self.l_list = compress_basis(wfcs_sq, lprime_list, overlap_ninds=self.n_max)
        ## sort radials by lprime and save without the lprime index -> we trust that the functions are assigned correctly based on ordering


    def __call__(self, r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r)
        radii = np.linalg.norm(r_arr, axis=1)  

        result = np.squeeze(np.array([radial_func(radii) for radial_func in self.radials]))

        return result
    
    def estimate_cutoff(self, threshold: float = 1e-5) -> float:
        r = np.linspace(0, 20.0, 1000)
        radial_values = [radial_func(r) for radial_func in self.radials]
        max_value = np.max(np.abs(radial_values))
        cutoff_indices = np.where(np.abs(radial_values) < threshold * max_value)[0]
        if len(cutoff_indices) == 0:
            return r[-1]
        else:
            return r[cutoff_indices[0]]
    
    def _compute_norm_amplitude(self, method: str = "analytical") -> float:
        match method:
            case "analytical":
                raise ValueError("Analytical normalization not implemented for this type of radial function.")
            case "numerical":
                def integrand(r):
                    return r**2 * self.__call__(r)**2
                integral, _ = quad(integrand, 0, np.inf)
                amplitude = np.sqrt(1.0 / integral)
            case _:
                raise ValueError(f"Unknown normalization method: {method}")
        return amplitude