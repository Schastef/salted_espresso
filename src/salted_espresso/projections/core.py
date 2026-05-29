import numpy as np
from salted_espresso.electronic_density.types import DensityFunction
from salted_espresso.ri_basis.core import RIBasisSet

def compute_overlap_matrix(basis_set: RIBasisSet) -> np.ndarray:
    """Computeds the overlap matrix S_ij = <chi_i | chi_j> for the given density and basis set."""
    pass


def compute_projection_vector(
    rho: DensityFunction,
    basis_set: RIBasisSet,
    n_cartesian_grid: int = 64,
    r_max: float = 8.0,
) -> np.ndarray:
    """Computes the individual projection coefficients P_i = <rho | chi_i> for the given density and basis set."""
    if hasattr(basis_set, "ribases"):
        origins = np.array([b.origin for b in basis_set])
    else:
        origins = np.array([basis_set.origin])

    min_bounds = origins.min(axis=0) - r_max
    max_bounds = origins.max(axis=0) + r_max

    x = np.linspace(min_bounds[0], max_bounds[0], n_cartesian_grid)
    y = np.linspace(min_bounds[1], max_bounds[1], n_cartesian_grid)
    z = np.linspace(min_bounds[2], max_bounds[2], n_cartesian_grid)

    dV = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    P = np.zeros(len(basis_set))
    chunk_size = 10000
    for i in range(0, len(points), chunk_size):
        chunk = points[i:i+chunk_size]
        P += (basis_set(chunk).T @ rho(chunk)) * dV

    return P


def solve_projections_coeffs(overlap_matrix: np.ndarray, projection_vector: np.ndarray) -> np.ndarray:
    """Solves the linear system S c = P to obtain the projection coefficients c_i."""
    pass
