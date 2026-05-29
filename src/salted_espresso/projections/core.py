import numpy as np
from salted_espresso.electronic_density.types import DensityFunction
from salted_espresso.ri_basis.core import RIBasisSet

def compute_overlap_matrix(basis_set: RIBasisSet) -> np.ndarray:
    """Computeds the overlap matrix S_ij = <chi_i | chi_j> for the given density and basis set."""
    pass


def compute_projection_vector(
    rho: DensityFunction,
    basis_set: RIBasisSet,
    n_cartesian_grid: int | tuple[int, int, int] | None = None,
    cell_grid: np.ndarray = None,
) -> np.ndarray:
    """Computes the individual projection coefficients P_i = <rho | chi_i> for the given density and basis set."""
    if cell_grid is None:
        if hasattr(rho, "cell_grid") and rho.cell_grid is not None:
            cell_grid = rho.cell_grid
        else:
            raise ValueError("cell_grid must be provided either as an argument or as an attribute of the density.")

    if n_cartesian_grid is None:
        if hasattr(rho, "grid_shape") and rho.grid_shape is not None:
            n_cartesian_grid = rho.grid_shape
        else:
            n_cartesian_grid = (64, 64, 64)

    if isinstance(n_cartesian_grid, int):
        n_cartesian_grid = (n_cartesian_grid, n_cartesian_grid, n_cartesian_grid)

    nx, ny, nz = n_cartesian_grid

    f1 = np.linspace(0, 1, nx, endpoint=False)
    f2 = np.linspace(0, 1, ny, endpoint=False)
    f3 = np.linspace(0, 1, nz, endpoint=False)

    dV = abs(np.linalg.det(cell_grid)) / (nx * ny * nz)
    F1, F2, F3 = np.meshgrid(f1, f2, f3, indexing='ij')
    frac_pts = np.stack([F1.ravel(), F2.ravel(), F3.ravel()], axis=-1)

    points = frac_pts @ cell_grid

    P = None
    chunk_size = 10000
    for i in range(0, len(points), chunk_size):
        chunk = points[i:i+chunk_size]
        dP = (basis_set(chunk).T @ rho(chunk)) * dV
        if P is None:
            P = dP
        else:
            P += dP

    return P


def solve_projections_coeffs(overlap_matrix: np.ndarray, projection_vector: np.ndarray) -> np.ndarray:
    """Solves the linear system S c = P to obtain the projection coefficients c_i."""
    pass
