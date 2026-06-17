import numpy as np
from typing import cast
from salted_espresso.electronic_density.types import DensityFunction
from salted_espresso.ri_basis.core import RIBasisSet

from salted_espresso.projections.overlap_integrals import (
    compute_overlap_matrix_overlap_metric,
    compute_overlap_matrix_coulomb_metric,
)

from salted_espresso.projections.projection import (
     compute_projection_vector_cartesian_grid,
     compute_projection_vector_native_grid,
     compute_projection_vector_coulomb_metric,
)

from tqdm.auto import tqdm

def compute_overlap_matrix(
    basis_set: RIBasisSet,
    metric: str = "overlap",
    print_progress_bar: bool = False,
) -> np.ndarray:

    if metric == "overlap":
        return compute_overlap_matrix_overlap_metric(
            basis_set,
            print_progress_bar=print_progress_bar,
        )

    elif metric == "coulomb":
        return compute_overlap_matrix_coulomb_metric(
            basis_set,
            print_progress_bar=print_progress_bar,
        )

    raise ValueError(f"Unknown metric '{metric}'")


def compute_projection_vector(
    rho: DensityFunction,
    basis_set: RIBasisSet,
    n_cartesian_grid: int | tuple[int, int, int] | None = None,
    cell_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Computes the individual projection coefficients P_i = <rho | chi_i> for the given density and basis set."""
    if cell_grid is None:
        rho_cell_grid = getattr(rho, "cell_grid", None)
        if rho_cell_grid is not None:
            cell_grid = rho_cell_grid
        else:
            raise ValueError("cell_grid must be provided either as an argument or as an attribute of the density.")

    if n_cartesian_grid is None:
        rho_grid_shape = getattr(rho, "grid_shape", None)
        if rho_grid_shape is not None:
            n_cartesian_grid = rho_grid_shape
        else:
            n_cartesian_grid = (32, 32, 32)

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

    if P is None:
        return np.zeros(0, dtype=float)
    return np.asarray(P, dtype=float)


def compute_projection_vector(
    rho: DensityFunction,
    basis_set: RIBasisSet,
    mode: str = "fft",
    n_cartesian_grid: int | tuple[int, int, int] | None = None,
    cell_grid: np.ndarray | None = None,
    print_progress_bar: bool = False,
) -> np.ndarray:
    """Compute projection coefficients P_i = <rho | chi_i>."""

    if mode == "cartesian":
        return compute_projection_vector_cartesian_grid(
            rho=rho,
            basis_set=basis_set,
            n_cartesian_grid=n_cartesian_grid,
            cell_grid=cell_grid,
        )

    if mode == "fft":
        return compute_projection_vector_native_grid(
            rho=rho,
            basis_set=basis_set,
            print_progress_bar=print_progress_bar,
        )

    if mode == "coulomb":
        return compute_projection_vector_coulomb_metric(
            rho=rho,
            basis_set=basis_set,
            print_progress_bar=print_progress_bar,
        )

    raise ValueError(f"Unknown projection mode: {mode!r}")

def solve_projections_coeffs(overlap_matrix: np.ndarray, projection_vector: np.ndarray) -> np.ndarray:
    """Solves the linear system S c = P to obtain the projection coefficients c_i."""
    M = np.asarray(overlap_matrix)
    b = np.asarray(projection_vector)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"overlap_matrix must be square, got shape {M.shape}")
    if b.ndim != 1 or b.shape[0] != M.shape[0]:
        raise ValueError(f"projection_vector must be 1D with length {M.shape[0]}, got shape {b.shape}")

    # Ensure symmetry (numerical noise may introduce tiny asymmetries)
    M_sym = 0.5 * (M + M.T.conj())
    # Cast near-real matrices to real for solver stability
    M_sym = np.asarray(np.real_if_close(M_sym, tol=1000))

    try:
        x = np.linalg.solve(M_sym, b)
    except np.linalg.LinAlgError:
        # Fallback to least-squares / pseudo-inverse for singular or rank-deficient matrices
        x, *_ = np.linalg.lstsq(M_sym, b, rcond=None)

    return np.asarray(x, dtype=float)
