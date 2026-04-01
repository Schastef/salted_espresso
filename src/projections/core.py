from __future__ import annotations

from typing import Literal

import numpy as np

from .utils import (_parse_method,
                    _radial_extent,
                    _build_radial_grid,
                    _compute_single_basis_overlap,
                    MetricMethod)

from salted_espresso.ri_basis.core import RIBasis, RIBasisSet
from salted_espresso.electronic_density.types import DensityFunction


def compute_overlap(
    ri_basis_set: RIBasis | RIBasisSet,
    method: MetricMethod | Literal["overlap", "coulomb", "s", "j"] | str,
    *,
    n_radial_grid: int = 512,
    initial_r_max: float = 8.0,
    radial_cutoff: float = 1e-10,
) -> np.ndarray:
    """Compute RI metric matrix in the basis lexicographic order (n,l,m).

    The returned matrix has shape ``(len(ri_basis), len(ri_basis))``.
    Only basis functions with equal ``(l, m)`` couple, which yields a block
    structure in the final matrix.


    Parameters:
    -----------

    ri_basis_set: RIBasis or RIBasisSet
        The RI basis or basis set for which to compute the metric.
    method: MetricMethod or str
        The metric to compute. Supported values are:
        - "overlap" or "s": Compute the overlap metric S_ij = <i|j>.
        - "coulomb" or "j": Compute the Coulomb metric J_ij = <i|1/r|j>.
    n_radial_grid: int
        The number of points in the radial quadrature grid. Must be >= 8.
    initial_r_max: float
        Initial guess for the maximum radius of the quadrature grid. The actual
        maximum radius will be adjusted to ensure the tail of the radial functions is below the specified cutoff.
    radial_cutoff: float
        The cutoff value for the tail of the radial functions. The quadrature grid will be extended until the maximum
        value of any radial function at the grid's maximum radius is below this cutoff.
    """

    metric_method = _parse_method(method)

    if isinstance(ri_basis_set, RIBasis):
        return _compute_single_basis_overlap(
            ri_basis_set,
            metric_method,
            n_radial_grid=n_radial_grid,
            initial_r_max=initial_r_max,
            radial_cutoff=radial_cutoff,
        )

    # This is a simplified version for RIBasisSet that only computes diagonal blocks.
    # A full implementation would require off-diagonal blocks (inter-atomic overlaps).
    s_blocks = [
        _compute_single_basis_overlap(
            basis,
            metric_method,
            n_radial_grid=n_radial_grid,
            initial_r_max=initial_r_max,
            radial_cutoff=radial_cutoff,
        )
        for basis in ri_basis_set
    ]
    from scipy.linalg import block_diag
    return block_diag(*s_blocks)


def compute_condition_number(overlap_matrix: np.ndarray, tol: float = 1e-10) -> float:
    """Compute the condition number of the overlap matrix.

    The condition number is defined as the ratio of the largest to the smallest
    Eigenvalue of the overlap matrix. A large condition number indicates that the
    basis set is nearly linearly dependent, which can lead to numerical instability in projections.

    Parameters:
    -----------
    overlap_matrix: np.ndarray
        The overlap matrix for which to compute the condition number. Must be square and symmetric.
    """

    eigenvalues = np.linalg.eigvalsh(overlap_matrix)

    # Clamp small negative eigenvalues (numerical noise) to tiny positive value
    eigenvalues = np.maximum(eigenvalues, tol)

    max_eig = np.max(eigenvalues)
    min_eig = np.min(eigenvalues)

    return max_eig / min_eig


def compute_projection_coefficients(
    target_density: DensityFunction,
    ri_basis_set: RIBasis | RIBasisSet,
    *,
    n_radial_grid: int = 512,
    initial_r_max: float = 8.0,
    radial_cutoff: float = 1e-10,
) -> np.ndarray:
    """
    Computes the projection coefficients b_i = <target_density|ri_basis_i>.

    This function computes the projection of a target density function onto each
    basis function in the given RI basis or basis set. The projection is
    calculated as the integral of the product of the density and the basis
    function over all space.

    For performance, this implementation evaluates the density and the basis set
    on a single universal grid and then performs the integration using vectorized
    operations, avoiding per-atom loops.

    Parameters:
    -----------
    target_density: DensityFunction
        The target electronic density to be projected.
    ri_basis_set: RIBasis or RIBasisSet
        The RI basis or basis set onto which to project the target density.
    n_radial_grid: int
        Number of points for the radial quadrature grid.
    initial_r_max: float
        Initial guess for the maximum radius of the grid.
    radial_cutoff: float
        Cutoff for the tail of radial functions to determine grid extent.

    Returns:
    --------
    np.ndarray
        A vector containing the projection coefficients <rho|chi_i>.
    """
    if isinstance(ri_basis_set, RIBasis):
        bases = [ri_basis_set]
    else:  # RIBasisSet
        bases = ri_basis_set

    # 1. Determine the maximum radial extent required for any basis function
    max_r = 0.0
    for basis in bases:
        max_r = max(max_r, _radial_extent(basis, initial_r_max, radial_cutoff))

    # 2. Build a single universal grid for the integration
    # We use a temporary dummy basis to build the grid.
    r, w = _build_radial_grid(bases[0], n_radial_grid, max_r, radial_cutoff)

    # Create a 3D grid of points for evaluation
    grid_points = np.zeros((len(r), 3))
    grid_points[:, 0] = r

    # 3. Evaluate the density and the entire basis set ONCE on this grid
    print("Evaluating density on the universal grid...")
    density_on_grid = target_density(grid_points)

    print("Evaluating basis set on the universal grid...")
    basis_on_grid = ri_basis_set(grid_points) # Shape: (n_grid_points, n_total_basis_funcs)

    # 4. Perform the integration in a vectorized manner
    # The integral is integral(rho(r) * chi_i(r) * 4pi * r^2 dr)
    # This is simplified for s-functions (l=0), which is what we project onto.
    # The basis_on_grid contains all chi_i(r), so we can do this with a matrix-vector product.

    # The integration weights for the radial integral
    integrand_weights = 4 * np.pi * r**2 * w

    # The dot product sums over the grid points (the first axis)
    # (n_total_basis_funcs, n_grid_points) @ (n_grid_points,) -> (n_total_basis_funcs,)
    print("Performing vectorized integration...")
    coeffs = basis_on_grid.T @ (density_on_grid * integrand_weights)

    # The current implementation of basis functions only returns non-zero values for s-orbitals
    # when the input is purely radial. If it were a full 3D evaluation, we would need to
    # filter for the s-orbitals here. For now, the zeros are implicitly handled.

    print("Done.")
    return coeffs


def solve_projection_equations(
    overlap_matrix: np.ndarray,
    projection_coefficients: np.ndarray
) -> np.ndarray:
    """
    Solves the linear system Mc = b for the expansion coefficients.

    Given the overlap matrix M (S_ij = <chi_i|chi_j>) and the projection
    vector b (b_i = <rho|chi_i>), this function solves the linear system
    of equations Mc = b to find the coefficients c that represent the
    projection of the density rho onto the basis {chi_i}.

    Parameters:
    -----------
    overlap_matrix: np.ndarray
        The overlap matrix M of the basis set.
    projection_coefficients: np.ndarray
        The vector b of projection coefficients.

    Returns:
    --------
    np.ndarray
        The vector of expansion coefficients c.
    """
    return np.linalg.solve(overlap_matrix, projection_coefficients)


_sentinel = object()  # Unique sentinel value for uninitialized variables

def compute_projectability(
    rho: DensityFunction,
    basis: RIBasis | RIBasisSet,
    expansion_coefficients: np.ndarray | object = _sentinel,
    *,
    n_radial_grid: int = 512,
    initial_r_max: float = 8.0,
    radial_cutoff: float = 1e-10,
    rcond: float = 1e-12,
    clip_tolerance: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Calculate projectability P = ||rho_proj||^2 / ||rho||^2.

    This implementation computes an explicit weighted least-squares projection on
    the same quadrature grid used for the norm, so it is consistent for
    non-orthonormal bases and numerically stable.

    Returns:
    --------
    P: float
        Projectability
    coeffs: np.ndarray
        The expansion coefficients of the projection of rho onto the basis.
    """
    if isinstance(basis, RIBasis):
        bases = [basis]
    else:
        bases = list(basis)

    if not bases:
        return 0.0

    max_r = 0.0
    for single_basis in bases:
        max_r = max(max_r, _radial_extent(single_basis, initial_r_max, radial_cutoff))

    r, w = _build_radial_grid(bases[0], n_radial_grid, max_r, radial_cutoff)
    grid_points = np.zeros((len(r), 3), dtype=float)
    grid_points[:, 0] = r

    rho_on_grid = np.asarray(rho(grid_points), dtype=float)
    basis_on_grid = np.asarray(basis(grid_points), dtype=float)

    if basis_on_grid.ndim != 2 or basis_on_grid.shape[0] != grid_points.shape[0]:
        raise ValueError(
            "Basis evaluation must return shape (N_grid, N_basis), "
            f"got {basis_on_grid.shape} for N_grid={grid_points.shape[0]}."
        )

    weights = np.clip(4.0 * np.pi * (r ** 2) * w, a_min=0.0, a_max=None)
    sqrt_w = np.sqrt(weights)

    y_w = rho_on_grid * sqrt_w
    a_w = basis_on_grid * sqrt_w[:, None]

    if expansion_coefficients is _sentinel:
        coeffs, *_ = np.linalg.lstsq(a_w, y_w, rcond=rcond)
    else:
        coeffs = np.asarray(expansion_coefficients, dtype=float).reshape(-1)
        if coeffs.shape[0] != basis_on_grid.shape[1]:
            raise ValueError(
                "expansion_coefficients must have length equal to total basis size "
                f"({basis_on_grid.shape[1]}), got {coeffs.shape[0]}."
            )

    rho_proj_w = a_w @ coeffs

    numerator = float(np.dot(rho_proj_w, rho_proj_w))
    denominator = float(np.dot(y_w, y_w))

    if denominator <= 0.0:
        return 0.0

    projectability = numerator / denominator

    # Clamp tiny floating-point excursions outside [0, 1].
    if -clip_tolerance < projectability < 0.0:
        projectability = 0.0
    elif 1.0 < projectability < 1.0 + clip_tolerance:
        projectability = 1.0

    return float(projectability), coeffs
