from __future__ import annotations

from enum import Enum
from typing import Literal, List

import numpy as np

from ri_basis.core import RIBasis, RIBasisSet
from electronic_density.types import DensityFunction


class MetricMethod(str, Enum):
    OVERLAP = "overlap"
    COULOMB = "coulomb"


def _parse_method(method: MetricMethod | str) -> MetricMethod:
    if isinstance(method, MetricMethod):
        return method

    normalized = str(method).strip().lower()
    aliases = {
        "overlap": MetricMethod.OVERLAP,
        "s": MetricMethod.OVERLAP,
        "coulomb": MetricMethod.COULOMB,
        "j": MetricMethod.COULOMB,
    }
    if normalized not in aliases:
        raise ValueError(
            "Unknown metric method "
            f"{method!r}. Supported methods are: {sorted(aliases.keys())}."
        )
    return aliases[normalized]


def _radial_extent(ri_basis: RIBasis, initial_r_max: float, radial_cutoff: float) -> float:
    r_max = max(float(initial_r_max), 1e-6)
    if len(ri_basis.radial_funcs.radials) == 0:
        return r_max

    for _ in range(8):
        probe = np.array([r_max], dtype=float)
        tail = max(abs(func(probe)[0]) for func in ri_basis.radial_funcs.radials)
        if tail <= radial_cutoff:
            break
        r_max *= 1.75

    return r_max


def _build_radial_grid(ri_basis: RIBasis, n_radial_grid: int, initial_r_max: float, radial_cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    if n_radial_grid < 8:
        raise ValueError(f"n_radial_grid must be >= 8, got {n_radial_grid}")

    r_max = _radial_extent(ri_basis, initial_r_max=initial_r_max, radial_cutoff=radial_cutoff)
    r = np.linspace(0.0, r_max, int(n_radial_grid), dtype=float)
    if r.size > 0:
        r[0] = 1e-12

    dr = np.diff(r)
    w = np.empty_like(r)
    w[0] = 0.5 * dr[0]
    w[-1] = 0.5 * dr[-1]
    if len(r) > 2:
        w[1:-1] = 0.5 * (dr[1:] + dr[:-1])
    return r, w


def _overlap_radial_block(radials_by_l: np.ndarray, r: np.ndarray, w: np.ndarray) -> np.ndarray:
    weights = (r ** 2) * w
    return (radials_by_l * weights[None, :]) @ radials_by_l.T


def _coulomb_radial_block(radials_by_l: np.ndarray, l: int, r: np.ndarray, w: np.ndarray) -> np.ndarray:
    r_i = r[:, None]
    r_j = r[None, :]
    r_min = np.minimum(r_i, r_j)
    r_max = np.maximum(r_i, r_j)

    # Avoid singularity at r=0: use l>0 safe formula and add epsilon for l=0
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = np.where(
            r_max > 1e-10,
            (r_min ** l) / (r_max ** (l + 1)),
            0.0
        )

    weighted = radials_by_l * ((r ** 2) * w)[None, :]
    radial_double = weighted @ kernel @ weighted.T
    prefactor = 4.0 * np.pi / (2 * l + 1)
    return prefactor * radial_double



def _compute_single_basis_overlap(
    ri_basis: RIBasis,
    method: MetricMethod,
    *,
    n_radial_grid: int = 512,
    initial_r_max: float = 8.0,
    radial_cutoff: float = 1e-10,
) -> np.ndarray:
    """
    Computes the overlap matrix for a single RIBasis instance.
    This is a helper function for compute_overlap.
    """
    n_max = ri_basis.n_max
    l_max = ri_basis.l_max
    n_basis = len(ri_basis)

    if n_basis == 0:
        return np.zeros((0, 0), dtype=float)

    r, w = _build_radial_grid(
        ri_basis,
        n_radial_grid=n_radial_grid,
        initial_r_max=initial_r_max,
        radial_cutoff=radial_cutoff,
    )

    # Precompute radial values grouped by angular momentum l:
    # shape per l is (n_max, n_radial_grid)
    radial_values_by_l: dict[int, np.ndarray] = {}
    for l in range(l_max + 1):
        funcs = [ri_basis.radial_funcs.radials[n * (l_max + 1) + l] for n in range(n_max)]
        radial_values_by_l[l] = np.stack([func(r) for func in funcs], axis=0)

    radial_blocks: dict[int, np.ndarray] = {}
    for l in range(l_max + 1):
        if method == MetricMethod.OVERLAP:
            radial_blocks[l] = _overlap_radial_block(radial_values_by_l[l], r=r, w=w)
        else:
            radial_blocks[l] = _coulomb_radial_block(radial_values_by_l[l], l=l, r=r, w=w)

    matrix = np.zeros((n_basis, n_basis), dtype=float)

    for l in range(l_max + 1):
        radial_block = radial_blocks[l]
        for m in range(-l, l + 1):
            indices = [ri_basis.lexographic_to_running_index((n, l, m)) for n in range(n_max)]
            for i_local, i_global in enumerate(indices):
                matrix[i_global, indices] = radial_block[i_local, :]

    # Remove tiny asymmetries from numerical integration.
    matrix = 0.5 * (matrix + matrix.T)
    return matrix


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
