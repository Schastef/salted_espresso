from __future__ import annotations

from enum import Enum
from typing import Literal, List

from matplotlib import axes
import numpy as np

from salted_espresso.ri_basis.core import RIBasis, RIBasisSet
from salted_espresso.electronic_density.types import DensityFunction


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

def _build_cartesian_grid(ri_basis: RIBasis, n_cartesian_grid: int, initial_r_max: float, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    if n_cartesian_grid < 8:
        raise ValueError(f"n_cartesian_grid must be >= 8, got {n_cartesian_grid}")

    #r_max = _radial_extent(ri_basis, initial_r_max=initial_r_max, radial_cutoff=cutoff)
    r_max = initial_r_max
    axes = np.linspace(-r_max, r_max, int(n_cartesian_grid), dtype=float)
    gridx, gridy, gridz = np.meshgrid(axes, axes, axes, indexing='ij')# (N,N,N)
    dx = axes[1] - axes[0]
    w = np.full_like(axes, dx) #one length element --> will need to multiply weights of all coordintes together to get volume element
    w[0] *= 0.5
    w[-1] *= 0.5
    return axes, w



def _eval_basis_on_cartesian_grid(
    ri_basis: RIBasis,
    axes: np.ndarray,
) -> np.ndarray:
    """
    Evaluate all basis functions on a 3D Cartesian grid.

    Returns
    -------
    values : np.ndarray, shape (n_basis, N, N, N)
        values[i, ix, iy, iz] = phi_i(x, y, z)
    """
    N = len(axes)
    gridx, gridy, gridz = np.meshgrid(axes, axes, axes, indexing='ij')  # (N, N, N)

    r_grid = np.sqrt(gridx**2 + gridy**2 + gridz**2)          # (N, N, N)
    grid_2d = np.vstack(list(map(np.ravel, [gridx, gridy, gridz]))).T
    # theta_grid = np.arccos(np.clip(gridz / np.where(r_grid > 1e-12, r_grid, 1.0), -1.0, 1.0))
    # phi_grid = np.arctan2(gridy, gridx)

    n_basis = len(ri_basis)
    values = np.zeros((n_basis, N, N, N), dtype=float)

    for idx in range(n_basis):
        n, l, m = ri_basis.running_to_lexographic_index(idx)

        # Radial part: evaluate on flat r array, then reshape
        radial_func = ri_basis.radial_funcs.radials[
            ri_basis.radial_funcs.lexographic_to_running_index((n, l))
        ]
        R_vals = radial_func(r_grid.ravel()).reshape(N, N, N)  # (N, N, N)

        # Real spherical harmonic Y_l^m evaluated on the grid
        sph_idx = ri_basis.angular_funcs.lexographic_to_running_index((l, m))  # column index for (l,m)
        Y_vals = ri_basis.angular_funcs(grid_2d)[:, sph_idx].reshape(N, N, N)   # (N, N, N)
        #print(np.shape(Y_vals))

        values[idx] = R_vals * Y_vals

    return values

def _overlap_radial_block(radials_by_l: np.ndarray, r: np.ndarray, w: np.ndarray) -> np.ndarray:
    weights = (r ** 2) * w
    return (radials_by_l * weights[None, :]) @ radials_by_l.T


def _overlap_cartesian_block(
    basis_values: np.ndarray,
    dV: float,
) -> np.ndarray:
    """
    S_ij = integral phi_i(r) phi_j(r) d^3r
         ~ sum_{x,y,z} phi_i(x,y,z) * phi_j(x,y,z) * dV

    Parameters
    ----------
    basis_values : (n_basis, N, N, N)
    dV           : scalar volume element dx*dy*dz

    Returns
    -------
    S : (n_basis, n_basis)
    """
    n_basis = basis_values.shape[0]
    flat = basis_values.reshape(n_basis, -1)   # (n_basis, N^3)
    return (flat @ flat.T) * dV


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


def _coulomb_cartesian_block(
    basis_values: np.ndarray,
    axes: np.ndarray,
    dV: float,
    coulomb_eps: float = 1e-10,
) -> np.ndarray:
    """
    V_ij = integral integral phi_i(r) * (1/|r-r'|) * phi_j(r') d^3r d^3r'

    Evaluated via direct double sum — expensive but exact on the grid.
    For large grids prefer a Fourier-space Poisson solver instead.

    Parameters
    ----------
    basis_values : (n_basis, N, N, N)
    axes         : (N,)  1-D coordinate array (same for x, y, z)
    dV           : scalar volume element dx*dy*dz
    coulomb_eps  : regularisation for |r-r'| -> 0

    Returns
    -------
    V : (n_basis, n_basis)
    """
    N = len(axes)
    n_basis = basis_values.shape[0]
    n_pts = N ** 3

    gridx, gridy, gridz = np.meshgrid(axes, axes, axes, indexing='ij')
    coords = np.stack(
        [gridx.ravel(), gridy.ravel(), gridz.ravel()], axis=1
    )  # (N^3, 3)

    # Pairwise distance matrix  |r - r'|,  shape (N^3, N^3)
    diff = coords[:, None, :] - coords[None, :, :]   # (N^3, N^3, 3)
    dist = np.sqrt((diff**2).sum(axis=-1))            # (N^3, N^3)
    with np.errstate(divide='ignore'):
        kernel = np.where(dist > coulomb_eps, 1.0 / dist, 0.0)  # (N^3, N^3)

    flat = basis_values.reshape(n_basis, n_pts)  # (n_basis, N^3)

    # V_ij = dV^2 * phi_i^T @ kernel @ phi_j
    rho = flat @ kernel   # (n_basis, N^3)  — inner integral over r'
    V = (rho @ flat.T) * (dV ** 2)
    return V


def _compute_single_basis_overlap(
        ri_basis: RIBasis,
        method: MetricMethod,
        *,
        n_radial_grid: int = 512,
        initial_r_max: float = 8.0,
        cutoff: float = 1e-10,
) -> np.ndarray:
    """
    Computes the overlap matrix for a single RIBasis instance.
    This is a helper function for compute_overlap.
    """
    n_max_by_l = ri_basis.n_max
    l_max = ri_basis.l_max
    n_basis = len(ri_basis)

    if n_basis == 0:
        return np.zeros((0, 0), dtype=float)

    r, w = _build_radial_grid(
        ri_basis,
        n_radial_grid=n_radial_grid,
        initial_r_max=initial_r_max,
        cutoff= cutoff,
    )

    # Precompute radial values grouped by angular momentum l:
    # shape per l is (n_max[l], n_radial_grid)
    radial_values_by_l: dict[int, np.ndarray] = {}
    for l in range(l_max + 1):
        n_l = n_max_by_l[l]
        if n_l == 0:
            continue
        funcs = [
            ri_basis.radial_funcs.radials[ri_basis.radial_funcs.lexographic_to_running_index((n, l))]
            for n in range(n_l)
        ]
        radial_values_by_l[l] = np.stack([func(r) for func in funcs], axis=0)

    radial_blocks: dict[int, np.ndarray] = {}
    for l in range(l_max + 1):
        if l not in radial_values_by_l:
            continue
        if method == MetricMethod.OVERLAP:
            radial_blocks[l] = _overlap_radial_block(radial_values_by_l[l], r=r, w=w)
        else:
            radial_blocks[l] = _coulomb_radial_block(radial_values_by_l[l], l=l, r=r, w=w)

    matrix = np.zeros((n_basis, n_basis), dtype=float)

    for l in range(l_max + 1):
        if l not in radial_blocks:
            continue
        radial_block = radial_blocks[l]
        n_l = n_max_by_l[l]
        for m in range(-l, l + 1):
            indices = [ri_basis.lexographic_to_running_index((n, l, m)) for n in range(n_l)]
            for i_local, i_global in enumerate(indices):
                matrix[i_global, indices] = radial_block[i_local, :]

    # Remove tiny asymmetries from numerical integration.
    matrix = 0.5 * (matrix + matrix.T)
    return matrix


def _compute_single_basis_overlap_cart(
    ri_basis: RIBasis,
    method: MetricMethod,
    *,
    n_cartesian_grid: int = 64,
    initial_r_max: float = 8.0,
    cutoff: float = 1e-10,
    coulomb_eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute the overlap (or Coulomb) matrix for a single RIBasis instance
    using a full 3-D Cartesian integration grid.
    """
    n_basis = len(ri_basis)
    if n_basis == 0:
        return np.zeros((0, 0), dtype=float)

    # ------------------------------------------------------------------ grid
    axes, w = _build_cartesian_grid(
        ri_basis,
        n_cartesian_grid=n_cartesian_grid,
        initial_r_max=initial_r_max,
        cutoff=cutoff,
    )
    dx = axes[1] - axes[0]
    dV = dx ** 3   # uniform cubic volume element

    # ------------------------------------------------- evaluate basis on grid
    grid = np.meshgrid(axes, axes, axes, indexing='ij')  # (N, N, N)
    grid_2d = np.vstack(list(map(np.ravel, grid))).T  # (N^3, 3)
    basis_values = ri_basis(grid_2d)  # (n_basis, N, N, N)

    # --------------------------------------------------------- build matrix
    if method == MetricMethod.OVERLAP:
        matrix = _overlap_cartesian_block(basis_values, dV)
    else:
        matrix = _coulomb_cartesian_block(basis_values, axes, dV, coulomb_eps=coulomb_eps)

    # Symmetrise to remove tiny numerical asymmetries.
    matrix = 0.5 * (matrix + matrix.T)
    return matrix
