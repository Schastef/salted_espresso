from __future__ import annotations

from enum import Enum
from typing import Literal, List

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
    n_max_by_l = ri_basis.n_max
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
