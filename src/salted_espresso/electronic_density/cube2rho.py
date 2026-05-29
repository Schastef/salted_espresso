from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, TypedDict
from collections.abc import Iterable, Iterator
import warnings

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.io.cube import read_cube

from .types import DensityFunction


class CubeDict(TypedDict):
    atoms: Atoms
    data: npt.NDArray[np.floating]
    origin: npt.NDArray[np.floating]
    spacing: npt.NDArray[np.floating]


@dataclass
class RhoG:
    rho_g: np.ndarray
    G: np.ndarray
    grid_shape: tuple[int, int, int]
    cell_grid: np.ndarray


@dataclass
class PlaneWaveDensity:
    rho_g: np.ndarray
    G: np.ndarray
    cell_grid: np.ndarray = None
    grid_shape: tuple[int, int, int] = None
    # Upper bound for temporary arrays used during rho(r) evaluation.
    max_batch_memory_mb: float = 64.0
    # Imaginary residual tolerated before treating the result as inconsistent.
    imag_abs_tol: float = 1e-6
    imag_rel_tol: float = 1e-10
    # Behavior when imag(rho) exceeds tolerance: raise, warn+coerce, or coerce.
    complex_result_policy: Literal["raise", "warn", "coerce"] = "warn"
    _warned_complex_result: bool = field(default=False, init=False, repr=False)

    def _batch_chunk_size(self) -> int:
        """Number of G-vectors processed per streamed dot-product chunk."""
        n_g = len(self.rho_g)
        if n_g == 0:
            return 1

        target_bytes = max(int(self.max_batch_memory_mb * 1024 * 1024), 16)
        # Per element we allocate a real phase and a complex exponential.
        approx_bytes_per_g = np.dtype(np.float64).itemsize + np.dtype(np.complex128).itemsize
        return max(min(target_bytes // max(approx_bytes_per_g, 1), n_g), 1)

    def _evaluate_single_point(self, point: npt.NDArray[np.floating]) -> np.complex128:
        n_terms = len(self.rho_g)
        if n_terms == 0:
            return np.complex128(0.0)

        chunk_size = self._batch_chunk_size()
        accum = np.complex128(0.0)

        for start in range(0, n_terms, chunk_size):
            stop = min(start + chunk_size, n_terms)
            phase = self.G[start:stop] @ point
            accum += np.dot(self.rho_g[start:stop], np.exp(1j * phase))

        return accum / n_terms

    def _to_real_scalar(self, value: np.complexfloating) -> np.floating:
        real_part = float(np.real(value))
        imag_abs = float(abs(np.imag(value)))

        tol = self.imag_abs_tol + self.imag_rel_tol * max(abs(real_part), 1.0)
        if imag_abs <= tol:
            return np.float64(real_part)

        if self.complex_result_policy == "raise":
            raise ValueError(
                "Result is complex beyond tolerance: "
                f"|Im(rho)|={imag_abs:.3e}, tolerance={tol:.3e}. "
                "Check reciprocal-space symmetry rho(-G)=conj(rho(G)) or relax tolerances."
            )

        if self.complex_result_policy == "warn" and not self._warned_complex_result:
            warnings.warn(
                "rho(r) has a non-negligible imaginary residual; returning the real part. "
                f"|Im(rho)|={imag_abs:.3e}, tolerance={tol:.3e}. "
                "Set complex_result_policy='raise' for strict behavior.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_complex_result = True

        if self.complex_result_policy not in {"warn", "coerce", "raise"}:
            raise ValueError(
                f"Invalid complex_result_policy={self.complex_result_policy!r}; "
                "expected one of {'raise', 'warn', 'coerce'}."
            )

        return np.float64(real_part)

    def _iter_values(self, points: Iterable[object]) -> Iterator[np.floating]:
        for index, point in enumerate(points):
            point_arr = np.asarray(point, dtype=float)
            if point_arr.shape != (3,):
                raise ValueError(f"Each streamed point must have shape (3,), got {point_arr.shape} at index {index}")
            yield self._to_real_scalar(self._evaluate_single_point(point_arr))

    def __call__(
        self,
        r: npt.NDArray[np.floating] | Iterable[object],
    ) -> npt.NDArray[np.floating] | np.floating | Iterator[np.floating]:
        if isinstance(r, np.ndarray):
            r_arr = np.asarray(r, dtype=float)
            if r_arr.shape == (3,):
                return self._to_real_scalar(self._evaluate_single_point(r_arr))

            if r_arr.ndim == 2 and r_arr.shape[1] == 3:
                result = np.empty(r_arr.shape[0], dtype=float)
                for index, point in enumerate(r_arr):
                    result[index] = self._to_real_scalar(self._evaluate_single_point(point))
                return result

            raise ValueError(f"r must have shape (3,) or (n, 3), got {r_arr.shape}")

        if isinstance(r, Iterable):
            return self._iter_values(r)

        r_arr = np.asarray(r, dtype=float)
        if r_arr.shape == (3,):
            return self._to_real_scalar(self._evaluate_single_point(r_arr))
        raise ValueError(f"r must have shape (3,) or (n, 3), got {r_arr.shape}")

    def memory_usage_mb(self) -> float:
        """Estimation of memory usage in megabytes to store the plane wave density data."""
        return (self.rho_g.nbytes + self.G.nbytes) / (1024 * 1024)


def load_cubefile(path: Path) -> CubeDict:
    with open(path, "r") as f:
        raw = read_cube(f)

    return CubeDict(
        atoms=raw["atoms"],
        data=raw["data"],
        origin=raw["origin"],
        spacing=raw["spacing"],
    )


def compute_rho_g(rho_r: npt.NDArray[np.floating], spacing: npt.NDArray[np.floating]) -> RhoG:
    """
    Apply a fast Fourier transform to an electron density on a regular real-space grid
    to obtain rho_G and the corresponding reciprocal vectors G.

    The .cube file contains the electronic density in real space, rho(r), sampled on a
    regular grid. Using the fast Fourier transform, we obtain the coefficients of the
    electronic density in reciprocal space, rho_G. From these, we can reconstruct an
    analytical expression for the electronic density in real space:

        rho(r) = sum_G rho_G * exp(i * G · r)

    Args:
        rho_r:
            Electronic density in real space, sampled on a regular 3D grid
            of shape (nx, ny, nz).

        spacing:
            Cube-grid spacing information.

            Expected formats:
            - shape (3,): orthorhombic spacing [dx, dy, dz]
            - shape (3, 3): grid step vectors, one row per grid direction

    Returns:
        RhoG:
            A dataclass containing:
            - rho_g: flattened FFT coefficients, shape (nG,)
            - G: flattened Cartesian G vectors, shape (nG, 3)
            - grid_shape: original FFT grid shape (nx, ny, nz)
    """

    if rho_r.ndim != 3:
        raise ValueError(f"rho_r must be a 3D array, got shape {rho_r.shape}")

    grid_shape = rho_r.shape
    nx, ny, nz = grid_shape

    spacing = np.asarray(spacing, dtype=float)

    if spacing.shape == (3,):
        step_matrix = np.diag(spacing)
    elif spacing.shape == (3, 3):
        step_matrix = spacing
    else:
        raise ValueError(
            f"spacing must have shape (3,) or (3, 3), got {spacing.shape}"
        )

    rho_G_grid = np.fft.fftn(rho_r)

    # Integer FFT frequency indices
    fx = np.fft.fftfreq(nx) * nx
    fy = np.fft.fftfreq(ny) * ny
    fz = np.fft.fftfreq(nz) * nz

    # Full cell vectors, not step vectors
    cell_matrix = step_matrix.copy()
    cell_matrix[0] *= nx
    cell_matrix[1] *= ny
    cell_matrix[2] *= nz

    # Reciprocal basis of the FULL cell
    reciprocal_basis = 2.0 * np.pi * np.linalg.inv(cell_matrix).T

    I, J, K = np.meshgrid(fx, fy, fz, indexing="ij")
    index_grid = np.stack((I, J, K), axis=-1)

    # Cartesian G vectors
    G_grid = index_grid @ reciprocal_basis

    G_flat = G_grid.reshape(-1, 3)
    rho_G_flat = rho_G_grid.reshape(-1)

    return RhoG(
        rho_g=rho_G_flat,
        G=G_flat,
        grid_shape=grid_shape,
        cell_grid=cell_matrix,
    )


def load_rho_from_cube(path: Path) -> DensityFunction:
    cube_dict = load_cubefile(path)
    rhog = compute_rho_g(cube_dict["data"], cube_dict["spacing"])
    return PlaneWaveDensity(rhog.rho_g, rhog.G, cell_grid=rhog.cell_grid, grid_shape=rhog.grid_shape)
