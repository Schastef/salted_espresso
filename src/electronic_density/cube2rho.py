from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.io.cube import read_cube

from .types import DensityFunction


class CubeDict(TypedDict):
    atoms: Atoms
    data: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray


@dataclass
class RhoG:
    rho_g: np.ndarray
    G: np.ndarray
    grid_shape: tuple[int, int, int]


@dataclass
class PlaneWaveDensity:
    rho_g: np.ndarray
    G: np.ndarray

    def __call__(self, r: npt.NDArray[np.floating]) -> npt.NDArray[np.floating] | np.floating:
        N = len(self.rho_g)
        r = np.asarray(r, dtype=float)
        result = None

        if r.shape == (3,):
            result = np.sum(self.rho_g * np.exp(1j * (self.G @ r))) / N

        elif r.ndim == 2 and r.shape[1] == 3:
            result = np.exp(1j * (r @ self.G.T)) @ self.rho_g / N

        else:
            raise ValueError(f"r must have shape (3,) or (n, 3), got {r.shape}")

        result = np.real_if_close(result, tol=1000)

        if np.isreal(result).all():
            return result
        else:
            raise ValueError("Result is complex, but expected real. Check if G and rho_g are correct.")


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
    )


def load_rho_from_cube(path: Path) -> DensityFunction:
    cube_dict = load_cubefile(path)
    rhog = compute_rho_g(cube_dict["data"], cube_dict["spacing"])
    return PlaneWaveDensity(rhog.rho_g, rhog.G)



