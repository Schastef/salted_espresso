#!/usr/bin/env python3
"""
cube2cube: Project electronic density to a basis set and reconstruct.

This script performs the following operations:
1. Load electronic density from a .cube file
2. Load basis set specifications from a .yaml or .json file
3. Calculate projection vector using FFT-based integration
4. Calculate overlap matrix for the basis set
5. Solve for projection coefficients
6. Use basis set span to reconstruct projected electron density
7. Write reconstructed density back to a .cube file

Usage:
    python cube2cube.py <input.cube> <basis_spec.json> <output.cube>

The basis specification file should contain definitions for each atomic species.
Example structure for a single atom at the origin:

    {
        "H": {
            "n_max": 1,
            "l_max": 0,
            "radial_method": "gaussian",
            "angular_method": "real_spherical",
            "radial_params": {"alphas": [1.0]}
        }
    }
"""

import argparse
import os
import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io.cube import write_cube

from salted_espresso.electronic_density.cube2rho import (
    load_cubefile,
    load_rho_from_cube,
    compute_rho_g,
)
from salted_espresso.projections.core import (
    compute_projection_vector_FFT,
    compute_overlap_matrix,
    solve_projections_coeffs,
)
from salted_espresso.ri_basis.loader import load_basis_set
from salted_espresso.ri_basis.types import CutoffType

def reconstruct_density_on_grid(
    rho_function,
    grid_shape: tuple,
    cell_grid: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the projected density on the original grid.

    Args:
        rho_function: Callable representing rho
        grid_shape: Shape of the original grid (nx, ny, nz)
        cell_grid: Cell vectors as (3, 3) array
        origin: Origin of the grid in Cartesian coordinates

    Returns:
        Reconstructed density on the grid as (nx, ny, nz) array
    """
    nx, ny, nz = grid_shape

    # Create fractional coordinate grid
    f1 = np.linspace(0, 1, nx, endpoint=False)
    f2 = np.linspace(0, 1, ny, endpoint=False)
    f3 = np.linspace(0, 1, nz, endpoint=False)
    F1, F2, F3 = np.meshgrid(f1, f2, f3, indexing="ij")
    frac_pts = np.stack((F1.ravel(), F2.ravel(), F3.ravel()), axis=-1)

    # Convert to Cartesian coordinates
    points = origin + frac_pts @ np.asarray(cell_grid, dtype=float)

    # Evaluate the reconstructed density at all grid points
    rho_reconstructed = rho_function(points)

    # Reshape back to grid
    return rho_reconstructed.reshape(grid_shape)


def cube2cube(
    input_cube: Path,
    basis_spec: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Project electronic density to a basis set and reconstruct.

    Args:
        input_cube: Path to input .cube file
        basis_spec: Path to basis set specification (.json or .yaml)
        output_dir: Path to directory where to store outputs
        overwrite: Whether to overwrite existing output file
    """
    input_cube = Path(input_cube)
    basis_spec = Path(basis_spec)
    output_dir = Path(output_dir)

    if not input_cube.exists():
        raise FileNotFoundError(f"Input cube file not found: {input_cube}")
    if not basis_spec.exists():
        raise FileNotFoundError(f"Basis specification file not found: {basis_spec}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory file already exists: {output_dir}. Use --overwrite to replace."
        )

    print(f"Loading electron density from: {input_cube}")
    # Step 1: Load original density
    rho_original = load_rho_from_cube(input_cube)
    cube_dict = load_cubefile(input_cube)

    print(f"Grid shape: {rho_original.grid_shape}")
    print(f"Cell vectors:\n{rho_original.cell_grid}")

    # Step 3: Load basis set
    print(f"Loading basis set from: {basis_spec}")
    basis_set = load_basis_set(
        input_cube,
        str(basis_spec),
        cutoff=CutoffType.FIRST_NEIGHBOURS,
        order_by_species=False
    )

    # Set cell vectors from the original density if not already set
    # (needed for non-cubic cells or when structure file has no cell info)
    # First set on the RIBasisSet object itself for compute_overlap_matrix
    basis_set.cell_vectors = rho_original.cell_grid.copy()
    # Also set on individual ribases
    for ribasis in basis_set.ribases:
        ribasis.cell_vectors = rho_original.cell_grid.copy()

    total_basis_funcs = sum(len(b) for b in basis_set)
    print(f"Total basis functions: {total_basis_funcs}")

    # Step 4: Calculate overlap matrix
    print("Computing overlap matrix...")
    overlap_matrix = compute_overlap_matrix(basis_set, print_progress_bar=True)
    print(f"Overlap matrix shape: {overlap_matrix.shape}")
    print(f"Overlap matrix condition number: {np.linalg.cond(overlap_matrix):.2e}")

    # Step 5: Calculate projection vector using FFT grid
    print("Computing projection vector using FFT integration...")
    projection_vector = compute_projection_vector_FFT(rho_original, basis_set, print_progress_bar=True)
    print(f"Projection vector shape: {projection_vector.shape}")
    print(f"Projection vector norm: {np.linalg.norm(projection_vector):.6e}")

    # Step 6: Solve for projection coefficients
    print("Solving for projection coefficients...")
    projection_coeffs = solve_projections_coeffs(overlap_matrix, projection_vector)
    print(f"Projection coefficients shape: {projection_coeffs.shape}")
    print(f"Projection coefficients norm: {np.linalg.norm(projection_coeffs):.6e}")

    # Step 7: Reconstruct density using basis set span
    print("Reconstructing projected electron density...")
    rho_reconstructed_func = basis_set.span(projection_coeffs)

    # Step 8: Evaluate on original grid
    print("Evaluating reconstructed density on original grid...")
    rho_reconstructed_grid = reconstruct_density_on_grid(
        rho_reconstructed_func,
        rho_original.grid_shape,
        rho_original.cell_grid,
        rho_original.origin,
    )


    # Writing output files
    cube_file_path = os.path.join(output_dir, "rho.cube")
    overlap_matrix_path = os.path.join(output_dir, "overlap_matrix.npy")
    projection_vector_path = os.path.join(output_dir, "projection_vector.npy")
    coefficients_path = os.path.join(output_dir, "coefficients.npy")

    # Step 9: Write to cube file
    print(f"Writing reconstructed density to: {cube_file_path}")

    # Prepare atoms object
    atoms = cube_dict["atoms"]
    spacing = cube_dict["spacing"]
    origin = cube_dict["origin"]

    # Write cube file using ASE
    with open(cube_file_path, 'w') as f:
        write_cube(f, atoms, data=rho_reconstructed_grid.T)

    print(f"Writing overlap matrix to: {overlap_matrix_path}")
    np.save(overlap_matrix_path, overlap_matrix)

    print(f"Writing projection vector to: {projection_vector_path}")
    np.save(projection_vector_path, projection_vector)

    print(f"Writing coefficients to: {coefficients_path}")
    np.save(coefficients_path, projection_coeffs)

    print("Done!")

    # Print statistics
    print("\n=== Reconstruction Statistics ===")
    print(f"Original density integral: {np.sum(rho_original.fft_data) * np.linalg.det(rho_original.cell_grid) / rho_original.fft_data.size:.6e}")
    print(f"Reconstructed density integral: {np.sum(rho_reconstructed_grid) * np.linalg.det(rho_original.cell_grid) / rho_reconstructed_grid.size:.6e}")
    print(f"Max absolute difference: {np.max(np.abs(rho_original.fft_data - rho_reconstructed_grid)):.6e}")
    print(f"RMS difference: {np.sqrt(np.mean((rho_original.fft_data - rho_reconstructed_grid) ** 2)):.6e}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Project electronic density to a basis set and reconstruct.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "input_cube",
        type=str,
        help="Path to input .cube file"
    )
    parser.add_argument(
        "basis_spec",
        type=str,
        help="Path to basis set specification (.json or .yaml file)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to directory where to store output"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists"
    )

    args = parser.parse_args()

    cube2cube(
        input_cube=args.input_cube,
        basis_spec=args.basis_spec,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()



