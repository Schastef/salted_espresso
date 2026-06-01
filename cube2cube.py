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
    python cube2cube.py <input.cube> <basis_spec.json> <output.cube> [--structure-file <structure.xyz>]

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


def infer_structure_from_cube(cube_path: Path) -> Path:
    """
    Create a temporary structure file from cube file atomic information.

    Args:
        cube_path: Path to the .cube file

    Returns:
        Path to a temporary .xyz file
    """
    cube_dict = load_cubefile(cube_path)
    atoms = cube_dict["atoms"]

    # Create temporary .xyz file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False)
    temp_path = temp_file.name

    # Use ASE to write the structure
    ase_read(str(cube_path), format='cube').write(temp_path, format='xyz')
    temp_file.close()

    return Path(temp_path)


def reconstruct_density_on_grid(
    rho_original,
    basis_set,
    grid_shape: tuple,
    cell_grid: np.ndarray,
    origin: np.ndarray
) -> np.ndarray:
    """
    Reconstruct the projected density on the original grid.

    Args:
        rho_original: Original PlaneWaveDensity object
        basis_set: RIBasisSet object
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
    rho_reconstructed = rho_original(points)

    # Reshape back to grid
    return rho_reconstructed.reshape(grid_shape)


def cube2cube(
    input_cube: Path,
    basis_spec: Path,
    output_cube: Path,
    structure_file: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    """
    Project electronic density to a basis set and reconstruct.

    Args:
        input_cube: Path to input .cube file
        basis_spec: Path to basis set specification (.json or .yaml)
        output_cube: Path to output .cube file
        structure_file: Path to structure file for basis set. If None, inferred from cube.
        overwrite: Whether to overwrite existing output file
    """
    input_cube = Path(input_cube)
    basis_spec = Path(basis_spec)
    output_cube = Path(output_cube)

    if not input_cube.exists():
        raise FileNotFoundError(f"Input cube file not found: {input_cube}")
    if not basis_spec.exists():
        raise FileNotFoundError(f"Basis specification file not found: {basis_spec}")
    if output_cube.exists() and not overwrite:
        raise FileExistsError(
            f"Output cube file already exists: {output_cube}. Use --overwrite to replace."
        )

    print(f"Loading electron density from: {input_cube}")
    # Step 1: Load original density
    rho_original = load_rho_from_cube(input_cube)
    cube_dict = load_cubefile(input_cube)

    print(f"Grid shape: {rho_original.grid_shape}")
    print(f"Cell vectors:\n{rho_original.cell_grid}")

    # Step 2: Infer or use provided structure file
    if structure_file is None:
        print("Inferring structure from cube file...")
        structure_file = infer_structure_from_cube(input_cube)
        temp_structure = True
    else:
        structure_file = Path(structure_file)
        temp_structure = False

    print(f"Using structure file: {structure_file}")

    # Step 3: Load basis set
    print(f"Loading basis set from: {basis_spec}")
    basis_set = load_basis_set(
        str(structure_file),
        str(basis_spec),
        cutoff=CutoffType.NON_PERIODIC,
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
    overlap_matrix = compute_overlap_matrix(basis_set)
    print(f"Overlap matrix shape: {overlap_matrix.shape}")
    print(f"Overlap matrix condition number: {np.linalg.cond(overlap_matrix):.2e}")

    # Step 5: Calculate projection vector using FFT grid
    print("Computing projection vector using FFT integration...")
    projection_vector = compute_projection_vector_FFT(rho_original, basis_set)
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
        basis_set,
        rho_original.grid_shape,
        rho_original.cell_grid,
        rho_original.origin
    )

    # Step 9: Write to cube file
    print(f"Writing reconstructed density to: {output_cube}")

    # Prepare atoms object
    atoms = cube_dict["atoms"]
    spacing = cube_dict["spacing"]
    origin = cube_dict["origin"]

    # Write cube file using ASE
    with open(output_cube, 'w') as f:
        write_cube(f, atoms, data=rho_reconstructed_grid.T)

    print("Done!")

    # Print statistics
    print("\n=== Reconstruction Statistics ===")
    print(f"Original density integral: {np.sum(rho_original.fft_data) * np.linalg.det(rho_original.cell_grid) / rho_original.fft_data.size:.6e}")
    print(f"Reconstructed density integral: {np.sum(rho_reconstructed_grid) * np.linalg.det(rho_original.cell_grid) / rho_reconstructed_grid.size:.6e}")
    print(f"Max absolute difference: {np.max(np.abs(rho_original.fft_data - rho_reconstructed_grid)):.6e}")
    print(f"RMS difference: {np.sqrt(np.mean((rho_original.fft_data - rho_reconstructed_grid) ** 2)):.6e}")

    # Cleanup temporary file if created
    if temp_structure:
        Path(structure_file).unlink()


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
        "output_cube",
        type=str,
        help="Path to output .cube file"
    )
    parser.add_argument(
        "--structure-file",
        type=str,
        default=None,
        help="Path to structure file (e.g., .xyz). If not provided, inferred from cube file."
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
        output_cube=args.output_cube,
        structure_file=args.structure_file,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()



