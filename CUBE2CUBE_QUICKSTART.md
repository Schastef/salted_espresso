# cube2cube Quick Start Guide

## TL;DR

```bash
cd salted_espresso/
python3 cube2cube.py input.cube basis_spec.json output.cube
```

## What is cube2cube?

The cube2cube script takes an electron density from a .cube file, projects it onto a resolution of identity (RI) basis set, and writes the reconstructed density to a new .cube file. It also exports the overlap matrix, the projection vector and the projection coefficients as .npy files in the same directory as the .cube file.

## Step 1: Prepare Your Basis Set Specification

Create a JSON file defining your basis functions for each atom type:

**simple_basis.json** (s-orbitals only, recommended for beginners):
```json
{
    "C": {
        "n_max": 2,
        "l_max": 0,
        "radial_method": "gaussian",
        "angular_method": "real_spherical",
        "radial_params": {"alphas": [0.5, 1.0]}
    },
    "O": {
        "n_max": 2,
        "l_max": 0,
        "radial_method": "gaussian",
        "angular_method": "real_spherical",
        "radial_params": {"alphas": [0.5, 1.0]}
    }
}
```

**extended_basis.json** (s and p orbitals):
```json
{
    "C": {
        "n_max": 2,
        "l_max": 1,
        "radial_method": "gaussian",
        "angular_method": "real_spherical",
        "radial_params": {"alphas": [0.3, 0.6, 0.9, 1.2]}
    },
    "O": {
        "n_max": 2,
        "l_max": 1,
        "radial_method": "gaussian",
        "angular_method": "real_spherical",
        "radial_params": {"alphas": [0.3, 0.6, 0.9, 1.2]}
    }
}
```

## Step 2: Run the Script

```bash
python3 cube2cube.py co2.cube basis.json output_dir
```

**Optional flags:**
- `--overwrite`: Overwrite existing output file

## Step 3: Review the Output

The script prints detailed statistics:
```
=== Reconstruction Statistics ===
Original density integral: 4.297307e+00
Reconstructed density integral: 3.324549e+00
Max absolute difference: 7.440164e+00
RMS difference: 3.918894e-01
```

If reconstructed integral is close to original, projection was good
RMS difference shows average error per grid point

## Common Use Cases

### Project density with minimal basis (fastest)
```bash
echo '{
    "H": {"n_max": 1, "l_max": 0, "radial_method": "gaussian", 
          "angular_method": "real_spherical", "radial_params": {"alphas": [1.0]}},
    "C": {"n_max": 1, "l_max": 0, "radial_method": "gaussian",
          "angular_method": "real_spherical", "radial_params": {"alphas": [1.0]}}
}' > minimal.json

python3 cube2cube.py structure.cube minimal.json output_dir
```

### Project with custom structure file
```bash
python3 cube2cube.py density.cube basis.json output.cube --structure-file custom_structure.xyz
```

### Use in batch processing
```bash
for file in *.cube; do
    python3 cube2cube.py "$file" basis.json "${file%.cube}_proj.cube" --overwrite
done
```

## Troubleshooting

### Script says "Expected X alphas, but got Y"
**Solution:** The number of alphas must equal `n_max × (l_max + 1)`
- For `n_max=2, l_max=0`: Need 2 alphas
- For `n_max=2, l_max=1`: Need 4 alphas
- For `n_max=2, l_max=2`: Need 6 alphas
