# salted-espresso

Tools to construct **callable real-space electron densities** from quantum-chemistry / electronic-structure outputs.

## Installation (developer mode)

Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd salted-espresso
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This ensures that:

- package imports work correctly (no `src` in import paths)
- local code changes are immediately visible

---


## Project structure

```
src/
    electronic_density/   # Parse output of QM codes and create a callable \rho(r) from it
    ri_basis/             # Construct / Select auxiliary basis sets for resolution of identity (RI) 
    density_fitting/      # Project \rho(r) onto RI basis, create output for SALTED training
tests/
    data/                 # small reference datasets
```

---

## Running tests

From the repository root:

```bash
pytest
```

---
## Density Functions

The central entry point is:

```python
from electronic_density import load_rho
rho = load_rho(path)
```

### Quick example

```python
from electronic_density import load_rho
import numpy as np

rho = load_rho("tests/data/nvp_rho.cube")

# single point
val = rho(np.array([0.0, 0.0, 0.0]))

# many points
points = np.random.rand(10, 3)
vals = rho(points)
```

### Adding a new density backend (`x2rho`)

To support a new file format:

1. Create a module in `electronic_density/`, e.g.

```
qe2rho.py
```

2. Implement:

```python
def load_rho_from_<format>(path: Path) -> DensityFunction:
    ...
```

3. Register it in the central dispatcher (`load_rho`).

All loaders must return an object implementing the `DensityFunction`
protocol (a callable `rho(r)`).

---
The returned object implements the `DensityFunction` interface:

- input shape `(3,)` → scalar density
- input shape `(n, 3)` → array of densities

---
which returns a callable object representing the analytical density  

$$
\rho(\mathbf r)
$$

---

