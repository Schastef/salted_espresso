# salted-espresso

Tools to 
- construct **callable real-space electron densities** from quantum-chemistry / electronic-structure outputs
- construct atom centered basis sets composed of a radial and spherical component
- project electron densities onto atomic basis sets

## Installation (developer mode)

Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd salted-espresso
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
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
from salted_espresso.electronic_density import load_rho
rho = load_rho(path)
```

### Quick example

```python
from salted_espresso.electronic_density import load_rho
import numpy as np

rho = load_rho("tests/data/nvp_rho.cube")

# single point
val = rho(np.array([0.0, 0.0, 0.0]))

# many points (eager, returns ndarray)
points = np.random.rand(10, 3)
vals = rho(points)

# many points (streamed, keeps temporary memory bounded)
streamed_vals = rho((point for point in points))
for value in streamed_vals:
    ...
```

`PlaneWaveDensity` now evaluates points in a streamed way internally, trading
runtime for lower peak temporary memory during `rho(r)` calls.

If tiny imaginary residuals appear from floating-point noise, `rho(r)` returns
the real part by default and emits a one-time warning. For strict behavior, set
`complex_result_policy="raise"` on the `PlaneWaveDensity` object.

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

## RI Basis Sets
The central entry points are

```python
from salted_espresso.ri_basis import load_basis
from salted_espresso.ri_basis import load_basis_set
```

`load_basis` returns `RIBasis` object, which represents a set of functions $\{\chi_{nl}^m(\mathbf{r}-\mathbf{R})= R_{nl}(\mathbf{r}-\mathbf{R})Y_l^m(\mathbf{r}-\mathbf{R})\}_S$, centered around and atom with chemical species $S$ at position $R$. Each function is a product of a radial part $R_{nl}$ and an angular part $Y_l^m$ (spherical harmonics).

The latter returns a `RIBasisSet` object, which is a collection of `RIBasis` objects, one for each atom. `load_basis_set` requires a structure file that can be parsed by ASE (such as .cube, .xyz, .cif etc) and a `specifications` dictionary that, for each element, specifies the parameters for constructing a basis set. 

An individual `RIBasis` object is callable, taking a set of $N$ cartesian coordinates $\mathbf{r}$ as input, mapping it to a Numpy array of the shape `(N, n_basis)` containing the values of each basis function at each input point.
The returned array is in lexographic order with (n, l, m), where n runs fastest and m slowest. For example, calling an ```RIBasis```
object with ```n_max=2``` and ```l_max=1``` will return an array with following resutls:

$$[[\chi_{10}^0(\mathbf{r_1}),\:\chi_{11}^{-1}(\mathbf{r_1}),\:\chi_{11}^0(\mathbf{r_1}),\:\chi_{11}^1(\mathbf{r_1}),\:\chi_{20}^0(\mathbf{r_1}),\:\chi_{21}^{-1}(\mathbf{r_1}),\:...],\;[\chi_{10}^{0}(\mathbf{r_2}),...], ...]$$

Calling the `RIBasisSet` returns a block matrix where each block is the result of calling the individual `RIBasis`. The ordering corresponds to the order of appearance of the structure file from which `RIBasisSet` was created.

### Quick Example: Load RIBasis of a single atom

```python
import numpy as np
from salted_espresso.ri_basis import load_basis

species = 'H'
R = (0.0, 0.0, 0.0)
n_max = 2
l_max = 2
alphas=[1 for i in range(n_max * (l_max+1))]

basis = load_basis(species, R, n_max, l_max,
                   radial_method="gaussian",
                   angular_method="spherical",
                   radial_params={"alphas": alphas})

r = np.array([[0.0, 0.0, 0.1], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
print(basis(r))
```

### Quick Example: Load RIBasisSet from a structure
```python
import numpy as np
from salted_espresso.ri_basis import load_basis_set

structure_file = "./tests/data/nvp_rho.cube"
specifications_file = "./tests/data/example_primitive_gaussian.json"

basis_set = load_basis_set(structure_file, specifications_file)

```

### Adding a new radial method
To add a new way to compute the radial part of the basis function:
1. Create a module in `ri_basis/`, e.g.
```ptyhon 
new_radial.py
```
2. Implement
```python
from .core import RadialFunction

def NewRadials(RadialFunction):
    def __init__(self, n_max, l_max, **kwargs):
        ...
        self.radials = ... # Set of n_max * (l_max + 1) callables representing the radial functions R_{nl}(r)
```

3. Register it in the ```__init__.py```:
```python
from .new_radial import NewRadials
register_radial("my_method", NewRadials)
```
