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

## RI Basis
The central entry point is

```python
from ri_basis import load_basis
```

This returns a `RIBasis` object, which represents a set of functions $\{\chi_{nl}^m(\mathbf{r}-\mathbf{R})= R_{nl}(\mathbf{r}-\mathbf{R})Y_l^m(\mathbf{r}-\mathbf{R})\}_S$, 
centered around and atom with chemical species $S$ at position $R$. Each function is a product of a radial part $R_{nl}$ and an angular part $Y_l^m$ (spherical harmonics).

The `RIBasis` object is callable, taking a set of $N$ cartesian coordinates $\mathbf{r}$ as input, mapping it to a Numpy array of the shape `(N, n_basis)` containing the values of each basis function at each input point.
The returned array is in lexographic order with (n, l, m), where n runs fastest and m slowest. For example, calling an ```RIBasis```
object with ```n_max=2``` and ```l_max=1``` will return an array with following resutls:

$$[[\chi_{10}^0(\mathbf{r_1}),\:\chi_{11}^{-1}(\mathbf{r_1}),\:\chi_{11}^0(\mathbf{r_1}),\:\chi_{11}^1(\mathbf{r_1}),\:\chi_{20}^0(\mathbf{r_1}),\:\chi_{21}^{-1}(\mathbf{r_1}),\:...],\;[\chi_{10}^{0}(\mathbf{r_2}),...], ...]$$

When the ```RIBasis``` is instantiated, the user has to specify a method for computing the radial and angular part of the basis function.

### Quick Example

```python
import numpy as np
from ri_basis.loader import load_basis, register_angular, register_radial

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

