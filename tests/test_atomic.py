from __future__ import annotations

from typing import Literal

import numpy as np
import torch  #Pytorch for algorthmic differentiation
import scipy.optimize


from salted_espresso.ri_basis import load_basis
from salted_espresso.ri_basis.core import AngularFunctions, RadialFunctions, RIBasis
from salted_espresso.ri_basis.multiply_wfcs import Atomic_Rad
from salted_espresso.ri_basis.realspherharmonic import RealSphericalHarmonics



def _make_basis(n_max: int | list[int], l_max: int, origin=(0.0, 0.0, 0.0)) -> RIBasis:
    """Convenience factory that creates a Gaussian/spherical RI basis."""
    if isinstance(n_max, int):
        n_alphas = n_max * (l_max + 1)
    else:
        n_alphas = sum(n_max)
    return load_basis(
        species="Na",
        origin=origin,
        n_max=n_max,
        l_max=l_max,
        radial_method="atomic",
        angular_method="spherical",
        radial_params={"filename": "tests/data/Na_TZP_rc10.0_qe.dat"}
    )



class TestAtomicRadials:
    
    def test_radial_evaluation(self):
        """Test that the Atomic_Rad class correctly evaluates the radial functions."""
        origin = (0.0, 0.0, 0.0)
        n_max = 2
        l_max = 2
        basis = _make_basis(n_max, l_max, origin)

        # Test evaluation at a single point
        r = np.array([[0.5, 0.0, 0.0]])  # (N=1, 3)
        expected_shape = (len(basis.radial_funcs),)  # (n_radials,)
        radial_values = basis.radial_funcs(r)  # (n_radials,)
        assert radial_values.shape == expected_shape, f"Expected shape {expected_shape}, got {radial_values.shape}"

        # Test evaluation at multiple points
        r_multi = np.array([[0.5, 0.0, 0.0], [1.0, 1.0, 1.0]])  # (N=2, 3)
        expected_shape_multi = (len(basis.radial_funcs), r_multi.shape[0])  # (n_radials, N)
        radial_values_multi = basis.radial_funcs(r_multi)  # (n_radials, N)
        assert radial_values_multi.shape == expected_shape_multi, f"Expected shape {expected_shape_multi}, got {radial_values_multi.shape}"