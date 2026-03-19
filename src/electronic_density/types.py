from typing import Protocol
import numpy as np
import numpy.typing as npt

class DensityFunction(Protocol):
    """
     Callable representation of an electronic density.

     Implementations must represent a real-space electron density rho(r)
     and support evaluation at Cartesian positions.

     Input conventions
     -----------------
     r may be:
     - a single Cartesian point of shape (3,)
     - an array of Cartesian points of shape (n, 3)

     Return conventions
     ------------------
     - for input shape (3,), return a scalar density value
     - for input shape (n, 3), return an array of shape (n,)

     Notes
     -----
     - The returned density must be real-valued up to numerical noise.
     - Coordinates are expected in the same length units as used by the
       originating density source.
     """
    def __call__(self, r: npt.NDArray[np.floating]) -> npt.ndarray[np.floating] | np.floating:
        """
        Evaluate the electronic density at one or more Cartesian points.

        Args:
            r:
                Either a single point of shape (3,) or an array of points
                of shape (n, 3).

        Returns:
            The density value at the given point, or an array of density
            values for multiple points.
        """
        ...




