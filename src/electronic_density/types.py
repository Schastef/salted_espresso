from collections.abc import Iterable, Iterator
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
     - an iterable / generator yielding points of shape (3,)

     Return conventions
     ------------------
     - for input shape (3,), return a scalar density value
     - for input shape (n, 3), return an array of shape (n,)
     - for streamed iterable input, return an iterator of scalar values

     Notes
     -----
     - The returned density must be real-valued up to numerical noise.
     - Coordinates are expected in the same length units as used by the
       originating density source.
     """

    def __call__(
        self,
        r: npt.NDArray[np.floating] | Iterable[object],
    ) -> npt.NDArray[np.floating] | np.floating | Iterator[np.floating]:
        """
        Evaluate the electronic density at one or more Cartesian points.

        Args:
            r:
                A single point (3,), an array of points (n, 3), or an
                iterable yielding points of shape (3,).

        Returns:
            Scalar density, array of densities, or a streamed iterator of
            scalar densities.
        """
        ...
