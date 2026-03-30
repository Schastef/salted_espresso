import numpy as np
from typing import Optional

class DensityCache:
    """A simple cache for storing evaluated density on a grid."""
    def __init__(self):
        self._grid_hash: Optional[int] = None
        self._density_on_grid: Optional[np.ndarray] = None

    def get(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Retrieve density from cache if the grid is the same.
        """
        current_hash = hash(grid.tobytes())
        if self._grid_hash == current_hash:
            return self._density_on_grid
        return None

    def set(self, grid: np.ndarray, density_on_grid: np.ndarray):
        """
        Store a new density evaluation in the cache.
        """
        self._grid_hash = hash(grid.tobytes())
        self._density_on_grid = density_on_grid

density_cache = DensityCache()

