import numpy as np

from salted_espresso.ri_basis.core import RIBasisSet

import numpy as np
from tqdm import tqdm

from salted_espresso.ri_basis.core import RIBasisSet


def build_integration_grid(
    cell_vectors: np.ndarray,
    n_cartesian_grid: int = 64,
) -> tuple[np.ndarray, float]:

    nx = ny = nz = n_cartesian_grid

    f1 = np.linspace(0, 1, nx, endpoint=False)
    f2 = np.linspace(0, 1, ny, endpoint=False)
    f3 = np.linspace(0, 1, nz, endpoint=False)

    F1, F2, F3 = np.meshgrid(f1, f2, f3, indexing="ij")
    frac_pts = np.stack((F1.ravel(), F2.ravel(), F3.ravel()), axis=-1)

    points = frac_pts @ cell_vectors

    dV = abs(np.linalg.det(cell_vectors)) / (nx * ny * nz)

    return points, dV


def compute_overlap_matrix_overlap_metric(
    basis_set: RIBasisSet,
    print_progress_bar: bool = False,
) -> np.ndarray:

    cell_vectors = np.asarray(getattr(basis_set, "cell_vectors", None), dtype=float)
    if cell_vectors.size == 0:
        raise ValueError(
            "basis_set must provide cell_vectors to compute the overlap matrix."
        )

    basis_probe = np.asarray(basis_set(np.zeros((1, 3), dtype=float)))

    if basis_probe.ndim == 1:
        n_basis = basis_probe.shape[0]
    elif basis_probe.ndim == 2 and basis_probe.shape[0] == 1:
        n_basis = basis_probe.shape[1]
    else:
        raise ValueError(
            "basis_set(points) must return shape (1, n_basis) or (n_basis,) "
            "for a single point."
        )

    points, dV = build_integration_grid(cell_vectors)

    overlap = np.zeros((n_basis, n_basis), dtype=np.complex128)
    chunk_size = 10000

    iterator = range(0, len(points), chunk_size)

    if print_progress_bar:
        iterator = tqdm(
            iterator,
            total=len(iterator),
            desc="Evaluating Overlap Matrix S",
            unit="matrix element chunk",
        )

    for i in iterator:
        chunk = points[i:i + chunk_size]
        values = np.asarray(basis_set(chunk))
        overlap += values.conj().T @ values * dV

    return np.asarray(
        np.real_if_close(
            0.5 * (overlap + overlap.T.conj()),
            tol=1000,
        )
    )