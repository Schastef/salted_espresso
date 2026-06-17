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


def build_reciprocal_grid(
    cell_vectors: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build reciprocal-space G vectors compatible with an FFT grid."""
    cell_vectors = np.asarray(cell_vectors, dtype=float)

    reciprocal_vectors = 2.0 * np.pi * np.linalg.inv(cell_vectors).T

    nx, ny, nz = grid_shape

    gx = np.fft.fftfreq(nx) * nx
    gy = np.fft.fftfreq(ny) * ny
    gz = np.fft.fftfreq(nz) * nz

    G1, G2, G3 = np.meshgrid(gx, gy, gz, indexing="ij")
    miller = np.stack((G1.ravel(), G2.ravel(), G3.ravel()), axis=-1)

    g_vectors = miller @ reciprocal_vectors
    g2 = np.einsum("ij,ij->i", g_vectors, g_vectors)

    return g_vectors, g2


def coulomb_kernel(g2: np.ndarray) -> np.ndarray:
    """Return 4*pi/G^2 with the G=0 mode removed."""
    g2 = np.asarray(g2, dtype=float)

    kernel = np.zeros_like(g2, dtype=float)
    mask = g2 > 0.0

    kernel[mask] = 4.0 * np.pi / g2[mask]

    return kernel


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


def compute_overlap_matrix_coulomb_metric(
    basis_set: RIBasisSet,
    print_progress_bar: bool = False,
) -> np.ndarray:
    """Compute S_ij = <chi_i | v_c | chi_j> in reciprocal space."""
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

    g_vectors, g2 = build_reciprocal_grid(
        cell_vectors=cell_vectors,
        grid_shape=(64, 64, 64),
    )

    kernel = coulomb_kernel(g2)

    overlap = np.zeros((n_basis, n_basis), dtype=np.complex128)
    chunk_size = 10_000

    iterator = range(0, len(g_vectors), chunk_size)

    if print_progress_bar:
        iterator = tqdm(
            iterator,
            total=len(iterator),
            desc="Evaluating Coulomb Overlap Matrix S",
            unit="G chunk",
        )

    for start in iterator:
        stop = start + chunk_size

        g_chunk = g_vectors[start:stop]
        kernel_chunk = kernel[start:stop]

        chi_g = basis_set.evaluate_fourier(g_chunk)
        chi_g = np.asarray(chi_g, dtype=np.complex128)

        if chi_g.ndim == 1:
            chi_g = chi_g[:, None]

        overlap += chi_g.conj().T @ (kernel_chunk[:, None] * chi_g)

    overlap = 0.5 * (overlap + overlap.T.conj())

    return np.asarray(np.real_if_close(overlap, tol=1000))

