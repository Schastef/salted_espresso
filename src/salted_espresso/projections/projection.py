import numpy as np

from tqdm import tqdm

from salted_espresso.electronic_density.types import DensityFunction
from salted_espresso.ri_basis.core import RIBasisSet


def coulomb_kernel(g2: np.ndarray) -> np.ndarray:
    """Return 4*pi/G² with the G=0 mode removed."""
    g2 = np.asarray(g2, dtype=float)

    kernel = np.zeros_like(g2, dtype=float)
    mask = g2 > 0.0

    kernel[mask] = 4.0 * np.pi / g2[mask]

    return kernel


def compute_projection_vector_cartesian_grid(
    rho: DensityFunction,
    basis_set: RIBasisSet,
    n_cartesian_grid: int | tuple[int, int, int] | None = None,
    cell_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Compute P_i = <rho | chi_i> on a generated Cartesian cell grid."""

    if cell_grid is None:
        rho_cell_grid = getattr(rho, "cell_grid", None)
        if rho_cell_grid is not None:
            cell_grid = rho_cell_grid
        else:
            raise ValueError(
                "cell_grid must be provided either as an argument or "
                "as an attribute of the density."
            )

    if n_cartesian_grid is None:
        rho_grid_shape = getattr(rho, "grid_shape", None)
        if rho_grid_shape is not None:
            n_cartesian_grid = rho_grid_shape
        else:
            n_cartesian_grid = (32, 32, 32)

    if isinstance(n_cartesian_grid, int):
        n_cartesian_grid = (
            n_cartesian_grid,
            n_cartesian_grid,
            n_cartesian_grid,
        )

    nx, ny, nz = n_cartesian_grid

    f1 = np.linspace(0, 1, nx, endpoint=False)
    f2 = np.linspace(0, 1, ny, endpoint=False)
    f3 = np.linspace(0, 1, nz, endpoint=False)

    dV = abs(np.linalg.det(cell_grid)) / (nx * ny * nz)

    F1, F2, F3 = np.meshgrid(f1, f2, f3, indexing="ij")
    frac_pts = np.stack([F1.ravel(), F2.ravel(), F3.ravel()], axis=-1)

    points = frac_pts @ cell_grid

    P = None
    chunk_size = 10_000

    for i in range(0, len(points), chunk_size):
        chunk = points[i:i + chunk_size]
        dP = (basis_set(chunk).T @ rho(chunk)) * dV

        if P is None:
            P = dP
        else:
            P += dP

    if P is None:
        return np.zeros(0, dtype=float)

    return np.asarray(P, dtype=float)


def compute_projection_vector_native_grid(
    rho: DensityFunction,
    basis_set: RIBasisSet,
    print_progress_bar: bool = False,
) -> np.ndarray:
    """Compute P_i = <rho | chi_i> on the density object's native FFT quadrature grid."""

    integrate_against = getattr(rho, "integrate_against", None)

    if not callable(integrate_against):
        raise ValueError(
            "rho must provide an 'integrate_against(func)' method to use "
            "FFT-grid projections."
        )

    origin = np.zeros((1, 3), dtype=float)

    if hasattr(rho, "origin") and getattr(rho, "origin") is not None:
        origin = np.asarray(getattr(rho, "origin"), dtype=float).reshape(1, 3)

    basis_probe = np.asarray(basis_set(origin))

    if basis_probe.ndim == 1:
        n_basis = basis_probe.shape[0]
    elif basis_probe.ndim == 2 and basis_probe.shape[0] == 1:
        n_basis = basis_probe.shape[1]
    else:
        raise ValueError(
            "basis_set(points) must return shape (n_points, n_basis) or "
            "(n_basis,) for a single point."
        )

    projections = np.empty(n_basis, dtype=float)

    iterator = range(n_basis)

    if print_progress_bar:
        iterator = tqdm(
            iterator,
            total=n_basis,
            desc="Evaluating <rho | chi_i> integrals",
            unit="integral",
        )

    for basis_index in iterator:

        def basis_component(
            points: np.ndarray,
            idx: int = basis_index,
        ) -> np.ndarray:
            values = np.asarray(basis_set(points))

            if values.ndim == 1:
                return np.array([values[idx]], dtype=float)

            if values.ndim != 2:
                raise ValueError(
                    "basis_set(points) returned an array with unsupported rank."
                )

            return np.asarray(values[:, idx], dtype=float)

        projections[basis_index] = float(integrate_against(basis_component))

    return projections


def compute_projection_vector_coulomb_metric(
    rho: DensityFunction,
    basis_set: RIBasisSet,
    print_progress_bar: bool = False,
) -> np.ndarray:
    """Compute W_i = <chi_i | v_c | rho> in reciprocal space."""

    g_vectors = rho.g_vectors
    g2 = rho.g2
    rho_g = rho.evaluate_fourier()

    kernel = coulomb_kernel(g2)

    origin = np.zeros((1, 3), dtype=float)
    basis_probe = np.asarray(basis_set(origin))

    if basis_probe.ndim == 1:
        n_basis = basis_probe.shape[0]
    elif basis_probe.ndim == 2 and basis_probe.shape[0] == 1:
        n_basis = basis_probe.shape[1]
    else:
        raise ValueError(
            "basis_set(points) must return shape (n_points, n_basis) or "
            "(n_basis,) for a single point."
        )

    projection = np.zeros(n_basis, dtype=np.complex128)

    chunk_size = 10_000
    iterator = range(0, len(g_vectors), chunk_size)

    if print_progress_bar:
        iterator = tqdm(
            iterator,
            total=len(iterator),
            desc="Evaluating Coulomb projection vector",
            unit="G chunk",
        )

    for start in iterator:
        stop = start + chunk_size

        g_chunk = g_vectors[start:stop]
        kernel_chunk = kernel[start:stop]
        rho_g_chunk = rho_g[start:stop]

        chi_g = np.asarray(
            basis_set.evaluate_fourier(g_chunk),
            dtype=np.complex128,
        )

        projection += chi_g.conj().T @ (kernel_chunk * rho_g_chunk)

    return np.asarray(np.real_if_close(projection, tol=1000))