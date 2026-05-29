from salted_espresso.electronic_density.types import DensityFunction
from salted_espresso.ri_basis.core import RIBasisSet

def compute_overlap_matrix(basis_set: RIBasisSet) -> np.ndarray:
    """Computeds the overlap matrix S_ij = <chi_i | chi_j> for the given density and basis set."""
    pass


def compute_projection_vector(rho: DensityFunction, basis_set: RIBasisSet) -> np.ndarray:
    """Computes the individual projection coefficients P_i = <rho | chi_i> for the given density and basis set."""
    pass


def solve_projections_coeffs(overlap_matrix: np.ndarray, projection_vector: np.ndarray) -> np.ndarray:
    """Solves the linear system S c = P to obtain the projection coefficients c_i."""
    pass
