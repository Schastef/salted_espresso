from __future__ import annotations

from typing import Literal

import numpy as np
import torch  #Pytorch for algorthmic differentiation
import scipy.optimize


from salted_espresso.ri_basis.core import RIBasis, RIBasisSet
from salted_espresso.electronic_density.types import DensityFunction
from salted_espresso.projections.core import compute_projectability


"""Iteratively optimize primitive Gaussian alphas to maximize projectability."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from ase.io import read
from scipy.optimize import minimize


REPO_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_DIR / "src"
if SRC_DIR.exists():
	sys.path.insert(0, str(SRC_DIR))

from salted_espresso.electronic_density import load_rho
from salted_espresso.projections.core import (compute_condition_number,
											  compute_overlap,
											  compute_projection_coefficients,
											  compute_projectability,
											  solve_projection_equations)
from salted_espresso.ri_basis import load_basis_set
from salted_espresso.ri_basis.types import CutoffType


DEFAULT_ALPHA_SEEDS: Dict[int, float] = {0: 0.15, 1: 0.25, 2: 0.40, 3: 0.65}
DEFAULT_BETA = 2.5


def _parse_cutoff(value: str) -> CutoffType | float:
	value = value.strip().lower()
	if value in {"none", "non-periodic", "nonperiodic", "np"}:
		return CutoffType.NON_PERIODIC
	if value in {"estimate", "est"}:
		return CutoffType.ESTIMATE
	return float(value)


def _parse_species_nl(value: str) -> dict[str, dict]:
	"""Parse per-species n_max and l_max from JSON string or file path."""
	import json
	try:
		# Try to parse as JSON string
		return json.loads(value)
	except json.JSONDecodeError:
		# Assume it's a file path
		with open(value, 'r') as f:
			return json.load(f)


def normalize_n_max(n_max: int | Sequence[int], l_max: int) -> List[int]:
	if isinstance(n_max, int):
		if n_max < 0:
			raise ValueError(f"n_max must be non-negative, got {n_max}.")
		return [n_max] * (l_max + 1)

	n_max_list = [int(v) for v in n_max]
	if len(n_max_list) != l_max + 1:
		raise ValueError(
			f"If n_max is a list, it must have length l_max + 1 ({l_max + 1}), "
			f"but got {len(n_max_list)}."
		)
	if any(v < 0 for v in n_max_list):
		raise ValueError(f"All n_max entries must be non-negative, got {n_max_list}.")
	return n_max_list


def _enumerate_nl_pairs(n_max_by_l: Sequence[int]) -> List[tuple[int, int]]:
	pairs: List[tuple[int, int]] = []
	max_n = max(n_max_by_l, default=0)
	for n in range(max_n):
		for l, n_l in enumerate(n_max_by_l):
			if n < n_l:
				pairs.append((n, l))
	return pairs


def generate_alphas(species_specs: dict[str, dict], *, beta: float = DEFAULT_BETA,
					alpha_seeds: Dict[int, float] | None = None) -> dict[str, List[float]]:
	seeds = alpha_seeds or DEFAULT_ALPHA_SEEDS
	fallback_l = max(seeds)
	alphas: dict[str, List[float]] = {}
	for species, spec in species_specs.items():
		n_max = spec["n_max"]
		l_max = spec["l_max"]
		n_max_by_l = normalize_n_max(n_max, l_max)
		alphas[species] = []
		for n, l in _enumerate_nl_pairs(n_max_by_l):
			base = seeds.get(l, seeds[fallback_l])
			alphas[species].append(float(base * (beta ** n)))
	return alphas


def _alphas_to_map(alphas_dict: dict[str, Sequence[float]], species_specs: dict[str, dict]) -> Dict[str, float]:
	result = {}
	for species, alphas in alphas_dict.items():
		spec = species_specs[species]
		n_max = spec["n_max"]
		l_max = spec["l_max"]
		n_max_by_l = normalize_n_max(n_max, l_max)
		pairs = _enumerate_nl_pairs(n_max_by_l)
		
		if len(alphas) != len(pairs):
			raise ValueError(f"Expected {len(pairs)} alphas for {species}, got {len(alphas)}.")
		for (n, l), alpha in zip(pairs, alphas):
			result[f"{species}_n{n}_l{l}"] = float(alpha)
	return result


def _unique_species(structure_path: Path) -> List[str]:
	atoms = read(str(structure_path))
	return sorted({str(atom.symbol) for atom in atoms})


def _build_specification(species_specs: dict[str, dict],
						 alphas_dict: dict[str, Sequence[float]]) -> Dict[str, dict]:
	spec = {}
	for species, params in species_specs.items():
		n_max = params["n_max"]
		l_max = params["l_max"]
		n_max_by_l = normalize_n_max(n_max, l_max)
		spec[species] = {
			"n_max": n_max_by_l,
			"l_max": l_max,
			"radial_method": "gaussian",
			"angular_method": "real_spherical",
			"radial_params": {"alphas": list(map(float, alphas_dict[species]))},
		}
	return spec


@dataclass
class GridParams:
	n_grid: int
	initial_r_max: float
	cutoff: float


@dataclass
class IterationResult:
	iteration: int
	alphas_dict: dict[str, List[float]]
	projectability: float
	loss: float
	expansion_coefficients: List[float]
	condition_number: float


class JSONLogger:
	def __init__(self, path: Path, *, append: bool = False):
		self.path = path
		self.path.parent.mkdir(parents=True, exist_ok=True)
		if not append:
			self.path.write_text("")

	def log(self, payload: dict) -> None:
		def _default(obj):
			if isinstance(obj, np.ndarray):
				return obj.tolist()
			if isinstance(obj, Path):
				return str(obj)
			raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

		with self.path.open("a", encoding="utf-8") as f:
			f.write(json.dumps(payload, default=_default) + "\n")


def _evaluate(rho_path: Path, structure_path: Path, alphas_dict: dict[str, Sequence[float]],
			  species_specs: dict[str, dict], grid: GridParams, cutoff: CutoffType | float,
			  verbose: bool) -> tuple[float, np.ndarray, np.ndarray]:
	if verbose:
		print(f"Building basis set with alphas: {alphas_dict}")

	rho = load_rho(rho_path)
	species = _unique_species(structure_path)
	specs = _build_specification(species_specs, alphas_dict)
	basis_set = load_basis_set(str(structure_path), specs, cutoff=cutoff)

	if verbose:
		print("Compute overlap matrix...")
	overlap = compute_overlap(basis_set, "overlap" ,
							  n_cartesian_grid = grid.n_grid,
							  initial_r_max = grid.initial_r_max,
							  cutoff = grid.cutoff,
	                          )

	if verbose:
		print("Computing projectability...")
	projectability, expansion_coeffs = compute_projectability(
		rho,
		basis_set,
		n_cartesian_grid=grid.n_grid,
		initial_r_max=grid.initial_r_max,
		cutoff=grid.cutoff,
	)

	return (
		float(projectability),
		np.asarray(expansion_coeffs, dtype=float),
		np.asarray(overlap, dtype=float)
	)


def optimize_alphas(
	rho_path: Path,
	structure_path: Path,
	species_specs: dict[str, dict],
	*,
	beta: float,
	alpha_seeds: Dict[int, float],
	cutoff: CutoffType | float,
	grid: GridParams,
	tol: float,
	alpha_xtol: float,
	max_iter: int,
	log_path: Path,
	verbose: bool,
	append_log: bool,
) -> IterationResult:
	initial_alphas_dict = generate_alphas(species_specs, beta=beta, alpha_seeds=alpha_seeds)
	# Flatten alphas for optimizer
	species_order = sorted(species_specs.keys())
	initial_alphas_flat = [a for sp in species_order for a in initial_alphas_dict[sp]]
	logger = JSONLogger(log_path, append=append_log)

	iteration_counter = {"count": 0}

	def _objective(log_alphas: np.ndarray) -> float:
		alphas_flat = np.exp(log_alphas)
		# Unflatten to dict
		alphas_dict = {}
		offset = 0
		for sp in species_order:
			n_alphas = len(initial_alphas_dict[sp])
			alphas_dict[sp] = alphas_flat[offset:offset + n_alphas].tolist()
			offset += n_alphas

		projectability, expansion_coeffs, overlap = _evaluate(
			rho_path,
			structure_path,
			alphas_dict,
			species_specs,
			grid,
			cutoff,
			verbose,
		)
		loss = float(max(0.0, 1.0 - projectability))
		iteration_counter["count"] += 1
		iteration = iteration_counter["count"]
		condition_number = compute_condition_number(overlap)

		result = IterationResult(
			iteration=iteration,
			alphas_dict=alphas_dict,
			projectability=projectability,
			loss=loss,
			expansion_coefficients=list(map(float, expansion_coeffs)),
			condition_number=float(condition_number),
		)

		if verbose:
			print(f"Iter {iteration:03d}: P={projectability:.6f}, loss={loss:.6e}, cond={condition_number:.3e}")

		current_best = best_result_holder[0]
		if current_best is None or projectability > current_best.projectability:
			best_result_holder[0] = result

		logger.log({
			"iteration": iteration,
			"projectability": projectability,
			"loss": loss,
			"alphas": _alphas_to_map(alphas_dict, species_specs),
			"alphas_linear": alphas_flat.tolist(),
			"expansion_coefficients": result.expansion_coefficients,
			"condition_number": condition_number,
		})

		return loss

	best_result_holder: List[IterationResult | None] = [None]

	minimize(
		_objective,
		x0=np.log(initial_alphas_flat),
		method="Nelder-Mead",
		options={
			"maxiter": max_iter,
			"fatol": tol,
			"xatol": alpha_xtol,
			"disp": verbose,
		},
	)

	final_result = best_result_holder[0]
	if final_result is None:
		# Fallback: evaluate once in case optimizer short-circuited
		p, c_exp, overlap = _evaluate(
			rho_path,
			structure_path,
			initial_alphas_dict,
			species_specs,
			grid,
			cutoff,
			verbose,
		)
		final_result = IterationResult(
			iteration=0,
			alphas_dict=initial_alphas_dict,
			projectability=p,
			loss=max(0.0, 1.0 - p),
			expansion_coefficients=list(map(float, c_exp)),
			condition_number=float(compute_condition_number(overlap)),
		)

	return final_result


def _parse_args() -> argparse.Namespace:
	default_cube = REPO_DIR.parent.parent / "tests" / "data" / "nvp_rho.cube"
	default_log = REPO_DIR / "convergence" / "alphas_history.jsonl"
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--species-nl", type=str, help="JSON string or file path for per-species n_max and l_max, e.g. {\"H\": {\"n_max\": 2, \"l_max\": 1}, \"O\": {\"n_max\": 3, \"l_max\": 2}}")
	parser.add_argument("--n-max", type=str, help="Either an int or comma-separated list, e.g. 2 or 3,2,1 (used for all species if --species-nl not provided)")
	parser.add_argument("--l-max", type=int, help="Maximum angular momentum (inclusive, used for all species if --species-nl not provided)")
	parser.add_argument("--cube", type=Path, default=default_cube, help="Path to target electron density cube file")
	parser.add_argument("--structure", type=Path, default=None, help="Structure file for basis construction (defaults to cube path)")
	parser.add_argument("--log", type=Path, default=default_log, help="Path to JSONL log file")
	parser.add_argument("--max-iter", type=int, default=30, help="Maximum optimizer iterations")
	parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance on 1-P")
	parser.add_argument("--alpha-xtol", type=float, default=1e-3, help="Nelder-Mead xatol on log-alphas")
	parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Geometric progression factor for initial alphas")
	parser.add_argument("--alpha-s", type=float, default=DEFAULT_ALPHA_SEEDS[0], help="Initial alpha seed for s (l=0)")
	parser.add_argument("--alpha-p", type=float, default=DEFAULT_ALPHA_SEEDS[1], help="Initial alpha seed for p (l=1)")
	parser.add_argument("--alpha-d", type=float, default=DEFAULT_ALPHA_SEEDS[2], help="Initial alpha seed for d (l=2)")
	parser.add_argument("--alpha-f", type=float, default=DEFAULT_ALPHA_SEEDS[3], help="Initial alpha seed for f (l=3)")
	parser.add_argument("--cutoff", type=str, default="non-periodic", help="Cutoff strategy: non-periodic|estimate|<float>")
	parser.add_argument("--n-cartesian-grid", type=int, default=10, help="Cartesian grid points for quadrature")
	parser.add_argument("--initial-r-max", type=float, default=8.0, help="Initial guess for maximum radius")
	parser.add_argument("--cart-cutoff", type=float, default=1e-10, help="Tail cutoff for cartesian grid")
	parser.add_argument("--append-log", action="store_true", help="Append to existing log instead of overwriting")
	parser.add_argument("--verbose", action="store_true",default=True, help="Enable verbose logging")
	return parser.parse_args()


def _parse_n_max_arg(value: str, l_max: int) -> int | List[int]:
	text = value.strip()
	if "," not in text:
		return int(text)
	parts = [p.strip() for p in text.split(",") if p.strip()]
	parsed = [int(p) for p in parts]
	# Validate early so CLI errors are explicit.
	normalize_n_max(parsed, l_max)
	return parsed


def main() -> None:
	args = _parse_args()
	rho_path = args.cube
	structure_path = args.structure or args.cube

	if args.species_nl:
		species_specs = _parse_species_nl(args.species_nl)
	elif args.n_max and args.l_max is not None:
		n_max = _parse_n_max_arg(args.n_max, args.l_max)
		species = _unique_species(structure_path)
		species_specs = {sp: {"n_max": n_max, "l_max": args.l_max} for sp in species}
	else:
		raise ValueError("Either --species-nl or both --n-max and --l-max must be provided")

	alpha_seeds = {
		0: args.alpha_s,
		1: args.alpha_p,
		2: args.alpha_d,
		3: args.alpha_f,
	}
	print("Starting optimization with the following parameters:")
	print(f"Beta: {args.beta}")
	print(f"Alpha seeds: {alpha_seeds}")
	print(f"Cutoff: {args.cutoff}")

	grid = GridParams(
		n_grid=args.n_cartesian_grid,
		initial_r_max=args.initial_r_max,
		cutoff=args.cutoff,
	)

	cutoff = _parse_cutoff(str(args.cutoff))

	result = optimize_alphas(
		rho_path,
		structure_path,
		species_specs=species_specs,
		beta=args.beta,
		alpha_seeds=alpha_seeds,
		cutoff=cutoff,
		grid=grid,
		tol=args.tol,
		alpha_xtol=args.alpha_xtol,
		max_iter=args.max_iter,
		log_path=args.log,
		verbose=args.verbose,
		append_log=args.append_log,
	)

	print("Optimization finished.")
	print(f"Best projectability: {result.projectability:.6f}")
	print(f"Alphas: {_alphas_to_map(result.alphas_dict, species_specs)}")
	print(f"Log written to: {args.log}")


if __name__ == "__main__":
	main()



# def optimise_ri_basis(rho: DensityFunction,
#     basis: RIBasis | RIBasisSet,
#     *,
#     n_radial_grid: int = 512,
#     initial_r_max: float = 8.0,
#     radial_cutoff: float = 1e-10,
#     rcond: float = 1e-12,
#     clip_tolerance: float = 1e-8,
# ) -> tuple[float, np.ndarray]:
    
#     """Optimise the coefficients of a RI basis to best represent a given density function. Only the radial part of the basis functions is optimised.
#     The optimization is performed by maximising the projectability of the density function onto the RI basis, which is defined as the squared overlap between the density and its projection onto the basis, normalized by the norm of the projection. 
#     The optimization is performed using a weighted least-squares approach, where the weights are determined by the projectability of each basis function. 
#     The optimization is performed using a gradient-based optimization algorithm, 
#     which requires the computation of gradients of the projectability with respect to the coefficients of the RI basis. This implementation uses PyTorch for automatic differentiation to compute these gradients efficiently. (PyTorch still to be implemented)
    
#     Parameters:
#     -----------
#     rho: DensityFunction
#         The density function to be represented.
#     basis: RIBasis | RIBasisSet
#         The RI basis to be optimized.
#     n_radial_grid: int, optional
#         The number of points in the radial grid used for integration. Default is 512.
#     initial_r_max: float, optional
#         The initial maximum radius for the radial grid. The actual maximum radius will be determined based on the extent of the basis functions. Default is 8.0.
#     radial_cutoff: float, optional
#         The cutoff value for the radial functions. The radial grid will be extended until the tail of the radial functions is below this cutoff. Default is 1e-10.
#     rcond: float, optional
#         The cutoff for small singular values when computing the pseudo-inverse of the overlap matrix. Default is 1e-12.
#     clip_tolerance: float, optional
#         The tolerance for clipping the projectability values to avoid numerical instabilities. Default is 1e-8. """
    
#     proj, coeffs = compute_projectability_cart(rho, basis, n_radial_grid=n_radial_grid, initial_r_max=initial_r_max, radial_cutoff=radial_cutoff, rcond=rcond, clip_tolerance=clip_tolerance)

#     print(f"Initial projectability: {proj:.6f}")


    
    
#     pass