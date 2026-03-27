"""Lightweight tests for streamed rho(r) evaluation behavior."""

from collections.abc import Iterator

import numpy as np

from electronic_density.cube2rho import PlaneWaveDensity


def make_small_density() -> PlaneWaveDensity:
    rng = np.random.default_rng(11)
    n_pairs = 16
    g_half = rng.normal(size=(n_pairs, 3))
    coeff_half = rng.normal(size=n_pairs) + 1j * rng.normal(size=n_pairs)

    # Conjugate symmetry rho(-G)=conj(rho(G)) guarantees real rho(r).
    rho_g = np.concatenate([coeff_half, np.conjugate(coeff_half)])
    G = np.vstack([g_half, -g_half])
    return PlaneWaveDensity(rho_g=rho_g, G=G, max_batch_memory_mb=0.0005)


def dense_reference(rho_obj: PlaneWaveDensity, points: np.ndarray) -> np.ndarray:
    n_terms = len(rho_obj.rho_g)
    values = np.exp(1j * (points @ rho_obj.G.T)) @ rho_obj.rho_g / n_terms
    return np.real_if_close(values, tol=1e-6)


def test_streamed_ndarray_matches_dense_reference() -> None:
    rho_obj = make_small_density()
    points = np.random.default_rng(7).normal(size=(9, 3))

    actual = rho_obj(points)
    expected = dense_reference(rho_obj, points)

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_generator_input_returns_iterator_and_matches_reference() -> None:
    rho_obj = make_small_density()
    points = np.random.default_rng(8).normal(size=(6, 3))

    streamed = rho_obj((point for point in points))
    assert isinstance(streamed, Iterator)

    actual = np.fromiter(streamed, dtype=float, count=len(points))
    expected = dense_reference(rho_obj, points)

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_batch_chunk_size_is_at_least_one_for_tiny_memory_budget() -> None:
    rho_obj = make_small_density()
    rho_obj.max_batch_memory_mb = 0.0
    assert rho_obj._batch_chunk_size() >= 1
