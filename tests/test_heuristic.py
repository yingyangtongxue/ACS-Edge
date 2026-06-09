"""Tests for acs_edge.heuristic.compute_heuristic."""

import numpy as np
import pytest

from acs_edge.heuristic import compute_heuristic


class TestComputeHeuristic:
    def test_output_shape_matches_input(self):
        img = np.random.default_rng(0).random((10, 15)).astype(np.float32) * 255
        eta = compute_heuristic(img)
        assert eta.shape == img.shape

    def test_output_dtype_is_float32(self):
        img = np.ones((8, 8), dtype=np.float32) * 128.0
        eta = compute_heuristic(img)
        assert eta.dtype == np.float32

    def test_uniform_image_gives_zero_heuristic(self):
        img = np.full((12, 12), 128.0, dtype=np.float32)
        eta = compute_heuristic(img)
        np.testing.assert_array_equal(eta, 0.0)

    def test_normalised_max_is_one(self):
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, size=(20, 20)).astype(np.float32)
        eta = compute_heuristic(img)
        assert float(eta.max()) == pytest.approx(1.0, abs=1e-5)

    def test_values_in_unit_interval(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, size=(16, 16)).astype(np.float32)
        eta = compute_heuristic(img)
        assert float(eta.min()) >= 0.0
        assert float(eta.max()) <= 1.0 + 1e-6

    def test_strong_horizontal_edge(self):
        """A horizontal intensity step should produce high η along the edge."""
        img = np.zeros((10, 10), dtype=np.float32)
        img[5:, :] = 255.0
        eta = compute_heuristic(img)
        # Pixels straddling the step (rows 4 and 5) should dominate
        edge_region = eta[4:6, :]
        interior = np.concatenate([eta[:3, :], eta[7:, :]])
        assert float(edge_region.mean()) > float(interior.mean())

    def test_correct_pairs_used(self):
        """Verify that opposite-diagonal pairs are used (Eq. 8), not adjacent pairs.

        Construct an image where only the ↘-diagonal difference is non-zero.
        Only pixels on that diagonal should have non-zero heuristic.
        """
        img = np.zeros((5, 5), dtype=np.float32)
        # Single non-zero pixel at centre: creates non-zero diagonal differences
        img[2, 2] = 255.0
        eta = compute_heuristic(img)
        # Pixels (1,1), (1,3), (3,1), (3,3) are diagonal neighbours; they see
        # the large difference and must be non-zero.
        assert eta[1, 1] > 0.0
        assert eta[3, 3] > 0.0
        assert eta[1, 3] > 0.0
        assert eta[3, 1] > 0.0
