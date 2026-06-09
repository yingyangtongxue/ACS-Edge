"""Integration tests for acs_edge.core.ACSEdgeDetector."""

import numpy as np
import pytest
from skimage import filters

from acs_edge import ACSConfig, ACSEdgeDetector


@pytest.fixture
def tiny_image():
    """7×7 synthetic image with a clear vertical edge."""
    img = np.zeros((7, 7), dtype=np.float32)
    img[:, 3:] = 200.0
    return img


@pytest.fixture
def noise_image():
    """10×10 random noise image (reproducible)."""
    return np.random.default_rng(99).integers(0, 256, (10, 10)).astype(np.float32)


@pytest.fixture
def fast_config():
    """Minimal config for fast tests."""
    return ACSConfig(
        num_ants=10,
        num_iterations=2,
        steps_per_iteration=5,
        seed=0,
    )


class TestACSEdgeDetectorOutput:
    def test_pheromone_shape_matches_image(self, tiny_image, fast_config):
        detector = ACSEdgeDetector(fast_config, use_gpu=False)
        pheromone = detector.detect(tiny_image, q0=0.5)
        assert pheromone.shape == tiny_image.shape

    def test_pheromone_dtype_is_float32(self, tiny_image, fast_config):
        detector = ACSEdgeDetector(fast_config, use_gpu=False)
        pheromone = detector.detect(tiny_image, q0=0.5)
        assert pheromone.dtype == np.float32

    def test_pheromone_is_positive(self, tiny_image, fast_config):
        detector = ACSEdgeDetector(fast_config, use_gpu=False)
        pheromone = detector.detect(tiny_image, q0=0.5)
        assert float(pheromone.min()) >= 0.0

    def test_pheromone_is_cpu_array(self, noise_image, fast_config):
        detector = ACSEdgeDetector(fast_config, use_gpu=False)
        pheromone = detector.detect(noise_image, q0=0.5)
        assert isinstance(pheromone, np.ndarray)

    def test_otsu_produces_binary_output(self, tiny_image, fast_config):
        detector = ACSEdgeDetector(fast_config, use_gpu=False)
        pheromone = detector.detect(tiny_image, q0=0.5)
        threshold = filters.threshold_otsu(pheromone)
        binary = (pheromone >= threshold).astype(np.uint8)
        unique_values = set(binary.flatten().tolist())
        assert unique_values.issubset({0, 1})


class TestACSEdgeDetectorBehaviour:
    def test_reproducible_with_seed(self, noise_image):
        cfg = ACSConfig(num_ants=10, num_iterations=3, steps_per_iteration=5, seed=42)
        d1 = ACSEdgeDetector(cfg, use_gpu=False)
        d2 = ACSEdgeDetector(cfg, use_gpu=False)
        p1 = d1.detect(noise_image, q0=0.5)
        p2 = d2.detect(noise_image, q0=0.5)
        np.testing.assert_array_equal(p1, p2)

    def test_full_exploitation_differs_from_full_exploration(self, tiny_image):
        cfg = ACSConfig(num_ants=20, num_iterations=3, steps_per_iteration=8, seed=7)
        det = ACSEdgeDetector(cfg, use_gpu=False)
        p_exploit = det.detect(tiny_image, q0=1.0)

        cfg2 = ACSConfig(num_ants=20, num_iterations=3, steps_per_iteration=8, seed=7)
        det2 = ACSEdgeDetector(cfg2, use_gpu=False)
        p_explore = det2.detect(tiny_image, q0=0.0)

        assert not np.array_equal(p_exploit, p_explore)

    def test_invalid_q0_raises(self, noise_image, fast_config):
        detector = ACSEdgeDetector(fast_config, use_gpu=False)
        with pytest.raises(ValueError, match="q0"):
            detector.detect(noise_image, q0=1.5)
        with pytest.raises(ValueError, match="q0"):
            detector.detect(noise_image, q0=-0.1)

    def test_uint8_image_accepted(self, fast_config):
        img = np.random.default_rng(5).integers(0, 256, (8, 8)).astype(np.uint8)
        detector = ACSEdgeDetector(fast_config, use_gpu=False)
        pheromone = detector.detect(img, q0=0.5)
        assert pheromone.shape == (8, 8)

    def test_using_gpu_flag_false_without_cupy(self, fast_config):
        """Detector should report using_gpu=False when GPU is unavailable."""
        detector = ACSEdgeDetector(fast_config, use_gpu=False)
        assert detector.using_gpu is False
