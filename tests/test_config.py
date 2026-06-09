"""Tests for acs_edge.config.ACSConfig."""

import pytest

from acs_edge.config import ACSConfig


class TestACSConfig:
    def test_default_values(self):
        cfg = ACSConfig()
        assert cfg.num_ants == 0
        assert cfg.num_iterations == 10
        assert cfg.steps_per_iteration == 40
        assert cfg.initial_pheromone == pytest.approx(0.1)
        assert cfg.decay_coefficient == pytest.approx(0.05)
        assert cfg.evaporation_rate == pytest.approx(0.1)
        assert cfg.seed is None

    def test_resolve_num_ants_zero_scales_with_area(self):
        cfg = ACSConfig(num_ants=0)
        # 256×256 must match the paper's K=512
        assert cfg.resolve_num_ants(256, 256) == 512
        # 512×512 is 4× the area → 4×512 = 2048
        assert cfg.resolve_num_ants(512, 512) == 2048
        # Very small image is clamped to 64
        assert cfg.resolve_num_ants(8, 10) == 64

    def test_resolve_num_ants_explicit(self):
        cfg = ACSConfig(num_ants=50)
        assert cfg.resolve_num_ants(8, 10) == 50

    def test_negative_num_ants_raises(self):
        with pytest.raises(ValueError, match="num_ants"):
            ACSConfig(num_ants=-1)

    def test_zero_iterations_raises(self):
        with pytest.raises(ValueError, match="num_iterations"):
            ACSConfig(num_iterations=0)

    def test_zero_steps_raises(self):
        with pytest.raises(ValueError, match="steps_per_iteration"):
            ACSConfig(steps_per_iteration=0)

    def test_initial_pheromone_out_of_range_raises(self):
        with pytest.raises(ValueError, match="initial_pheromone"):
            ACSConfig(initial_pheromone=0.0)
        with pytest.raises(ValueError, match="initial_pheromone"):
            ACSConfig(initial_pheromone=1.5)

    def test_decay_coefficient_out_of_range_raises(self):
        with pytest.raises(ValueError, match="decay_coefficient"):
            ACSConfig(decay_coefficient=0.0)
        with pytest.raises(ValueError, match="decay_coefficient"):
            ACSConfig(decay_coefficient=1.1)

    def test_evaporation_rate_out_of_range_raises(self):
        with pytest.raises(ValueError, match="evaporation_rate"):
            ACSConfig(evaporation_rate=0.0)
        with pytest.raises(ValueError, match="evaporation_rate"):
            ACSConfig(evaporation_rate=2.0)

    def test_seed_is_propagated(self):
        cfg = ACSConfig(seed=123)
        assert cfg.seed == 123
