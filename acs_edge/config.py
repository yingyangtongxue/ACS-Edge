"""ACS edge detection algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ACSConfig:
    """Parameters for the Ant Colony System edge detector.

    Default values follow Baterina & Oppus (2010), Table 1.

    Attributes:
        num_ants: Number of ants; 0 uses the paper default of 512.
        num_iterations: Outer loop count (N in the paper).
        steps_per_iteration: Construction steps per iteration (L in the paper).
        initial_pheromone: τ_init – initial and reset pheromone level.
        decay_coefficient: φ – local pheromone decay rate (Eq. 10).
        evaporation_rate: ρ – global pheromone evaporation rate (Eq. 11).
        seed: Random seed for reproducibility; None means non-deterministic.
    """

    num_ants: int = 0
    num_iterations: int = 10
    steps_per_iteration: int = 40
    initial_pheromone: float = 0.1
    decay_coefficient: float = 0.05
    evaporation_rate: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_ants < 0:
            raise ValueError(f"num_ants must be >= 0, got {self.num_ants}")
        if self.num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {self.num_iterations}")
        if self.steps_per_iteration < 1:
            raise ValueError(
                f"steps_per_iteration must be >= 1, got {self.steps_per_iteration}"
            )
        if not (0.0 < self.initial_pheromone <= 1.0):
            raise ValueError(
                f"initial_pheromone must be in (0, 1], got {self.initial_pheromone}"
            )
        if not (0.0 < self.decay_coefficient <= 1.0):
            raise ValueError(
                f"decay_coefficient must be in (0, 1], got {self.decay_coefficient}"
            )
        if not (0.0 < self.evaporation_rate <= 1.0):
            raise ValueError(
                f"evaporation_rate must be in (0, 1], got {self.evaporation_rate}"
            )

    def resolve_num_ants(self, height: int, width: int) -> int:
        """Return the effective ant count.

        0 resolves to 512, matching the value used in Baterina & Oppus (2010)
        for a 256×256 image.  Using one ant per pixel (H*W) floods every cell
        with visits, making pheromone uniform and Otsu thresholding ineffective.
        """
        return 512 if self.num_ants == 0 else self.num_ants
