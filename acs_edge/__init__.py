"""ACS Edge Detection – Baterina & Oppus (2010).

Public API::

    from acs_edge import ACSConfig, ACSEdgeDetector, gpu_available

    config = ACSConfig(num_ants=0, num_iterations=10, seed=42)
    detector = ACSEdgeDetector(config, use_gpu=True)
    pheromone = detector.detect(image_array, q0=0.5)
"""

from .config import ACSConfig
from .core import ACSEdgeDetector, gpu_available
from .heuristic import compute_heuristic

__all__ = ["ACSConfig", "ACSEdgeDetector", "compute_heuristic", "gpu_available"]
