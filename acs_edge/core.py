"""Vectorized Ant Colony System (ACS) image edge detector.

Implements the algorithm from:
    Baterina & Oppus (2010). Image edge detection using ant colony
    optimization. IAENG International Journal of Computer Science, 37(4).

Key design choices:
- All K ants advance in a single vectorised step (no Python loops over ants).
- Pheromone accumulation uses scatter-add (xp.add.at) instead of nested loops,
  reducing the global-update complexity from O(H·W·K·L) to O(K·L + H·W).
- Backend is selected once: CuPy when a GPU is available, NumPy otherwise.
  The caller does not need to change any code to switch backends.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import ACSConfig
from .heuristic import compute_heuristic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU / CPU backend selection
# ---------------------------------------------------------------------------
try:
    import cupy as _cp  # type: ignore[import]

    _HAS_GPU = True
    logger.debug("CuPy found – GPU acceleration available.")
except ImportError:
    _cp = None
    _HAS_GPU = False
    logger.debug("CuPy not found – falling back to NumPy (CPU).")

# 8-connectivity neighbour offsets (row-delta, col-delta), stored on CPU;
# transferred to the chosen backend inside ACSEdgeDetector.__init__.
_NEIGHBOUR_DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
_NEIGHBOUR_DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)


def gpu_available() -> bool:
    """Return True if CuPy is installed and a CUDA device is accessible."""
    return _HAS_GPU


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------


class ACSEdgeDetector:
    """Vectorized Ant Colony System edge detector with optional GPU acceleration.

    Example::

        config = ACSConfig(num_ants=0, num_iterations=10, seed=42)
        detector = ACSEdgeDetector(config, use_gpu=True)
        pheromone = detector.detect(image_array, q0=0.5)
    """

    def __init__(self, config: ACSConfig, use_gpu: bool = True) -> None:
        """Initialise the detector.

        Args:
            config: Algorithm hyper-parameters.
            use_gpu: Request GPU execution.  Silently falls back to CPU when
                CuPy is unavailable or no CUDA device is found.
        """
        self._cfg = config
        self._xp: Any = (_cp if (use_gpu and _HAS_GPU) else np)
        self.using_gpu: bool = self._xp is not np

        self._dr = self._xp.asarray(_NEIGHBOUR_DR)
        self._dc = self._xp.asarray(_NEIGHBOUR_DC)
        self._rng = np.random.default_rng(config.seed)

        backend_name = "CuPy (GPU)" if self.using_gpu else "NumPy (CPU)"
        logger.info("ACSEdgeDetector initialised – backend: %s", backend_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray, q0: float = 0.5) -> np.ndarray:
        """Run ACS and return the final pheromone matrix.

        Args:
            image: Grayscale image, shape (H, W), float or uint8.
            q0: Exploitation–exploration trade-off parameter in [0, 1].
                q0 → 1.0 favours exploitation; q0 → 0.0 favours exploration.

        Returns:
            Pheromone matrix, shape (H, W), float32, on the CPU.
        """
        if not (0.0 <= q0 <= 1.0):
            raise ValueError(f"q0 must be in [0, 1], got {q0}")

        xp = self._xp
        cfg = self._cfg

        img_f32 = np.asarray(image, dtype=np.float32)
        H, W = img_f32.shape
        K = cfg.resolve_num_ants(H, W)

        # Pre-compute heuristic matrix η once for this image (Eq. 7–8)
        eta: Any = compute_heuristic(img_f32, xp=xp)

        # Initialise pheromone matrix τ (Eq. 10 initial condition)
        tau: Any = xp.full((H, W), cfg.initial_pheromone, dtype=xp.float32)

        # Random initial positions for all ants
        ant_r = xp.asarray(self._rng.integers(0, H, K), dtype=xp.int32)
        ant_c = xp.asarray(self._rng.integers(0, W, K), dtype=xp.int32)
        # Previous positions initialised to current (first step has no history)
        ant_last_r = ant_r.copy()
        ant_last_c = ant_c.copy()

        for _ in tqdm(
            range(cfg.num_iterations),
            desc=f"  ACS iterations (q0={q0:.1f})",
            leave=False,
        ):
            # Accumulators for global pheromone update (Eq. 11)
            visit_count: Any = xp.zeros((H, W), dtype=xp.float32)
            eta_sum: Any = xp.zeros((H, W), dtype=xp.float32)

            for _ in range(cfg.steps_per_iteration):
                ant_r, ant_c, ant_last_r, ant_last_c = self._construction_step(
                    ant_r, ant_c, ant_last_r, ant_last_c,
                    tau, eta, q0, H, W,
                )
                # Scatter-add visit statistics for global update
                xp.add.at(visit_count, (ant_r, ant_c), 1.0)
                xp.add.at(eta_sum, (ant_r, ant_c), eta[ant_r, ant_c])

            # Global pheromone update (Eq. 11)
            delta_tau = xp.where(
                visit_count > 0,
                eta_sum / xp.maximum(visit_count, 1e-12),
                xp.float32(0.0),
            )
            tau = (1.0 - cfg.evaporation_rate) * tau + cfg.evaporation_rate * delta_tau

        return self._to_cpu(tau)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _construction_step(
        self,
        ant_r: Any,
        ant_c: Any,
        ant_last_r: Any,
        ant_last_c: Any,
        tau: Any,
        eta: Any,
        q0: float,
        H: int,
        W: int,
    ) -> tuple[Any, Any, Any, Any]:
        """Advance all K ants by one step (vectorised).

        Decision rule (ACS pseudo-random proportional, Eq. 9):
            if q ≤ q0  →  exploit: choose j = argmax_{Ω_i} τ_j · η_j
            if q > q0  →  explore: sample j ~ τ_j · η_j / Σ τ_k · η_k

        Local pheromone update applied after movement (Eq. 10):
            τ_ij ← (1 − φ) · τ_ij + φ · τ_init
        """
        xp = self._xp
        K = len(ant_r)

        # --- Compute 8-neighbour positions for every ant: shape [K, 8] ---
        neigh_r = ant_r[:, None] + self._dr[None, :]  # [K, 8]
        neigh_c = ant_c[:, None] + self._dc[None, :]  # [K, 8]

        # Validity mask: in-bounds AND not the previously visited cell
        in_bounds = (
            (neigh_r >= 0) & (neigh_r < H) & (neigh_c >= 0) & (neigh_c < W)
        )
        not_last = ~(
            (neigh_r == ant_last_r[:, None]) & (neigh_c == ant_last_c[:, None])
        )
        mask = (in_bounds & not_last).astype(xp.float32)  # [K, 8]

        # Safe clipped indices for array lookup (out-of-bounds → 0 via mask)
        r_safe = xp.clip(neigh_r, 0, H - 1)
        c_safe = xp.clip(neigh_c, 0, W - 1)

        # Decision values: τ_j · η_j (α = β = 1 per paper), masked: [K, 8]
        decision = tau[r_safe, c_safe] * eta[r_safe, c_safe] * mask

        # --- Exploitation: argmax τ·η (q ≤ q0) ---
        exploit_idx = xp.argmax(decision, axis=1)  # [K]

        # --- Exploration: sample proportional to τ·η (q > q0, Eq. 9) ---
        row_sums = decision.sum(axis=1, keepdims=True)
        safe_sums = xp.maximum(row_sums, 1e-12)
        probs = decision / safe_sums  # [K, 8] – normalised probabilities

        # Vectorised categorical sampling via CDF inversion
        cumprobs = xp.cumsum(probs, axis=1)  # [K, 8]
        u = xp.asarray(
            self._rng.random(K), dtype=xp.float32
        )[:, None]  # [K, 1]
        explore_idx = xp.clip(
            (cumprobs < u).sum(axis=1), 0, 7
        )  # [K]

        # --- Mix exploitation and exploration per ant ---
        q_vals = xp.asarray(self._rng.random(K), dtype=xp.float32)
        use_exploit = q_vals <= q0
        chosen_idx = xp.where(use_exploit, exploit_idx, explore_idx)  # [K]

        # Gather new positions
        arange_k = xp.arange(K, dtype=xp.int32)
        new_r = xp.clip(neigh_r[arange_k, chosen_idx], 0, H - 1)
        new_c = xp.clip(neigh_c[arange_k, chosen_idx], 0, W - 1)

        # Local pheromone update at newly visited pixels (Eq. 10)
        tau[new_r, new_c] = (
            (1.0 - self._cfg.decay_coefficient) * tau[new_r, new_c]
            + self._cfg.decay_coefficient * self._cfg.initial_pheromone
        )

        return new_r, new_c, ant_r, ant_c

    def _to_cpu(self, arr: Any) -> np.ndarray:
        """Transfer array from GPU to CPU if necessary."""
        if self.using_gpu:
            return _cp.asnumpy(arr)  # type: ignore[union-attr]
        return np.asarray(arr)
