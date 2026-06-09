"""Heuristic information matrix for ACS edge detection.

Implements Equations 7–8 from:
    Baterina & Oppus (2010). Image edge detection using ant colony
    optimization. IAENG International Journal of Computer Science, 37(4).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_heuristic(image: np.ndarray, xp: Any = np) -> Any:
    """Compute the normalised local-variation heuristic matrix η.

    Each pixel's heuristic η_ij = V_c(I_ij) / V_max  (Eq. 7), where:

        V_c(I_ij) = |I_{i-1,j-1} - I_{i+1,j+1}|   (↘ diagonal pair)
                  + |I_{i-1,j}   - I_{i+1,j}  |   (↓  vertical pair)
                  + |I_{i-1,j+1} - I_{i+1,j-1}|   (↙ anti-diagonal pair)
                  + |I_{i,j-1}   - I_{i,j+1}  |   (→  horizontal pair)
                                                    (Eq. 8)

    Border pixels are handled by edge-padding.

    Args:
        image: Grayscale image as a 2-D float array with shape (H, W).
        xp: Array backend – numpy (CPU) or cupy (GPU).  Defaults to numpy.

    Returns:
        η matrix with the same shape as *image*, dtype float32, values in [0, 1].
    """
    img = xp.asarray(image, dtype=xp.float32)
    pad = xp.pad(img, pad_width=1, mode="edge")

    # Opposite-pair absolute differences (Eq. 8 – correct neighbor indices)
    Vc = (
        xp.abs(pad[:-2, :-2] - pad[2:, 2:])    # (r-1,c-1) ↔ (r+1,c+1)
        + xp.abs(pad[:-2, 1:-1] - pad[2:, 1:-1])  # (r-1,c)   ↔ (r+1,c)
        + xp.abs(pad[:-2, 2:] - pad[2:, :-2])    # (r-1,c+1) ↔ (r+1,c-1)
        + xp.abs(pad[1:-1, :-2] - pad[1:-1, 2:])  # (r,c-1)   ↔ (r,c+1)
    )

    v_max = float(Vc.max())
    return Vc / v_max if v_max > 0.0 else Vc
