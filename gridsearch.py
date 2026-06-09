"""Grid search over ACS hyper-parameters for a single image.

Sweeps num_ants × num_iterations × steps_per_iteration for a fixed set of
q0 values and saves every result image together with a CSV summary so the
best parameters can be identified visually and quantitatively.

Usage::

    python gridsearch.py samples/lena.png
    python gridsearch.py samples/lena.png --q0 0.3 0.5 0.7 --seed 42
    python gridsearch.py samples/lena.png --no-gpu
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters
from tqdm import tqdm

from acs_edge import ACSConfig, ACSEdgeDetector, gpu_available


# ---------------------------------------------------------------------------
# Default search grid
# ---------------------------------------------------------------------------

# Ant counts scaled around the paper ratio:
#   K=512 for 256×256 → ~0.0078 ants/pixel.
# Four candidate multipliers: ×0.5, ×1, ×2, ×4 relative to that density
# evaluated at the actual image size.
_ANT_SCALE_FACTORS = [0.5, 1.0, 2.0, 4.0]

_ITERATIONS_GRID = [10, 20]
_STEPS_GRID = [40, 80]
_DEFAULT_Q0 = [0.3, 0.5, 0.7]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edge_density(binary: np.ndarray) -> float:
    return float(binary.mean())


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def _save(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="acs-gridsearch",
        description="Grid search over ACS hyper-parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--q0",
        type=float,
        nargs="+",
        default=_DEFAULT_Q0,
        metavar="Q",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="S")
    parser.add_argument("--no-gpu", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=None, metavar="DIR")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    log = logging.getLogger(__name__)

    if not args.image.is_file():
        log.error("Image not found: %s", args.image)
        return 1

    image = _load_image(args.image)
    H, W = image.shape
    log.info("Image: %s  (%d × %d px)", args.image.name, W, H)

    use_gpu = not args.no_gpu
    if use_gpu and not gpu_available():
        log.warning("CuPy not available – running on CPU.")
        use_gpu = False

    # Derive ant counts from scale factors relative to the paper ratio
    # (512 ants for 256×256 = 1 ant per ~128 pixels)
    pixels = H * W
    paper_ratio = 512 / (256 * 256)
    ant_counts = sorted({max(64, int(paper_ratio * pixels * f)) for f in _ANT_SCALE_FACTORS})
    log.info("Ant counts to try: %s", ant_counts)

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%dT%Hh%Mmin%Sseg")
        output_dir = Path("results") / f"gridsearch_{args.image.stem}_{ts}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    q0_values = sorted(set(args.q0))
    combinations = list(product(ant_counts, _ITERATIONS_GRID, _STEPS_GRID, q0_values))
    log.info("Total runs: %d", len(combinations))

    csv_path = output_dir / "results.csv"
    csv_rows: list[dict] = []

    with tqdm(total=len(combinations), desc="Grid search", unit="run") as pbar:
        for ants, iters, steps, q0 in combinations:
            cfg = ACSConfig(
                num_ants=ants,
                num_iterations=iters,
                steps_per_iteration=steps,
                seed=args.seed,
            )
            detector = ACSEdgeDetector(cfg, use_gpu=use_gpu)
            pheromone = detector.detect(image, q0=q0)

            threshold = filters.threshold_otsu(pheromone)
            edge_map = ((pheromone >= threshold) * 255).astype(np.uint8)
            density = _edge_density(edge_map > 0)

            label = f"k{ants}_n{iters}_l{steps}_q{q0:.1f}"
            img_path = output_dir / f"{args.image.stem}_{label}.png"
            _save(edge_map, img_path)

            csv_rows.append(
                {
                    "ants": ants,
                    "iterations": iters,
                    "steps": steps,
                    "q0": q0,
                    "otsu_threshold": round(float(threshold), 6),
                    "edge_density": round(density, 4),
                    "file": img_path.name,
                }
            )

            pbar.set_postfix(
                ants=ants, iters=iters, steps=steps, q0=f"{q0:.1f}", density=f"{density:.1%}"
            )
            pbar.update()

    # Write CSV summary
    fieldnames = ["ants", "iterations", "steps", "q0", "otsu_threshold", "edge_density", "file"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print top-10 by lowest edge density (closest to sparse true edges)
    csv_rows.sort(key=lambda r: r["edge_density"])
    log.info("")
    log.info("Top 10 runs by lowest edge density (sparser = fewer false positives):")
    log.info("%-6s %-5s %-6s %-5s  %-8s  %s", "ants", "iters", "steps", "q0", "density", "file")
    log.info("%s", "-" * 70)
    for row in csv_rows[:10]:
        log.info(
            "%-6d %-5d %-6d %-5.1f  %-7s  %s",
            row["ants"], row["iterations"], row["steps"], row["q0"],
            f"{row['edge_density']:.1%}", row["file"],
        )

    log.info("")
    log.info("Full results saved to %s", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
