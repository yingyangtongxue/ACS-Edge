"""CLI entry point for ACS-based image edge detection.

Usage::

    python main.py Input/pikachu.png
    python main.py Input/lena.png --ants 0 --iterations 10 --seed 42
    python main.py Input/pikachu.png --q0 0.0 0.3 0.5 0.7 1.0 --no-gpu

Results are saved under Output/<ISO-timestamp>/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters
from tqdm import tqdm

from acs_edge import ACSConfig, ACSEdgeDetector, gpu_available


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acs-edge",
        description=(
            "ACS Image Edge Detection – "
            "Baterina & Oppus (2010), IAENG IJCS 37(4)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="Path to the input grayscale image.")
    parser.add_argument(
        "--ants",
        type=int,
        default=0,
        metavar="K",
        help="Number of ants (0 = one per pixel, as in the paper).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        metavar="N",
        help="Number of ACS outer iterations.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        metavar="L",
        help="Construction steps per iteration per ant.",
    )
    parser.add_argument(
        "--q0",
        type=float,
        nargs="+",
        default=[round(i / 10, 1) for i in range(11)],
        metavar="Q",
        help="One or more q0 values to evaluate (exploitation–exploration ratio).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Disable GPU acceleration (force NumPy/CPU).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: Output/<timestamp>).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    return parser


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def _save_edge_map(edge_map: np.ndarray, path: Path) -> None:
    Image.fromarray(edge_map.astype(np.uint8)).save(path)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    # --- Load image ---
    if not args.image.is_file():
        log.error("Image not found: %s", args.image)
        return 1

    image = _load_image(args.image)
    H, W = image.shape
    log.info("Image: %s  (%d × %d px)", args.image.name, W, H)

    # --- Configure detector ---
    use_gpu = not args.no_gpu
    if use_gpu and not gpu_available():
        log.warning("CuPy not available – running on CPU.")
        use_gpu = False

    config = ACSConfig(
        num_ants=args.ants,
        num_iterations=args.iterations,
        steps_per_iteration=args.steps,
        seed=args.seed,
    )
    effective_k = config.resolve_num_ants(H, W)
    detector = ACSEdgeDetector(config, use_gpu=use_gpu)

    log.info(
        "Config: ants=%d  iterations=%d  steps=%d  backend=%s",
        effective_k,
        args.iterations,
        args.steps,
        "GPU" if detector.using_gpu else "CPU",
    )

    # --- Output directory ---
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%dT%Hh%Mmin%Sseg")
        output_dir = Path("Output") / timestamp
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    # --- q0 sweep ---
    q0_values: list[float] = sorted(set(args.q0))
    results = []

    for q0 in tqdm(q0_values, desc="q0 sweep", unit="run"):
        pheromone = detector.detect(image, q0=q0)

        threshold = filters.threshold_otsu(pheromone)
        edge_map = ((pheromone >= threshold) * 255).astype(np.uint8)

        stem = args.image.stem
        out_path = output_dir / f"{stem}_q0={q0:.1f}.png"
        _save_edge_map(edge_map, out_path)

        results.append((q0, threshold, out_path))

    # --- Summary table ---
    log.info("")
    log.info("%-6s  %-12s  %s", "q0", "Otsu thresh", "Output file")
    log.info("%s", "-" * 60)
    for q0, thresh, path in results:
        log.info("%-6.1f  %-12.6f  %s", q0, thresh, path.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
