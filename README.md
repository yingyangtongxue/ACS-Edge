# ACS Edge Detection

[![CI](https://github.com/yingyangtongxue/ACS-Edge/actions/workflows/ci.yml/badge.svg)](https://github.com/yingyangtongxue/ACS-Edge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GPU-accelerated image edge detection via **Ant Colony System (ACS)**, based on the algorithm described in:

> Baterina, A. V., & Oppus, C. M. (2010). **Image edge detection using ant colony optimization.** *WSEAS Transactions on Signal Processing*, 6(2), 58–67.

A copy of the paper is available in [`docs/`](docs/Image_edge_detection_using_ant_colony_optimization.pdf).

---

## Academic Context

This project was developed as coursework for the graduate course **Digital Image Processing** (*Processamento de Imagens Digitais*), offered in the **first semester of 2024** at:

| | |
|---|---|
| **Program** | PPGCC — Graduate Program in Computer Science |
| **Institution** | IBILCE / UNESP — São José do Rio Preto, SP, Brazil |
| **Instructor** | Prof. Dr. Leandro Alves Neves |
| **Workload** | 120 hours |

The goal of the assignment was to implement a swarm-intelligence approach to low-level image processing, demonstrating mastery of the ACS meta-heuristic and its adaptation to pixel-graph traversal problems.

---

## Algorithm Overview

The algorithm models image pixels as nodes in a fully-connected graph. A colony of *K* ants performs a biased random walk guided by two signals:

| Signal | Symbol | Definition |
|---|---|---|
| Pheromone | τ_ij | Accumulated edge evidence at pixel (i, j) |
| Heuristic | η_ij | Local intensity variation (Eq. 7–8) |

### Heuristic (local variation function)

```
η_ij = V_c(I_ij) / V_max

V_c(I_ij) = |I_{i-1,j-1} − I_{i+1,j+1}|   ← ↘ diagonal pair
           + |I_{i−1,j}   − I_{i+1,j}  |   ← ↓  vertical pair
           + |I_{i-1,j+1} − I_{i+1,j-1}|   ← ↙ anti-diagonal pair
           + |I_{i,j−1}   − I_{i,j+1}  |   ← → horizontal pair
```

### Decision rule (ACS pseudo-random proportional)

At each step, ant *k* at position (i, j) draws a uniform random value *q*:

- **q ≤ q₀ (exploitation):** move to *argmax*_{Ω(i,j)} τ · η
- **q > q₀ (exploration):** sample from *p_j = τ_j · η_j / Σ_{k∈Ω} τ_k · η_k*

### Pheromone updates

| Update | Equation | When |
|---|---|---|
| Local (Eq. 10) | τ_ij ← (1−φ)·τ_ij + φ·τ_init | After each ant step |
| Global (Eq. 11) | τ_ij ← (1−ρ)·τ_ij + ρ·η̄_ij | After all construction steps |

After *N* iterations the final pheromone matrix is thresholded with **Otsu's method** to produce a binary edge map.

### Default parameters (Table 1, Baterina & Oppus 2010)

| Parameter | Symbol | Default |
|---|---|---|
| Number of ants | K | H × W (one per pixel) |
| Iterations | N | 10 |
| Construction steps | L | 40 |
| Initial pheromone | τ_init | 0.1 |
| Local decay | φ | 0.05 |
| Global evaporation | ρ | 0.1 |
| Exploitation bias | q₀ | 0.0 – 1.0 (swept) |

---

## Project Structure

```
acs-edge/
├── acs_edge/              # Core library
│   ├── __init__.py        # Public API
│   ├── config.py          # ACSConfig dataclass with parameter validation
│   ├── heuristic.py       # η matrix computation — fully vectorised (Eq. 7–8)
│   └── core.py            # ACSEdgeDetector — GPU/CPU vectorised detector
├── tests/                 # pytest test suite (27 tests)
│   ├── test_config.py
│   ├── test_core.py
│   └── test_heuristic.py
├── docs/                  # Reference paper (PDF)
├── samples/               # Sample input images
│   ├── lena.png           # 256 × 256 — standard benchmark
│   └── pikachu.png        # 50 × 43  — quick-run sample
├── results/               # Runtime output (git-ignored)
├── .github/workflows/     # GitHub Actions CI
├── main.py                # CLI entry point
├── pyproject.toml
└── requirements.txt
```

---

## Installation

**Requirements:** Python ≥ 3.10

```bash
# Clone the repository
git clone https://github.com/yingyangtongxue/ACS-Edge.git
cd ACS-Edge

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

# Install the package with all dependencies
pip install -e ".[dev]"
```

### Optional GPU acceleration

Install the CuPy variant that matches your CUDA version:

```bash
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda11x   # CUDA 11.x
```

The detector falls back to NumPy (CPU) automatically when CuPy is unavailable.

---

## Usage

### Quick run

```bash
python main.py samples/pikachu.png
```

### Full q₀ sweep

```bash
python main.py samples/pikachu.png \
    --q0 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --seed 42
```

### Larger image with explicit ant count

```bash
python main.py samples/lena.png \
    --ants 0 \
    --iterations 10 \
    --steps 40 \
    --seed 42
```

### CLI reference

```
usage: acs-edge [-h] [--ants K] [--iterations N] [--steps L]
                [--q0 Q [Q ...]] [--seed S] [--no-gpu]
                [--output-dir DIR] [-v]
                image

positional arguments:
  image              Path to the input grayscale image

options:
  --ants K           Number of ants; 0 = one per pixel (default: 0)
  --iterations N     Outer ACS iterations (default: 10)
  --steps L          Construction steps per iteration (default: 40)
  --q0 Q [Q ...]     q0 values to evaluate (default: 0.0 … 1.0)
  --seed S           Random seed for reproducibility
  --no-gpu           Force CPU execution
  --output-dir DIR   Output directory (default: results/<timestamp>)
  -v, --verbose      Enable DEBUG logging
```

---

## Running Tests

```bash
pytest tests/ -v
```

```
27 passed in 0.45s
```

---

## Performance

The vectorised implementation replaces the original O(H · W · K · L) global pheromone update with an O(K · L + H · W) scatter-add, providing significant speedups:

| Image | Size | Original | This implementation |
|---|---|---|---|
| pikachu | 50 × 43 | ~3–10 min | < 1 s (CPU) |
| lena | 256 × 256 | > 9 h | ~30 s (CPU) / < 5 s (GPU) |

*GPU timings measured on NVIDIA RTX 3060.*

---

## Reference

```bibtex
@article{baterina2010,
  author  = {Baterina, Anna V. and Oppus, Carlos M.},
  title   = {Image edge detection using ant colony optimization},
  journal = {WSEAS Transactions on Signal Processing},
  volume  = {6},
  number  = {2},
  pages   = {58--67},
  year    = {2010}
}
```
