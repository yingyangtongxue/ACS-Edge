#!/usr/bin/env python3
"""LinkedIn carousel — ACS Edge Detection.  Run from repo root."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

OUT   = Path("linkedin")
OUT.mkdir(exist_ok=True)

BG    = "#0d1117"
CARD  = "#161b22"
BDR   = "#30363d"
FG    = "#e6edf3"
MT    = "#8b949e"
BL    = "#58a6ff"
GR    = "#3fb950"
AM    = "#d29922"
TOTAL = 6
DPI   = 108


def new_slide():
    fig = plt.figure(figsize=(10, 10), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def hdr(ax, title, sub=None):
    ax.text(0.5, 0.947, title, ha="center", va="center",
            fontsize=30, color=FG, fontweight="bold")
    if sub:
        ax.text(0.5, 0.897, sub, ha="center", va="center",
                fontsize=13.5, color=MT)
        dy = 0.862
    else:
        dy = 0.907
    ax.axhline(dy, color=BDR, lw=0.7, xmin=0.04, xmax=0.96)
    return dy


def ftr(ax, n):
    ax.text(0.5, 0.030, f"{n} / {TOTAL}", ha="center", va="center",
            fontsize=11, color=MT)


def box(ax, x0, y0, w, h, fc=CARD, ec=BDR, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.008",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))


def abox(ax, x0, y0, w, h, accent=BL):
    box(ax, x0, y0, w, h)
    ax.add_patch(Rectangle(
        (x0 + 0.003, y0 + 0.005), 0.011, h - 0.010,
        facecolor=accent, edgecolor="none", zorder=3))


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Hook
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_slide()
ftr(ax, 1)

ax.text(0.5, 0.800, "8.7  hours", ha="center", va="center",
        fontsize=46, color=MT, fontweight="bold", family="monospace")

ax.annotate("", xy=(0.5, 0.700), xytext=(0.5, 0.745),
            arrowprops=dict(arrowstyle="-|>", color=BL, lw=4.0))

ax.text(0.5, 0.610, "628 ms", ha="center", va="center",
        fontsize=90, color=FG, fontweight="bold", family="monospace")

ax.text(0.5, 0.470, "~50,000 x  faster", ha="center", va="center",
        fontsize=36, color=GR, fontweight="bold")

ax.axhline(0.390, color=BDR, lw=0.8, xmin=0.15, xmax=0.85)

ax.text(0.5, 0.343, "Same algorithm.  Same hardware.",
        ha="center", va="center", fontsize=15, color=MT)
ax.text(0.5, 0.300, "Python loops  replaced by  vectorised NumPy.",
        ha="center", va="center", fontsize=15, color=FG)
ax.text(0.5, 0.248, "Ant Colony System for edge detection  |  PPGCC / UNESP",
        ha="center", va="center", fontsize=12, color=MT)

fig.savefig(OUT / "slide_01.png", dpi=DPI)
plt.close()
print("1/6")


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Algorithm overview
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_slide()
dy = hdr(ax, "Ant Colony System", "for image edge detection")
ftr(ax, 2)

steps = [
    (BL, "1", "Ants start at random pixels",
     "Walk through 8-neighbours guided by intensity gradient;\n"
     "sharper edges attract more ants."),
    (AM, "2", "Each step deposits pheromone",
     "Local update evaporates then redeposits based on gradient;\n"
     "prevents all ants converging to a single path."),
    (GR, "3", "Threshold the pheromone matrix",
     "After N iterations, Otsu's method on the pheromone map\n"
     "produces the final binary edge image."),
]

CH  = 0.215
GAP = 0.028
MX  = 0.050
CW  = 1.0 - 2 * MX
total = 3 * CH + 2 * GAP
start = dy - (dy - 0.060 - total) / 2

for i, (accent, num, title, body) in enumerate(steps):
    cy0 = start - i * (CH + GAP) - CH
    abox(ax, MX, cy0, CW, CH, accent=accent)
    ax.add_patch(plt.Circle((MX + 0.060, cy0 + CH * 0.50), 0.033,
                             color=accent, zorder=4))
    ax.text(MX + 0.060, cy0 + CH * 0.50, num,
            ha="center", va="center", fontsize=14,
            color=BG, fontweight="bold", zorder=5)
    ax.text(MX + 0.120, cy0 + CH * 0.67, title,
            ha="left", va="center", fontsize=16,
            color=FG, fontweight="bold", zorder=4)
    ax.text(MX + 0.120, cy0 + CH * 0.26, body,
            ha="left", va="center", fontsize=13,
            color=MT, linespacing=1.55, zorder=4)

fig.savefig(OUT / "slide_02.png", dpi=DPI)
plt.close()
print("2/6")


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Pheromone mechanics
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_slide()
dy = hdr(ax, "How the pheromone works")
ftr(ax, 3)

sections = [
    (BL, "LOCAL UPDATE   every ant step",
     "After each move, the visited pixel is partially evaporated,\n"
     "then redeposited based on local gradient.\n"
     "This prevents all ants from locking onto the same path."),
    (AM, "GLOBAL UPDATE   after each iteration",
     "Pixels many ants visited get a strong reinforcement,\n"
     "weighted by gradient strength at that pixel.\n"
     "Unvisited pixels continue fading toward zero."),
    (GR, "q0   exploitation vs exploration",
     "q <= q0  :  ant picks the best neighbour greedily.\n"
     "q  > q0  :  ant samples probabilistically from the neighbourhood.\n"
     "Low q0: diffuse edges.    High q0: sharp, concentrated paths."),
]

CH  = 0.215
GAP = 0.025
MX  = 0.050
CW  = 1.0 - 2 * MX
total = 3 * CH + 2 * GAP
start = dy - (dy - 0.060 - total) / 2

for i, (accent, title, body) in enumerate(sections):
    cy0 = start - i * (CH + GAP) - CH
    abox(ax, MX, cy0, CW, CH, accent=accent)
    ax.text(MX + 0.040, cy0 + CH * 0.70, title,
            ha="left", va="center", fontsize=14,
            color=accent, fontweight="bold", zorder=4)
    ax.text(MX + 0.040, cy0 + CH * 0.27, body,
            ha="left", va="center", fontsize=12,
            color=FG, linespacing=1.55, zorder=4)

fig.savefig(OUT / "slide_03.png", dpi=DPI)
plt.close()
print("3/6")


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Visual comparison
# ═══════════════════════════════════════════════════════════════════════════════
def load_sq(path, size=256):
    arr = np.asarray(Image.open(path).convert("L"))
    method = Image.NEAREST if max(arr.shape) < 100 else Image.LANCZOS
    return np.asarray(Image.fromarray(arr).resize((size, size), resample=method))


imgs = {k: load_sq(p) for k, p in [
    ("pi_in",  "docs/images/pikachu_input.png"),
    ("pi_old", "docs/images/pikachu_old_default.png"),
    ("pi_new", "docs/images/pikachu_new_default.png"),
    ("pi_tun", "docs/images/pikachu_tuned.png"),
    ("le_in",  "docs/images/lena_input.png"),
    ("le_old", "docs/images/lena_old_default.png"),
    ("le_new", "docs/images/lena_new_default.png"),
    ("le_tun", "docs/images/lena_tuned.png"),
]}

fig = plt.figure(figsize=(10, 10), facecolor=BG)
fig.text(0.5, 0.958, "Same input — three configurations",
         ha="center", va="center", fontsize=22, color=FG, fontweight="bold")
fig.text(0.5, 0.912, "input  /  legacy  /  new default  /  tuned",
         ha="center", va="center", fontsize=13, color=MT)
ax_div = fig.add_axes([0.04, 0.880, 0.92, 0.001], facecolor=BDR)
ax_div.axis("off")
fig.text(0.5, 0.030, f"4 / {TOTAL}", ha="center", va="center",
         fontsize=11, color=MT)

LBL  = 0.068
ML   = 0.018
MR   = 0.012
IGAP = 0.010
CTOP = 0.872
CBOT = 0.055
CLBH = 0.048
RGAP = 0.018
avail = CTOP - CBOT - CLBH - RGAP
rh    = avail / 2
iw    = (1.0 - LBL - ML - MR - 3 * IGAP) / 4

row1_bot = CTOP - CLBH - rh
row2_bot = row1_bot - RGAP - rh

col_labels = [
    ("Input",                FG),
    ("Legacy\nK=512  q0=0.5", MT),
    ("New default\nK=2048",  BL),
    ("Tuned\nK=1024  q0=0.7", GR),
]
for c, (lbl, col) in enumerate(col_labels):
    cx = LBL + ML + c * (iw + IGAP) + iw / 2
    fig.text(cx, CTOP - CLBH / 2, lbl, ha="center", va="center",
             fontsize=10, color=col, linespacing=1.3)

for label, keys, rb in [
    ("pikachu\n50x43", ["pi_in", "pi_old", "pi_new", "pi_tun"], row1_bot),
    ("lena\n512x512",  ["le_in", "le_old", "le_new", "le_tun"], row2_bot),
]:
    fig.text(LBL / 2, rb + rh / 2, label,
             ha="center", va="center", fontsize=10, color=MT, rotation=90)
    for c, key in enumerate(keys):
        lx = LBL + ML + c * (iw + IGAP)
        ax_i = fig.add_axes([lx, rb, iw, rh])
        ax_i.imshow(imgs[key], cmap="gray", aspect="auto", interpolation="nearest")
        ax_i.axis("off")

fig.savefig(OUT / "slide_04.png", dpi=DPI)
plt.close()
print("4/6")


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_slide()
dy = hdr(ax, "Benchmark")
ftr(ax, 5)

SC_H = 0.220
SC_Y = dy - 0.025 - SC_H
SC_W = (0.90 - 0.030) / 2
SX1  = 0.05
SX2  = SX1 + SC_W + 0.030

for sx, big, label, detail in [
    (SX1, "~320x",    "pikachu  50x43",  "18.9 s  ->  59 ms"),
    (SX2, "~50,000x", "lena  512x512",   "~8.7 h  ->  628 ms"),
]:
    box(ax, sx, SC_Y, SC_W, SC_H)
    ax.text(sx + SC_W / 2, SC_Y + SC_H * 0.72, big,
            ha="center", va="center", fontsize=38, color=GR, fontweight="bold")
    ax.text(sx + SC_W / 2, SC_Y + SC_H * 0.40, label,
            ha="center", va="center", fontsize=12, color=MT)
    ax.text(sx + SC_W / 2, SC_Y + SC_H * 0.16, detail,
            ha="center", va="center", fontsize=11, color=FG, family="monospace")

ty   = SC_Y - 0.048
hdrs = ["Image",       "Original",  "New (CPU)", "Speedup"]
rows = [
    ["pikachu  50x43", "18.9 s *",  "59 ms",     "~320x"],
    ["lena  512x512",  "~8.7 h **", "628 ms",    "~50,000x"],
]
cx  = [0.05, 0.37, 0.59, 0.79]
aln = ["left", "right", "right", "right"]

for i, h in enumerate(hdrs):
    ax.text(cx[i], ty, h, ha=aln[i], va="center",
            fontsize=13.5, color=BL, fontweight="bold")
ax.axhline(ty - 0.026, color=BDR, lw=0.6, xmin=0.04, xmax=0.96)
ty -= 0.078

for row in rows:
    for i, cell in enumerate(row):
        col = GR if ("x" in cell and "~" in cell) else FG
        ax.text(cx[i], ty, cell, ha=aln[i], va="center", fontsize=13, color=col)
    ax.axhline(ty - 0.024, color=BDR, lw=0.4, alpha=0.4, xmin=0.04, xmax=0.96)
    ty -= 0.074

ty -= 0.010
ax.text(0.05, ty, "*  measured on this hardware  (Python 3.13, Ryzen 7 4800H)",
        ha="left", va="center", fontsize=11, color=MT, style="italic")
ty -= 0.044
ax.text(0.05, ty, "**  extrapolated via  O(H x W x K x histLen x N)",
        ha="left", va="center", fontsize=11, color=MT, style="italic")

ty -= 0.060
box(ax, 0.04, 0.058, 0.92, ty - 0.058)
ax.text(0.100, ty - 0.025, "GPU available  (CuPy / CUDA)",
        ha="left", va="center", fontsize=13, color=BL, fontweight="bold", zorder=3)
ax.text(0.100, ty - 0.075,
        "CPU wins for these image sizes on GTX 1650 Ti.",
        ha="left", va="center", fontsize=12, color=FG, zorder=3)
ax.text(0.100, ty - 0.115,
        "Kernel-launch overhead dominates below a few thousand pixels.",
        ha="left", va="center", fontsize=11, color=MT, zorder=3)

fig.savefig(OUT / "slide_05.png", dpi=DPI)
plt.close()
print("5/6")


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Quality + GitHub CTA  (no "Still iterating" section)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = new_slide()
dy = hdr(ax, "Segmentation quality",
         "F1 score vs Canny edges (proxy ground truth)")
ftr(ax, 6)

qh = ["Image",   "Legacy",  "New default", "Tuned",   "vs legacy"]
qr = [
    ["pikachu",  "57.9%",   "59.4%",       "61.7%",   "+6.6 pp"],
    ["lena",     "22.0%",   "37.4%",       "27.1%",   "+70%"],
]
qx = [0.05, 0.27, 0.46, 0.65, 0.82]
qa = ["left", "right", "right", "right", "right"]

ty = dy - 0.028
for i, h in enumerate(qh):
    ax.text(qx[i], ty, h, ha=qa[i], va="center",
            fontsize=13.5, color=BL, fontweight="bold")
ax.axhline(ty - 0.024, color=BDR, lw=0.6, xmin=0.04, xmax=0.96)
ty -= 0.068

for row in qr:
    for i, cell in enumerate(row):
        col = GR if "+" in cell else FG
        ax.text(qx[i], ty, cell, ha=qa[i], va="center", fontsize=14, color=col)
    ax.axhline(ty - 0.022, color=BDR, lw=0.4, alpha=0.4, xmin=0.04, xmax=0.96)
    ty -= 0.062

ax.text(0.5, ty - 0.022,
        "For lena, the new default (K=2048) maximises F1 through better recall.",
        ha="center", va="center", fontsize=11.5, color=MT)
ax.text(0.5, ty - 0.062,
        "The tuned variant (K=1024, q0=0.7) maximises precision at the cost of recall.",
        ha="center", va="center", fontsize=11.5, color=MT)

# CTA block — centred in the lower portion
ax.axhline(0.400, color=BDR, lw=0.8, xmin=0.04, xmax=0.96)

ax.text(0.5, 0.332, "Code, benchmarks, and full gridsearch results:",
        ha="center", va="center", fontsize=14, color=MT)
ax.text(0.5, 0.265, "github.com/yingyangtongxue/ACS-Edge",
        ha="center", va="center", fontsize=18, color=BL, fontweight="bold")

ax.text(0.5, 0.185,
        "Open to roles in software engineering, computer vision, or HPC.",
        ha="center", va="center", fontsize=13, color=FG)
ax.text(0.5, 0.120,
        "#ComputerVision  #Python  #HPC  #SwarmIntelligence  #OpenSource",
        ha="center", va="center", fontsize=11, color=MT)

fig.savefig(OUT / "slide_06.png", dpi=DPI)
plt.close()
print("6/6")


# ═══════════════════════════════════════════════════════════════════════════════
# PDF
# ═══════════════════════════════════════════════════════════════════════════════
with PdfPages(OUT / "carousel.pdf") as pdf:
    for n in range(1, TOTAL + 1):
        img = Image.open(OUT / f"slide_0{n}.png")
        fig2 = plt.figure(figsize=(10, 10), facecolor=BG)
        ax2  = fig2.add_axes([0, 0, 1, 1])
        ax2.imshow(np.asarray(img))
        ax2.axis("off")
        pdf.savefig(fig2, dpi=DPI)
        plt.close()

print(f"\nDone -> {OUT.resolve()}")
