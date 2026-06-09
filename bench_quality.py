import numpy as np
from PIL import Image
from skimage import filters, feature
from acs_edge import ACSConfig, ACSEdgeDetector


def run_acs(img, ants, q0):
    cfg = ACSConfig(num_ants=ants, num_iterations=10, steps_per_iteration=40, seed=42)
    det = ACSEdgeDetector(cfg, use_gpu=False)
    ph = det.detect(img, q0=q0)
    t = filters.threshold_otsu(ph)
    return (ph >= t).astype(bool)


def f1_score(pred, ref):
    tp = (pred & ref).sum()
    fp = (pred & ~ref).sum()
    fn = (~pred & ref).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f


def print_table(rows):
    hdr = f"  {'Config':<26}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Density':>8}"
    print(hdr)
    print("  " + "-" * 66)
    for label, prec, rec, f, density in rows:
        print(f"  {label:<26}  {prec:>9.1%}  {rec:>7.1%}  {f:>7.1%}  {density:>8.2%}")


# --- lena 512x512 ---
lena = np.asarray(Image.open("samples/lena.png").convert("L"), dtype=np.float32)
canny_lena = feature.canny(lena / 255.0, sigma=2)

configs_lena = [
    ("Legacy  K=512  q0=0.5",  512,  0.5),
    ("Default K=2048 q0=0.5", 2048,  0.5),
    ("Tuned   K=1024 q0=0.7", 1024,  0.7),
]
print("lena 512x512 (Canny sigma=2 as reference)")
rows_lena = []
for label, k, q0 in configs_lena:
    pred = run_acs(lena, k, q0)
    p, r, f = f1_score(pred, canny_lena)
    rows_lena.append((label, p, r, f, pred.mean()))
print_table(rows_lena)

# --- pikachu 50x43 ---
pika = np.asarray(Image.open("samples/pikachu.png").convert("L"), dtype=np.float32)
canny_pika = feature.canny(pika / 255.0, sigma=1)

configs_pika = [
    ("Legacy  K=100  q0=0.5",  100, 0.5),
    ("Default K=64   q0=0.5",   64, 0.5),
    ("Tuned   K=64   q0=0.7",   64, 0.7),
]
print()
print("pikachu 50x43 (Canny sigma=1 as reference)")
rows_pika = []
for label, k, q0 in configs_pika:
    pred = run_acs(pika, k, q0)
    p, r, f = f1_score(pred, canny_pika)
    rows_pika.append((label, p, r, f, pred.mean()))
print_table(rows_pika)
