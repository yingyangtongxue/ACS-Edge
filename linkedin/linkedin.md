# LinkedIn Post — ACS Edge Detection

## English (main post)

---

Last semester I had to implement a swarm intelligence algorithm for a grad course.
I got it working, turned it in, and then couldn't stop thinking about how slow it was.

So I kept going.

The algorithm is called Ant Colony System. The idea is surprisingly visual.

You simulate a colony of virtual ants crawling over an image. Each ant starts at a
random pixel and moves through its 8 neighbors based on two signals: how strong
the local intensity variation is (the heuristic, a proxy for "this looks like an
edge"), and how much pheromone previous ants left on that pixel.

The pheromone works in two layers. While an ant is walking, it slightly evaporates
the trail on each pixel it steps on, then deposits a small amount based on the
local gradient. This is the local update, and it's what stops every ant from
converging to the exact same path. After a full construction phase, the colony
does a global update: pixels that many ants visited, especially pixels with high
gradient, get a stronger pheromone boost. Pixels nobody visited fade out.

There's also a parameter called q0 that controls exploitation vs exploration. When
a random draw is below q0, the ant picks the neighbor with the highest combined
score deterministically. Otherwise it samples probabilistically from the whole
neighborhood. Low q0 means the colony explores broadly and produces noisier, more
diffuse edges. High q0 means it exploits strong signals and produces sharper,
more concentrated paths.

After enough iterations you threshold the pheromone matrix with Otsu's method
and you have a binary edge map.

It's elegant. It's also, in naive Python, extremely slow.

The original code I started from looped over every pixel, for every ant, at every
step. For a 512x512 image with around 2000 ants and 10 iterations, I measured
this empirically: roughly 8.7 hours per run. I didn't just estimate that number.
I ran the original code on a small image, timed the inner loop directly, and
extrapolated using the known complexity scaling. Same way you'd profile something
before touching it.

The fix was replacing all of that with vectorised NumPy. Every ant moves in
parallel. The pheromone update becomes a single scatter operation over a matrix.

Same algorithm. Same output. 628 ms on CPU. About 50,000x faster.

On a smaller 50x43 pixel image I measured the original at 18.9 seconds. The
new version runs it in 59 ms.

A few other things I ended up doing along the way: added GPU support via CuPy,
found out the CPU is actually faster for these image sizes because kernel launch
overhead dominates on a laptop GPU, wrote a grid search to tune the ant count
and other parameters across 48 combinations, and evaluated quality against Canny
edges as a reference. The default parameters now score 37.4% F1 on the standard
lena benchmark, up from 22% with the original settings.

Also 27 tests. CI. Reproducible seeds. The usual.

The course was Digital Image Processing, first semester of my master's at PPGCC /
UNESP. The refactoring happened on my own time, mostly because I wanted to
understand where the time was actually going.

I'm looking for opportunities in software engineering, computer vision, or
anything that involves making things run significantly faster.

Code and benchmarks: https://github.com/yingyangtongxue/ACS-Edge

#ComputerVision #Python #GPU #HPC #SwarmIntelligence #OpenSource #SoftwareEngineering

---

## Portuguese (first comment)

PT 🇧🇷 — No primeiro semestre do mestrado (PPGCC / UNESP) implementei o algoritmo
ACS para detecção de bordas em imagens como trabalho da disciplina de
Processamento de Imagens Digitais. Depois continuei otimizando por conta própria.

O resultado foi ~50.000x de speedup sobre o código original em Python puro
(de ~8,7 horas para 628 ms na mesma máquina), aceleração GPU via CuPy, busca
em grade de hiperparâmetros e melhora de +70% no F1 em relação aos parâmetros
originais.

Código aberto no GitHub, link no post.

---

## Format tips

- Comprimento: ~2.800 caracteres, vai pedir "see more" mas para post técnico isso funciona
- Imagem: screenshot da tabela comparativa do README (Input / Legacy / New default / Tuned)
- Primeiro comentário: cole o resumo PT + link do GitHub logo após publicar
