# KAN-DI: Discriminative Bayesian Optimization with a Kolmogorov–Arnold Prior

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22652792.svg)](https://doi.org/10.5281/zenodo.22652792)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Reference implementation for the paper
**"Experimental Design for Descriptor Discovery: Kolmogorov–Arnold Prior with Discriminative Bayesian Optimization."**

The paper calls the full framework **KAN-DoE**: a Kolmogorov–Arnold prior inside a
Bayesian-optimization loop for experimental design. **KAN-DI** is its discriminative
composite acquisition, and the variant reported in the paper is `DI-UCB-H`.
This repository implements both.

KAN-DI installs a Kolmogorov–Arnold Network (KAN) as the **prior mean of a
Gaussian process** and couples it with a **composite acquisition function**
so that a single Bayesian-optimization loop *optimizes the response and
identifies the governing descriptors simultaneously*.

- **KAN prior mean** — the trained KAN is pruned to its active inputs and reduced
  to a closed-form expression `f_KAN`; that expression is the GP's prior mean,
  `m(x) = f_KAN(x)`, so the GP models only the residual `y − f_KAN(x)`
  (`common/models.py:KANMean`).
- **AGE (Average Gradient Energy)** — the importance of descriptor `i` is the
  squared partial derivative `(∂f_KAN/∂x_i)²` of the closed-form expression,
  averaged over the scaled input domain by stratified sampling
  (`common/models.py:eval_AGE_numeric`). Sorting descriptors
  by AGE defines the *leading* descriptor and the *runner-up* used by `I_d` and
  `I_u` below.
- **Composite acquisition** `α(x) = I_e(x) + I_d(x) + I_u(x)`
  (`common/models.py:eval_DI`). `I_d` and `I_u` are bounded to `[0, 1]` by
  construction; `I_e` is scaled by the observed objective range and can exceed
  unity.
  - `I_e`, **exploitation** — the GP upper confidence bound `μ(x) + σ(x)`
    (the paper's `DI-UCB-H` variant; `DI-EI`, `DI-UCB-L`, and `DI-TS` are also
    implemented), min–max normalized by the objective range. It is gated to zero
    once the best observation exceeds the target threshold `y_0.9*`, after which
    the loop keeps sampling only to sharpen the descriptor identification.
  - `I_d`, **discrimination** — the share of local gradient energy that the
    leading descriptor carries against the runner-up at the candidate,
    `δ₁(x) / (δ₁(x) + δ₂(x))` with `δᵢ(x) = (∂f_KAN/∂x_i)²` evaluated at `x`.
    It drives sampling toward candidates that separate the top two descriptors,
    and it is gated to zero once no unmeasured candidate carries more
    leading-descriptor gradient energy than the measured set already contains.
  - `I_u`, **uniformity** — the candidate's distance to the nearest measured
    point *along the leading descriptor's axis*, normalized by the input range.
    It spreads measurements along the one axis the model currently ranks as
    governing, so the identification is not decided by a clustered sample.

This release contains the **method and a runnable example only**. It does not contain
figure scripts, plotted source data or the distributed (Ray) execution backend.

---

## Repository layout

```
common/                  Core framework
  models.py                DiscreteBO (BO loop), KANMean (KAN prior), ExactGPModel,
                           eval_DI (composite I_e/I_d/I_u acquisition), AGE evaluation
  config.py                DATASET / COMBINATIONS / feature & output ranges
  datamanager.py           dataset loading + synthetic-benchmark generation (AGE, get_X/Y)
  utils.py, plotter.py     helpers and the built-in BO monitoring plots
  putils.py                minimal serial stand-in for the cluster backend (no plumbing)
  kan/                     self-contained KAN implementation (custom.py, LBFGS.py, ...)
benchmark/
  quickstart.py            self-contained example; needs no downloaded data
  main.py                  the paper's benchmark sweep; needs the datasets under dat/
requirements.txt           core dependencies (everything the benchmarks import)
requirements-optional.txt  extra utilities, grouped by feature
CITATION.cff
LICENSE
```

## Installation

```bash
conda create -n kandi python=3.12
conda activate kandi
pip install -r requirements.txt            # core: runs quickstart.py and main.py
pip install -r requirements-optional.txt   # optional extras, see below
```
`requirements.txt` is the core set: `torch`, `gpytorch`, `linear-operator`, `sympy`,
`scikit-learn`, `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`, `PyYAML`, `dill`.
`sympy` is not incidental: the pruned KAN expression is differentiated symbolically,
and AGE is the sampled mean of the squared derivative.

`requirements-optional.txt` is not needed by either benchmark script. Each package
is imported only inside the feature that uses it, so you can install any subset:

| Group | Packages | Enables |
|---|---|---|
| Plotting extras | `imageio`, `colorcet` | GIF export of the BO monitoring frames (`plotter.render_animation`); the glasbey discrete palette (without `colorcet` the palette falls back to `tab10`) |
| Dataset embedding | `umap-learn`, `pacmap` | `datamanager.preprocessing`, which builds the 2-D embedding of a raw CSV dataset |
| Parallel backend | `ray` | detected by `putils` if installed; execution in this release stays serial |

For a GPU build of PyTorch, install it before the requirements file:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Running the example

```bash
cd benchmark
python quickstart.py
```
This runs without any downloaded data. It builds a candidate pool from one of the
synthetic functions generated analytically in `common/datamanager.py`, pads it with
inputs that do not enter the response, runs the KAN-DI loop, and prints both the
iteration at which the target was reached and the descriptor ranking the KAN
reports. A few minutes per seed on a laptop CPU; the cost is the LBFGS fit of the
KAN at every iteration.

**Scope of the example.** The synthetic functions are the paper's
*contrast case*: their gradient energy is spread across all input dimensions
(Gini < 0.24). In that regime the discriminative term has no descriptor hierarchy
to resolve, so the quickstart neither converges faster than a zero-mean GP
(`--compare` will often show the baseline reaching the target first) nor recovers
the padded inputs cleanly. It is a check that the implementation runs and produces
the expected quantities, not a reproduction of the paper's results. The sample
efficiency and descriptor-recovery claims are made on the nine curated datasets,
which have concentrated importance and are obtained as described under
**Data availability** below.

To reproduce the paper's sweep once the datasets are in place:

```bash
cd benchmark
python main.py
```
Convergence is the first BO iteration whose proposal exceeds `y > y_0.9*`, with
`y_0.9* = 0.82` in the `[0.1, 0.9]`-scaled objective.

Execution is **serial** (`common/putils.py`); the Ray-based cluster backend used
for the large-scale runs in the paper is not included.

## Data availability

Datasets are **not** distributed here.

- **Low-dimensional benchmarks** (d = 3–5; AgNP, AutoAM, P3HT, Perovskite,
  Crossed barrel) are the public sets compiled by Liang et al.,
  *npj Comput. Mater.* **7**, 188 (2021).
- **High-dimensional benchmarks** (d = 22–50) come from four separate sources:
  dilute-solute diffusion from Wu, Mayeshiba & Morgan, *Sci. Data* **3**, 160054 (2016);
  metallic-glass forming from Ward et al., *npj Comput. Mater.* **2**, 16028 (2016);
  MOF `T_d` from Nandy, Duan & Kulik, *J. Am. Chem. Soc.* **143**, 17535 (2021);
  polymer `C_p` from Kim et al., *J. Phys. Chem. C* **122**, 17575 (2018).
  The paper's Supplementary Tables S4 and S5 list the source of every dataset.
- **Synthetic benchmarks** (Rastrigin, Hartmann-6, Griewank, Rosenbrock, Ackley)
  are generated analytically in `common/datamanager.py:gen_benchmark_functions`.

Place datasets under `dat/<group>/` as expected by `datamanager.load_dataset`.

## Results reported in the paper

Median BO iterations to reach `y > y_0.9*` (first 0-based iteration whose proposal
exceeds the target), 10 random initial designs per method, as reported in the paper:

| Dataset | d | KAN-DI (`DI-UCB-H`) | best zero-mean baseline | KAN prior, EI acquisition |
|---|---|---|---|---|
| MOF `T_d` | 50 | 16 | 42 (ZERO-EI) | 128.5 (KAN-EI) |

The KAN-EI column shows that the KAN prior alone does not explain the difference.
The acquisition functions differ in several components, so this comparison does not
isolate the discriminative term. Full results are in the paper and its
Supplementary Information.

**Run-to-run variability.** These numbers are the paper's runs, not a guaranteed
output of `main.py`. The KAN is reduced to a closed-form expression by pruning and
symbolic fitting, and that step can select different descriptors when the fitted
spline scores are close to the pruning threshold. Floating-point differences that
change nothing else, such as `torch.use_deterministic_algorithms` or the number of
CPU threads, are then enough to move a run onto a different path. Individual runs
and per-dataset medians can therefore differ from the table across hardware and
software environments. Compare distributions over seeds rather than single runs.

## Known issues

- **`I_u` axis (`DiscreteBO.eval_DI`).** `I_u` is meant to measure distance along the
  leading descriptor's input column, `zindices[j1]`. The code behind the paper's
  results reads column `j1`, the position of that descriptor among the formula's
  symbols, which is a different column whenever the active descriptors are not
  `x0, x1, …` in order. Set `KANDOE_IU_FIX=1` to use the intended column. The default
  is left unchanged so that the released code matches the computation reported in
  the paper.
- **Constant derivatives (fixed in 1.0.2).** A pruned expression whose partial
  derivative simplifies to a constant made the AGE evaluation raise. `eval_func`
  now broadcasts the constant; runs that did not raise are unchanged.

## Method notes
- Features and targets are min–max scaled to `[0.1, 0.9]`.
- GP kernel: Matérn-5/2 + ARD. KAN trained with LBFGS (100 train + 50 retrain steps),
  built-in pruning yields the discrete active-descriptor subspace.
- Convergence threshold `y_0.9* = 0.82`; initial design of 10 experiments.

## License
MIT, see `LICENSE`. The KAN implementation under `common/kan/` derives from
[pykan](https://github.com/KindXiaoming/pykan) (Liu et al.), also MIT licensed.

## Citation
If you use this code, please cite the paper above (manuscript submitted) and the
archived software:

> Choi, J., Kim, K. & Lee, U. KAN-DI: Discriminative Bayesian Optimization with a
> Kolmogorov–Arnold Prior. Zenodo. https://doi.org/10.5281/zenodo.22652792

This concept DOI always resolves to the latest archived release. Machine-readable
metadata are in `CITATION.cff`. BibTeX for the paper will be added upon publication.
