# KAN-DI: Discriminative Bayesian Optimization with a Kolmogorov–Arnold Prior

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22652793.svg)](https://doi.org/10.5281/zenodo.22652793)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Reference implementation for the paper
**"Experimental Design for Descriptor Discovery: a Kolmogorov–Arnold Prior with Discriminative Bayesian Optimization."**

KAN-DI installs a Kolmogorov–Arnold Network (KAN) as the **prior mean of a
Gaussian process** and couples it with a **composite acquisition function**
so that a single Bayesian-optimization loop *optimizes the response and
identifies the governing descriptors simultaneously*.

- **KAN prior mean** — the trained KAN is pruned to its active inputs and reduced
  to a closed-form expression `f_KAN`; that expression is the GP's prior mean,
  `m(x) = f_KAN(x)`, so the GP models only the residual `y − f_KAN(x)`
  (`common/models.py:KANMean`).
- **AGE (Average Gradient Energy)** — the importance of descriptor `i` is the
  squared partial derivative `(∂f_KAN/∂x_i)²` averaged over the candidate pool
  (Monte-Carlo estimate; `common/models.py:eval_AGE_numeric`). Sorting descriptors
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

This release contains the **method and a runnable example only**. Figure-reproduction
scripts, the experimental-campaign analysis, the GNN screening pipeline, and the
distributed (Ray) execution backend are omitted.

---

## Repository layout

```
common/                  Core framework
  models.py                DiscreteBO (BO loop), KANMean (KAN prior), ExactGPModel,
                           eval_DI (composite I_e/I_d/I_u acquisition), AGE evaluation
  config.py                DATASET / COMBINATIONS / feature & output ranges
  datamanager.py           dataset loading + synthetic-benchmark generation (AGE, get_X/Y)
  utils.py, plotter.py     helpers and the built-in BO monitoring plots
  chemistry.py             SMILES/descriptor helpers
  putils.py                minimal serial stand-in for the cluster backend (no plumbing)
  plot_style.py            publication figure style
  kan/                     self-contained KAN implementation (custom.py, LBFGS.py, ...)
benchmark/
  quickstart.py            self-contained example; needs no downloaded data
  main.py                  the paper's benchmark sweep; needs the datasets under dat/
requirements.txt
LICENSE
```

## Installation

```bash
conda create -n kandi python=3.12
conda activate kandi
pip install -r requirements.txt
```
Core packages: `torch`, `gpytorch`, `sympy`, `scikit-learn`, `numpy`, `pandas`,
`matplotlib`. `sympy` is not incidental: the pruned KAN is differentiated
symbolically to obtain AGE in closed form. The remaining entries in
`requirements.txt` are marked optional and are only needed by modules the
quickstart does not exercise.

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

- **Benchmark datasets** (AgNP, AutoAM, P3HT, Perovskite, Crossed barrel,
  dilute-solute diffusion, metallic-glass, MOF `T_d`, polymer `C_p`) are the
  public sets compiled by Liang et al., *npj Comput. Mater.* **7**, 188 (2021).
- **Synthetic benchmarks** (Rastrigin, Hartmann-6, Griewank, Rosenbrock, Ackley)
  are generated analytically in `common/datamanager.py:gen_benchmark_functions`.
- **Amine-screening campaign data** are available from the authors on reasonable request.

Place datasets under `dat/<group>/` as expected by `datamanager.load_dataset`.

## Results reported in the paper

Median BO iterations to reach `y > y_0.9*`, 10 random initial designs per method.
Reproduce with `python benchmark/main.py` after placing the datasets under `dat/`
(see **Data availability**); the numbers below are read from the resulting logs.

| Dataset | d | KAN-DI | best conventional baseline | KAN prior, conventional acquisition |
|---|---|---|---|---|
| MOF `T_d` | 50 | **16** | 42 (ZERO-EI) | 128.5 (KAN-EI) |

The third column is the ablation: the same KAN prior with a standard EI acquisition
is no faster than the zero-mean baselines, so the gain comes from coupling the
importance signal to the acquisition, not from the prior alone. On the AutoAM
benchmark KAN-DI reaches the target in about a fifth of the baseline's iterations.

Full results for all nine datasets, the five synthetic contrast functions and the
complete acquisition ablation are in the paper and its Supplementary Information.

## Method notes
- Features and targets are min–max scaled to `[0.1, 0.9]`.
- GP kernel: Matérn-5/2 + ARD. KAN trained with LBFGS (100 train + 50 retrain steps),
  built-in pruning yields the discrete active-descriptor subspace.
- Convergence threshold `y_0.9* = 0.82`; initial design of 10 experiments.

## License
MIT, see `LICENSE`. The KAN implementation under `common/kan/` derives from
[pykan](https://github.com/KindXiaoming/pykan) (Liu et al.), also MIT licensed.

## Citation
If you use this code, please cite the paper above and the archived version of this
repository (the DOI badge at the top). BibTeX will be added upon publication.
