"""Runnable example that needs no downloaded data.

`main.py` reproduces the paper's benchmark sweep, but it reads the nine curated datasets from
`dat/`, which this release does not redistribute. This script builds its candidate pool instead from
the synthetic test functions defined analytically in `common/datamanager.py`, so it runs immediately
after `pip install -r requirements.txt`.

What it demonstrates is descriptor identification, which is the part of the loop a zero-mean GP has
no equivalent for. The pool is a synthetic function padded with additional inputs that do not enter
the response at all, so the ground truth is known: the padded inputs should end up outside the active
set the KAN reports, and the average gradient energy should rank them last. The script prints the
recovered active set against that ground truth.

    python quickstart.py                        # Hartmann-6 + 4 irrelevant inputs, 1 seed
    python quickstart.py --dummy 0 --compare    # no padding, and run the zero-mean baseline too

A note on what this example does not show. Sample efficiency is reported in the paper on the nine
curated datasets, where importance is concentrated. Synthetic functions are the paper's contrast
case, with gradient energy spread across dimensions, and there the discriminative term has no
hierarchy to resolve and convergence is not expected to improve. Running `--compare` here will often
show the baseline converging in a similar number of iterations or fewer. That is the reported
behavior for this class of function, not a failure of the example.

Convergence is defined as in the paper: the first BO iteration whose proposal exceeds y > y_0.9*,
which is 0.82 on the [0.1, 0.9]-scaled objective. Runtime is a few minutes per seed on a laptop CPU,
dominated by the LBFGS fit of the KAN at every iteration.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sympy as sp

from config import FEATURE_RANGE, OUTPUT_RANGE
from datamanager import gen_benchmark_functions, get_random_indices
from models import DiscreteBO

# The tuned values used for the paper runs; `main.py` reads the same numbers from
# dat/<group>/best_param_KAN.pkl, which this self-contained example does not depend on.
KAN_HYPERPARAMS = {'lamb': 0.01, 'lamb_coef': 0.1, 'lamb_coefdiff': 0.1,
                   'lamb_entropy': 0.01, 'pruning_th': 0.1}


def build_pool(function, n_candidates, n_dummy=0, seed=0):
    """Sample a discrete candidate pool in the schema `DiscreteBO` reads.

    The BO loop is discrete: it selects the next experiment from a fixed pool, the way a real
    campaign selects the next compound from a library. Inputs and the objective are min-max scaled
    into [0.1, 0.9] and the objective is negated so every benchmark is a maximization, both matching
    `datamanager.load_dataset`. Padded inputs are drawn independently of the response, so any
    gradient energy assigned to them is spurious by construction.

    Returns the DataFrame and the indices of the inputs that genuinely drive the response.
    """
    benchmarks = gen_benchmark_functions()
    if function not in benchmarks:
        raise SystemExit('unknown function %r; choose from %s'
                         % (function, ', '.join(sorted(benchmarks))))
    spec = benchmarks[function]
    names = sorted(spec['domain'], key=lambda s: int(s[1:]))
    symbols = [sp.Symbol(n, real=True) for n in names]
    f = sp.lambdify(symbols, spec['expr'], 'numpy')

    rng = np.random.default_rng(seed)
    lo = np.array([spec['domain'][n][0] for n in names])
    hi = np.array([spec['domain'][n][1] for n in names])
    raw = rng.uniform(lo, hi, size=(n_candidates, len(names)))
    y = -np.asarray(f(*[raw[:, j] for j in range(len(names))]), dtype=float)

    if n_dummy:
        raw = np.hstack([raw, rng.uniform(0.0, 1.0, size=(n_candidates, n_dummy))])

    def scale(v, target):
        v = np.asarray(v, dtype=float)
        span = v.max(axis=0) - v.min(axis=0)
        span = np.where(span == 0, 1.0, span)
        return target[0] + (target[1] - target[0]) * (v - v.min(axis=0)) / span

    X = scale(raw, FEATURE_RANGE)
    Y = scale(y, OUTPUT_RANGE)

    df = pd.DataFrame(X, columns=['x%d' % j for j in range(X.shape[1])])
    df['X'] = [row for row in X]
    df['Y'] = Y
    # the 2-D embedding is only read by the optional monitoring panel; the first two inputs stand in
    # for it so the example carries no extra dependency
    df['Z'] = [row[:2] for row in X]
    df.attrs['embedder'] = 'none'
    df.attrs['data_name'] = function
    return df, list(range(len(names)))


def converged_at(log, y_subopt):
    """First iteration whose proposal passes the target, or None if the budget ran out."""
    hit = log[log['y_next'] > y_subopt].sort_values('k')
    return int(hit['k'].iloc[0]) if len(hit) else None


def final_ranking(log):
    """Descriptor indices ordered by the average gradient energy of the last fitted KAN."""
    col = log['age_lambdas'].dropna()
    if not len(col):
        return None
    lam = np.asarray(col.iloc[-1], dtype=float).ravel()
    if not np.isfinite(lam).any():
        return None
    return list(np.argsort(-np.nan_to_num(lam))), lam


def run_one(m, acquisition, hyperparams, df, y_subopt, n_initial, seed):
    indices_initial = get_random_indices(y=df['Y'].values, lower=0.5, n_sample=n_initial, seed=seed)
    agent = DiscreteBO(
        df.attrs['data_name'], df, indices_initial=indices_initial,
        x_range=FEATURE_RANGE, y_range=OUTPUT_RANGE,
        m=m, kernel='GP', acquisition=acquisition, y_subopt=y_subopt,
        hyperparams=hyperparams,
        draw=False, showfig=False, savefig=False, parallel=False,
    )
    return agent.run(verbose=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--function', default='hartmann6', help='synthetic benchmark (default: hartmann6)')
    ap.add_argument('--dummy', type=int, default=4,
                    help='irrelevant inputs appended to the pool (default: 4)')
    ap.add_argument('--candidates', type=int, default=400, help='size of the candidate pool')
    ap.add_argument('--seeds', type=int, default=1, help='number of random initial designs')
    ap.add_argument('--n-initial', type=int, default=10, help='experiments in the initial design')
    ap.add_argument('--compare', action='store_true', help='also run the zero-mean GP baseline')
    args = ap.parse_args()

    y_subopt = OUTPUT_RANGE[0] + 0.9 * (OUTPUT_RANGE[1] - OUTPUT_RANGE[0])
    df0, real = build_pool(args.function, args.candidates, args.dummy, seed=0)
    d = df0['X'].iloc[0].shape[0]

    print('%s: %d candidates, %d inputs (%d genuine, %d irrelevant), target y > %.2f'
          % (args.function, args.candidates, d, len(real), args.dummy, y_subopt))
    print()

    kan_k, base_k = [], []
    for seed in range(args.seeds):
        df, real = build_pool(args.function, args.candidates, args.dummy, seed=seed)
        log = run_one('KAN', 'DI-UCB-H', KAN_HYPERPARAMS, df, y_subopt, args.n_initial, seed)
        k = converged_at(log, y_subopt)
        kan_k.append(k)
        print('KAN-DI-UCB-H  seed %d  converged at k = %s'
              % (seed, k if k is not None else 'not reached'))

        rank = final_ranking(log)
        if rank is not None and args.dummy:
            order, lam = rank
            top = order[:len(real)]
            hits = sum(1 for j in top if j in real)
            print('    leading %d descriptors by gradient energy: %s'
                  % (len(real), ', '.join('x%d' % j for j in top)))
            print('    %d of %d are genuine (padded inputs are x%d and above)'
                  % (hits, len(real), len(real)))
        elif rank is not None:
            order, lam = rank
            print('    descriptors by gradient energy: %s'
                  % ', '.join('x%d' % j for j in order[:min(6, d)]))

        if args.compare:
            logb = run_one('ZERO', 'EI', None, df, y_subopt, args.n_initial, seed)
            kb = converged_at(logb, y_subopt)
            base_k.append(kb)
            print('ZERO-EI       seed %d  converged at k = %s'
                  % (seed, kb if kb is not None else 'not reached'))
        print()

    def med(v):
        done = [x for x in v if x is not None]
        return '%.1f' % np.median(done) if done else 'n/a'

    print('median iterations to target')
    print('  KAN-DI-UCB-H  %s' % med(kan_k))
    if args.compare:
        print('  ZERO-EI       %s' % med(base_k))
        print()
        print('On synthetic functions the gradient energy is spread across dimensions, so the')
        print('discriminative term has no hierarchy to resolve and no speed-up is expected here.')
        print('The sample-efficiency results in the paper are on the curated datasets in dat/.')


if __name__ == '__main__':
    main()
