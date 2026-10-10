"""Analyze your own dataset with KAN-DI.

Give it a table of experiments (one row per experiment, descriptor columns and one response column)
and it reports

  1. descriptor importance: the KAN is fitted to all rows, pruned and reduced to a closed-form
     expression, and each descriptor's average gradient energy (AGE) under that expression is
     computed; descriptors absent from the expression get zero. The R2 of the expression on the rows
     is reported with it, because a ranking from an expression that does not fit means little;
  2. (with --simulate) a replay of campaigns on your table: the rows are used as the candidate pool
     and the campaign is run from random initial designs, once with KAN-DI and once with a zero-mean
     GP (ZERO-EI), counting the iterations until a proposal reaches the target, the top 10% of the
     response range.

By default the KAN is fitted twice on an 80/20 split, with and without the log transform of its
inputs that the paper's runs use, and the setting with the higher held-out R2 is kept (--log-inputs
on/off fixes it instead).

    python analyze.py data.csv --target yield
    python analyze.py data.csv --target yield --features T,P,ratio --simulate --seeds 5
    python analyze.py dat/small_feature/AutoAM.csv --target Score --simulate

The iteration counts are a replay on the rows you supply, so they describe this pool, not experiments
you have not run. Results go to <out>/ (default: results_<file name>/): report.txt,
descriptor_importance.csv, campaign_replay.csv (with --simulate) and summary.png.

From Python (used by app/streamlit_app.py): `from analyze import analyze` and
`res = analyze('data.csv', 'yield', fits=1)`; res['table'] holds the importance table.
"""
import argparse
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'common'))
sys.path.insert(0, os.path.join(HERE, 'benchmark'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sympy as sp
import torch

from config import FEATURE_RANGE, OUTPUT_RANGE
from models import KANMean, eval_AGE_numeric, eval_func, clip_functions, replace_half_powers
from quickstart import run_one, converged_at, KAN_HYPERPARAMS


def scale(v, target):
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(v, axis=0), np.nanmax(v, axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return target[0] + (target[1] - target[0]) * (v - lo) / span


class TableError(ValueError):
    """A column named on the command line (or passed to analyze) is not in the table."""


def read_table(table):
    """Return a DataFrame from a CSV/Excel path, or a copy of a DataFrame."""
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.read_csv(table) if not str(table).lower().endswith(('.xlsx', '.xls')) else pd.read_excel(table)


def load_table(path, target, features, minimize, name=None, log=print):
    """Scale a table of experiments into the candidate-pool format used by the BO code.

    path is a CSV/Excel path or a DataFrame; features is a comma-separated string, a list, or None
    (every column other than target). Raises TableError when a column is missing.
    """
    raw = read_table(path)
    if target not in raw.columns:
        raise TableError('target column %r not found; columns are: %s' % (target, ', '.join(map(str, raw.columns))))
    if isinstance(features, str):
        feats = features.split(',')
    elif features:
        feats = list(features)
    else:
        feats = [c for c in raw.columns if c != target]
    missing = [f for f in feats if f not in raw.columns]
    if missing:
        raise TableError('feature columns not found: %s' % ', '.join(map(str, missing)))
    sub = raw[feats + [target]].apply(pd.to_numeric, errors='coerce').dropna()
    if len(sub) < len(raw):
        log('note: %d rows with missing or non-numeric values were dropped' % (len(raw) - len(sub)))
    y = sub[target].to_numpy(float)
    X = scale(sub[feats].to_numpy(float), FEATURE_RANGE)
    Y = scale(-y if minimize else y, OUTPUT_RANGE)
    df = pd.DataFrame(X, columns=['x%d' % j for j in range(X.shape[1])])
    df['X'] = [row for row in X]
    df['Y'] = Y
    df['Z'] = [row[:2] for row in X]
    df.attrs['embedder'] = 'none'
    if name is None:
        name = 'table' if isinstance(path, pd.DataFrame) else os.path.splitext(os.path.basename(path))[0]
    df.attrs['data_name'] = name
    df.attrs['features'] = feats
    return df, feats


def _dataset(X, y, Xte=None, yte=None):
    Xte = X if Xte is None else Xte
    yte = y if yte is None else yte
    t = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float32)
    return {'train_input': t(X), 'test_input': t(Xte),
            'train_label': t(y).reshape(-1, 1), 'test_label': t(yte).reshape(-1, 1)}


def fit_expression(X, y, seed, hyperparams):
    """Fit one KAN and return its pruned closed-form expression, or None if it reduced to a constant."""
    m = KANMean()
    try:
        m.fit(_dataset(X, y), x_range=FEATURE_RANGE, seed=seed, verbose=False, **hyperparams)
    except RuntimeError:  # KANMean.fit raises when the pruned expression has no input left
        return None
    return m.formula


def expression_r2(f, X, y):
    """Coefficient of determination of the closed-form expression on (X, y)."""
    if f is None:
        return float('nan')
    zvars = sorted(f.free_symbols, key=lambda s: s.name)
    idx = [int(z.name.split('x')[1]) for z in zvars]
    pred = eval_func(sp.lambdify(zvars, f, [clip_functions, 'numpy']), X[:, idx])
    pred = np.nan_to_num(np.asarray(pred, float), nan=np.nanmean(y))
    return 1.0 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)


def age_share(f, X, d):
    """AGE of every input under expression f, normalized to sum to one; inputs absent from f get zero."""
    lam = np.zeros(d)
    if f is None:
        return lam, []
    zvars = sorted(f.free_symbols, key=lambda s: s.name)
    idx = [int(z.name.split('x')[1]) for z in zvars]
    dfs = [replace_half_powers(sp.diff(f, z) ** 2) for z in zvars]
    _, lambdas = eval_AGE_numeric(dfs, X[:, idx], zvars, FEATURE_RANGE)
    lam[idx] = np.nan_to_num(np.asarray(lambdas, float).ravel())
    return (lam / lam.sum() if lam.sum() > 0 else lam), idx


def choose_input_transform(X, y, hyperparams, holdout=0.2, log=print):
    """Pick the KAN input transform (log, the paper default, or none) by held-out R2 of the expression."""
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    n_te = max(int(round(holdout * len(y))), 5)
    te, tr = perm[:n_te], perm[n_te:]
    scores = {}
    for use_log in (True, False):
        hp = dict(hyperparams, log_transformation=use_log)
        scores[use_log] = expression_r2(fit_expression(X[tr], y[tr], 0, hp), X[te], y[te])
        log('  input transform %-4s held-out R2 = %.2f' % ('log' if use_log else 'none', scores[use_log]))
    best = max(scores, key=lambda k: -np.inf if not np.isfinite(scores[k]) else scores[k])
    return best, scores


def descriptor_importance(df, n_fits, hyperparams, log=print):
    """AGE share of every descriptor from KANs fitted to all rows, one row per initialization."""
    X = np.vstack(df['X'].values); y = df['Y'].to_numpy(float)
    d = X.shape[1]
    shares, kept, r2 = [], np.zeros(d), []
    for seed in range(n_fits):
        f = fit_expression(X, y, seed, hyperparams)
        s, idx = age_share(f, X, d)
        kept[idx] += 1
        shares.append(s)
        r2.append(expression_r2(f, X, y))
        log('  fit %d/%d: R2 = %.2f, inputs in expression: %s'
              % (seed + 1, n_fits, r2[-1], ', '.join(df.attrs['features'][j] for j in idx) or 'none'))
    return np.array(shares), kept / n_fits, np.array(r2)


def plot_summary(table, sim, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcdefaults()  # the project modules set large global rcParams at import
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 0.8})
    n = len(table)
    ncol = 2 if sim is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 + 2.8 * (ncol - 1), 0.32 * n + 1.6), squeeze=False,
                             gridspec_kw={'wspace': 0.35, 'width_ratios': [1.6, 1][:ncol]})
    axes = axes.ravel()
    ax = axes[0]
    t = table.iloc[::-1]
    ax.barh(t['descriptor'], 100 * t['AGE_share_mean'], xerr=100 * t['AGE_share_sd'], height=0.6,
            color='#2e7d32', ecolor='black', error_kw={'lw': 0.8, 'capsize': 2})
    ax.set_xlabel('AGE share (%)')
    ax.set_xlim(0, max(5.0, 100 * float((t['AGE_share_mean'] + t['AGE_share_sd']).max()) * 1.08))
    if sim is not None:
        ax = axes[1]
        vals = []
        for i, name in enumerate(('KAN-DI', 'ZERO-EI')):
            v = sim.loc[sim.method == name, 'iterations_to_target'].dropna().to_numpy(float)
            vals.extend(v)
            off = np.linspace(-0.1, 0.1, len(v)) if len(v) > 1 else np.zeros(len(v))
            ax.scatter(i + off, v, s=30, color=['#1f4e79', '#9e9e9e'][i], edgecolor='black', lw=0.5, zorder=3)
            if len(v):
                ax.hlines(np.median(v), i - 0.25, i + 0.25, color='black', lw=1.5, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['KAN-DI', 'ZERO-EI']); ax.set_xlim(-0.5, 1.5)
        top = max(vals) if vals else 1.0
        ax.set_ylim(0, top * 1.12 + 1)
        ax.set_ylabel('iterations to target')
    for a, letter in zip(axes, 'ab'):
        for s in a.spines.values():
            s.set_visible(True); s.set_color('black'); s.set_linewidth(0.8)
        a.minorticks_off()
        a.tick_params(length=3, width=0.8)
        a.text(0, 1.02, letter, transform=a.transAxes, fontsize=12, fontweight='bold', ha='right', va='bottom')
        assert a.get_title() == ''
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def analyze(table, target, features=None, minimize=False, fits=3, log_inputs='auto', simulate=False,
            seeds=3, n_initial=10, out=None, name=None, log=print):
    """Run the analysis on a table and return the results (the command line calls this).

    table is a CSV/Excel path or a pandas DataFrame; features is a comma-separated string, a list or
    None (every other column). With out=None nothing is written to disk; otherwise report.txt,
    descriptor_importance.csv and (with simulate) campaign_replay.csv are written to out.
    Progress lines go to log (print by default).

    Returns a dict: table (importance per descriptor, sorted), r2 (R2 of each fitted expression on the
    rows), log_inputs (transform used), transform_scores (held-out R2 per transform, auto only),
    sim (replay DataFrame or None), report (text), n_rows, features, out.
    """
    df, feats = load_table(table, target, features, minimize, name=name, log=log)
    if out is not None:
        os.makedirs(out, exist_ok=True)
    label = df.attrs['data_name'] if isinstance(table, pd.DataFrame) else table
    log('%s: %d experiments, %d descriptors' % (label, len(df), len(feats)))
    X = np.vstack(df['X'].values); y = df['Y'].to_numpy(float)

    log('\n[1] KAN input transform')
    scores = None
    if log_inputs == 'auto':
        use_log, scores = choose_input_transform(X, y, KAN_HYPERPARAMS, log=log)
        how = 'selected by held-out R2: log %.2f, none %.2f' % (scores[True], scores[False])
    else:
        use_log = log_inputs == 'on'
        how = 'set by --log-inputs'
    hp = dict(KAN_HYPERPARAMS, log_transformation=use_log)
    log('  using: %s' % ('log' if use_log else 'none'))

    log('\n[2] descriptor importance (KAN fitted to all rows)')
    shares, kept, r2 = descriptor_importance(df, fits, hp, log=log)
    table = pd.DataFrame({'descriptor': feats, 'AGE_share_mean': shares.mean(axis=0), 'AGE_share_sd': shares.std(axis=0),
                          'in_expression': kept}).sort_values('AGE_share_mean', ascending=False)
    if out is not None:
        table.to_csv(os.path.join(out, 'descriptor_importance.csv'), index=False)

    lines = ['KAN input transform: %s (%s)' % ('log' if use_log else 'none', how),
             'Fit of the closed-form expression to all rows: R2 = %s' % ', '.join('%.2f' % v for v in r2)]
    if not np.nanmedian(r2) >= 0.5:
        lines.append('  warning: the expression explains less than half of the variance; treat the ranking as unreliable')
    lines += ['',
              'Descriptor importance (mean AGE share over %d KAN fits; in expr. = fraction of fits whose '
              'pruned expression contains it)' % fits]
    for r in table.itertuples():
        lines.append('  %-24s %6.1f%%  (sd %4.1f)  in expr. %3.0f%%'
                     % (r.descriptor, 100 * r.AGE_share_mean, 100 * r.AGE_share_sd, 100 * r.in_expression))
    cum = np.cumsum(table['AGE_share_mean'].to_numpy())
    n90 = int(min(np.searchsorted(cum, 0.9 - 1e-9) + 1, len(feats)))
    lines.append('  %d of %d descriptors carry 90%% of the gradient energy' % (n90, len(feats)))

    sim = None
    if simulate:
        log('\n[3] campaign replay on this pool (%d seeds)' % seeds)
        y_target = OUTPUT_RANGE[0] + 0.9 * (OUTPUT_RANGE[1] - OUTPUT_RANGE[0])
        n_above = int((df['Y'] > y_target).sum())
        rows = []
        for seed in range(seeds):
            for method, m, acq, h in [('KAN-DI', 'KAN', 'DI-UCB-H', hp), ('ZERO-EI', 'ZERO', 'EI', None)]:
                # a run whose acquisition turns non-finite raises in models.eval_DI; the paper's sweep
                # marked such runs __error__ and reran them, so here the seed is recorded as failed
                try:
                    k, err = converged_at(run_one(m, acq, h, df, y_target, n_initial, seed, stop_at_target=True), y_target), None
                except Exception as e:
                    k, err = None, '%s: %s' % (type(e).__name__, e)
                rows.append({'seed': seed, 'method': method, 'iterations_to_target': k, 'error': err})
                log('  seed %d  %-7s  %s' % (seed, method, ('failed (%s)' % err) if err else
                                             (k if k is not None else 'not reached')))
        sim = pd.DataFrame(rows)
        if out is not None:
            sim.to_csv(os.path.join(out, 'campaign_replay.csv'), index=False)
        lines += ['',
                  'Campaign replay: %d initial experiments drawn from the lower half of the response range; '
                  'target = top 10%% of the range (%d of %d rows qualify)' % (n_initial, n_above, len(df))]
        for method in ('KAN-DI', 'ZERO-EI'):
            sm = sim[sim.method == method]
            v = sm['iterations_to_target'].dropna().to_numpy(float)
            n_fail = int(sm['error'].notna().sum())
            lines.append('  %-7s median %s iterations to the target (%d of %d seeds reached it%s)'
                         % (method, ('%.0f' % np.median(v)) if len(v) else 'n/a', len(v), seeds,
                            ', %d failed' % n_fail if n_fail else ''))
        lines.append('  (a replay on the rows supplied: it describes this pool, not experiments you have not run)')

    report = '\n'.join(lines)
    if out is not None:
        open(os.path.join(out, 'report.txt'), 'w', encoding='utf-8').write(report + '\n')
    return {'table': table, 'r2': r2, 'log_inputs': use_log, 'transform_scores': scores, 'sim': sim,
            'report': report, 'n_rows': len(df), 'features': feats, 'out': out}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n', 2)[2])
    ap.add_argument('table', help='CSV or Excel file, one row per experiment')
    ap.add_argument('--target', required=True, help='response column')
    ap.add_argument('--features', default=None, help='comma-separated descriptor columns (default: all others)')
    ap.add_argument('--minimize', action='store_true', help='lower response is better')
    ap.add_argument('--fits', type=int, default=3, help='KAN initializations averaged for the importance (default 3)')
    ap.add_argument('--log-inputs', choices=['auto', 'on', 'off'], default='auto',
                    help='log-transform inputs inside the KAN (on = paper default); auto picks by held-out R2')
    ap.add_argument('--simulate', action='store_true', help='replay campaigns on the table to estimate iterations')
    ap.add_argument('--seeds', type=int, default=3, help='random initial designs for --simulate (default 3)')
    ap.add_argument('--n-initial', type=int, default=10, help='initial experiments for --simulate (default 10)')
    ap.add_argument('--out', default=None, help='output folder')
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.table))[0]
    out = args.out or os.path.join(os.getcwd(), 'results_' + name)
    try:
        res = analyze(args.table, args.target, features=args.features, minimize=args.minimize, fits=args.fits,
                      log_inputs=args.log_inputs, simulate=args.simulate, seeds=args.seeds,
                      n_initial=args.n_initial, out=out, name=name)
    except TableError as e:
        raise SystemExit(str(e))
    print('\n' + res['report'])

    try:
        plot_summary(res['table'], res['sim'], os.path.join(out, 'summary.png'))
    except Exception as e:  # the figure is optional; the report and tables are already written
        print('figure skipped: %s' % e)
    print('\nwritten to %s' % out)


if __name__ == '__main__':
    main()
