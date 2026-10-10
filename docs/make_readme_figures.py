"""Figure for the README, made from one run of the quickstart example.

    python docs/make_readme_figures.py                      # about ten minutes on a laptop CPU
    python docs/make_readme_figures.py --cache run.npz      # keep the run, re-plot without re-running

The run is the quickstart setting: Hartmann-6 padded with four inputs that do not enter the response,
400 candidates, 10 initial experiments, seed 0. Everything plotted is read from the BO log, so the
figure shows what the released code does, not a result from the paper.

Writes docs/img/example_run.png (and .svg).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'benchmark'))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from config import OUTPUT_RANGE
from quickstart import build_pool, run_one, converged_at, KAN_HYPERPARAMS

N_DUMMY, SEED = 4, 0
OUT = os.path.join(HERE, 'img')


def compute():
    y_subopt = OUTPUT_RANGE[0] + 0.9 * (OUTPUT_RANGE[1] - OUTPUT_RANGE[0])
    df, real = build_pool('hartmann6', 400, N_DUMMY, seed=SEED)
    log = run_one('KAN', 'DI-UCB-H', KAN_HYPERPARAMS, df, y_subopt, 10, SEED).sort_values('k')
    k_hit = converged_at(log, y_subopt)
    d = len(real) + N_DUMMY
    terms = np.array([np.asarray(v, dtype=float).ravel()[:3] for v in log['I_next']])
    lam = np.array([np.asarray(v, dtype=float).ravel() if v is not None else np.full(d, np.nan)
                    for v in log['age_lambdas']])
    return dict(k=log['k'].to_numpy(), y=log['y_next'].to_numpy(dtype=float), terms=terms, lam=lam,
                y_subopt=y_subopt, k_hit=-1 if k_hit is None else k_hit, n_real=len(real))


def style(ax, letter):
    for s in ax.spines.values():
        s.set_visible(True); s.set_color('black'); s.set_linewidth(0.8)
    ax.minorticks_off()
    ax.tick_params(direction='out', length=3, width=0.8, colors='black')
    ax.text(-0.02, 1.03, letter, transform=ax.transAxes, fontsize=13, fontweight='bold', ha='right', va='bottom')
    assert ax.get_title() == ''


def plot(r):
    # the project modules set large global rcParams at import; start from matplotlib's defaults
    matplotlib.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
                         'axes.linewidth': 0.8, 'savefig.dpi': 200})
    k, y, terms, lam = r['k'], r['y'], r['terms'], r['lam']
    n_real, y_subopt, k_hit = int(r['n_real']), float(r['y_subopt']), int(r['k_hit'])
    best = np.maximum.accumulate(y)
    share = lam / np.nansum(lam, axis=1, keepdims=True)
    d = share.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), gridspec_kw={'width_ratios': [1, 1, 1.2], 'wspace': 0.32})

    # a) best observation so far
    ax = axes[0]
    ax.plot(k, best, color='#1f4e79', lw=1.8, label='best observed')
    ax.axhline(y_subopt, color='#7f7f7f', lw=1.0, ls='--', label='target $y^*_{0.9}$')
    if k_hit >= 0:
        ax.axvline(k_hit, color='#c55a11', lw=1.0, ls=':', label='reached (k = %d)' % k_hit)
    ax.set_xlabel('BO iteration k'); ax.set_ylabel('scaled objective')
    ax.set_xlim(k[0], k[-1])
    # the region right of the plateau and below the curve is empty; the legend sits there
    ax.legend(loc='lower right', bbox_to_anchor=((k_hit - 1.5 - k[0]) / (k[-1] - k[0]) if k_hit >= 0 else 1.0, 0.02),
              frameon=False, handlelength=1.6, borderaxespad=0.2)

    # b) acquisition terms at the chosen point
    ax = axes[1]
    for j, (lab, c) in enumerate([('$I_e$', '#d62728'), ('$I_d$', '#7b3294'), ('$I_u$', '#1f77b4')]):
        ax.plot(k, terms[:, j], color=c, lw=1.3, label=lab)
    ax.set_xlabel('BO iteration k'); ax.set_ylabel('term value')
    ax.set_xlim(k[0], k[-1])
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, borderaxespad=0.2,
              handlelength=1.6, columnspacing=1.2)

    # c) AGE share per input over iterations; padded inputs below the black line
    ax = axes[2]
    cmap = LinearSegmentedColormap.from_list('age', ['#ffffff', '#c7e9c0', '#41ab5d', '#00441b'])
    im = ax.imshow(share.T, aspect='auto', cmap=cmap, vmin=0, vmax=np.nanmax(share), interpolation='nearest',
                   extent=[k[0] - 0.5, k[-1] + 0.5, d - 0.5, -0.5])
    ax.axhline(n_real - 0.5, color='black', lw=1.2)
    ax.set_yticks(range(d)); ax.set_yticklabels(['x%d' % j for j in range(d)])
    for j, t in enumerate(ax.get_yticklabels()):
        if j >= n_real:
            t.set_color('#7f7f7f')
    ax.set_xticks(axes[0].get_xticks()[(axes[0].get_xticks() >= k[0]) & (axes[0].get_xticks() <= k[-1])])
    ax.set_xlabel('BO iteration k'); ax.set_ylabel('input')
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05); cb.set_label('AGE share'); cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(labelsize=9)

    for ax, letter in zip(axes, 'abc'):
        style(ax, letter)
    return fig, share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=None, help='npz file to store the run in, or to read it from')
    args = ap.parse_args()
    if args.cache and os.path.exists(args.cache):
        r = dict(np.load(args.cache))
    else:
        r = compute()
        if args.cache:
            np.savez(args.cache, **r)
    os.makedirs(OUT, exist_ok=True)
    fig, share = plot(r)
    fig.savefig(os.path.join(OUT, 'example_run.png'), dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUT, 'example_run.svg'), bbox_inches='tight', facecolor='white')
    n_real = int(r['n_real'])
    final = share[np.isfinite(share).all(axis=1)][-1]
    print('target reached at k =', int(r['k_hit']) if int(r['k_hit']) >= 0 else 'not reached')
    print('final AGE share, genuine inputs: %.2f, padded inputs: %.2f' % (final[:n_real].sum(), final[n_real:].sum()))


if __name__ == '__main__':
    main()
