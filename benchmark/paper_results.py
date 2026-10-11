"""Benchmark results of the paper, recomputed from the released run records.

    python paper_results.py                                   # median iterations, all pools and methods
    python paper_results.py --data AutoAM,MOF_Td --methods KAN-DI-UCB-H,ZERO-EI,KAN-EI
    python paper_results.py --per-seed --data AutoAM          # one row per seed

paper_runs/bo_runs_<group>.csv holds every proposal of the benchmark runs reported in the paper: one row
per BO iteration k of one (pool, prior, acquisition, seed) run, with the chosen candidate (pool_row, a row
of dat/<group>/<pool>.csv), its objective y_next, the acquisition value a_next and, for the DI
acquisitions, the three terms I_e, I_d, I_u and the two leading descriptors (x<column index>). Runs that
ended in an error were rerun or left out in the paper, so a few methods have fewer than 10 seeds;
`seeds` in the output counts them.

Iterations to the target are counted as in the paper: the first iteration k (0-based, after the 10 initial
experiments) whose proposal exceeds y_0.9* = 0.82 in the [0.1, 0.9]-scaled objective. A run that never
exceeds it counts as not reached and is left out of the median; `reached` counts the others.
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'paper_runs')
Y_TARGET = 0.1 + 0.9 * (0.9 - 0.1)


def load_runs():
    df = pd.concat([pd.read_csv(os.path.join(RUNS, 'bo_runs_%s.csv' % g)) for g in ('small_feature', 'large_feature')],
                   ignore_index=True)
    df['method'] = df['prior'] + '-' + df['acquisition']
    return df


def iterations_to_target(df):
    rows = []
    for (data, method, seed), run in df.groupby(['data_name', 'method', 'seed']):
        hit = run.loc[run['y_next'] > Y_TARGET, 'k']
        rows.append({'data_name': data, 'method': method, 'seed': seed,
                     'iterations': int(hit.min()) if len(hit) else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=None, help='comma-separated pools (default: all nine)')
    ap.add_argument('--methods', default=None, help='comma-separated methods, e.g. KAN-DI-UCB-H,ZERO-EI')
    ap.add_argument('--per-seed', action='store_true', help='print the iterations of every seed')
    args = ap.parse_args()

    runs = load_runs()
    if args.data:
        runs = runs[runs['data_name'].isin(args.data.split(','))]
    if args.methods:
        runs = runs[runs['method'].isin(args.methods.split(','))]
    it = iterations_to_target(runs)
    pd.set_option('display.width', 200)
    if args.per_seed:
        print(it.pivot_table(index=['data_name', 'seed'], columns='method', values='iterations').to_string())
        return
    table = it.groupby(['data_name', 'method'])['iterations'].agg(median='median', reached='count')
    table['seeds'] = it.groupby(['data_name', 'method']).size()
    print('Median iterations to y > %.2f (0-based; runs that never reached it are left out)\n' % Y_TARGET)
    print(table['median'].unstack('method').to_string(na_rep='-'))
    print('\nRuns that reached the target / runs recorded\n')
    print((table['reached'].astype(str) + '/' + table['seeds'].astype(str)).unstack('method').to_string(na_rep='-'))


if __name__ == '__main__':
    main()
