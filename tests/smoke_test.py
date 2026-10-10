"""Fast smoke test, run by CI (.github/workflows/ci.yml) and runnable locally:

    python tests/smoke_test.py

1. the core modules import;
2. models.eval_func returns one value per row for a constant and a non-constant expression;
3. the nine benchmark pools under dat/ load through datamanager.load_dataset with the shapes used in
   the paper;
4. benchmark/paper_results.py recomputes the paper's medians from paper_runs/ (AutoAM 5 vs 32, MOF 16 vs 42);
5. analyze.py runs end to end on the AutoAM pool (one KAN fit, no replay) and ranks the two offset
   corrections first, the leading pair the paper reports for AutoAM.

Threads are limited to 4 so the test does not saturate a shared machine.
"""
import os
import subprocess
import sys

for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(var, '4')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'common'))
sys.path.insert(0, os.path.join(ROOT, 'benchmark'))

import numpy as np
import pandas as pd
import sympy as sp
import torch

torch.set_num_threads(4)


def test_imports():
    import config, datamanager, models  # noqa: F401
    from models import DiscreteBO, KANMean, eval_AGE_numeric  # noqa: F401
    assert hasattr(DiscreteBO, 'eval_DI')
    import quickstart  # noqa: F401
    sys.path.insert(0, ROOT)
    import analyze  # noqa: F401
    assert callable(analyze.analyze)
    print('ok  imports')


def test_eval_func():
    from models import eval_func
    x0, x1 = sp.symbols('x0 x1')
    X = np.random.default_rng(0).uniform(0.1, 0.9, size=(7, 2))

    const = eval_func(sp.lambdify([x0, x1], sp.Integer(3)), X)  # a derivative that simplified to a constant
    assert const.shape == (7,), const.shape
    assert np.allclose(const, 3.0)

    prod = eval_func(sp.lambdify([x0, x1], x0 * x1 ** 2), X)
    assert prod.shape == (7,), prod.shape
    assert np.allclose(prod, X[:, 0] * X[:, 1] ** 2)
    print('ok  eval_func (constant and non-constant)')


POOLS = {'small_feature': {'AgNP': (164, 5), 'AutoAM': (100, 4), 'P3HT': (178, 5), 'Perovskite': (94, 3),
                           'Crossed barrel': (600, 4)},
         'large_feature': {'dilute_solute_diffusion': (408, 27), 'metallic_glass_forming': (585, 22),
                           'MOF_Td': (3131, 50), 'polymer_Cp': (67, 50)}}


def test_benchmark_pools():
    from datamanager import load_dataset, get_X, get_Y
    for group, pools in POOLS.items():
        ds = load_dataset(group)
        assert set(ds) == set(pools), (group, sorted(ds))
        for name, (n, d) in pools.items():
            X, y = get_X(ds[name], tensor=False), get_Y(ds[name], tensor=False)
            assert X.shape == (n, d), (name, X.shape)
            assert np.isfinite(X).all() and np.isfinite(y).all(), name
            assert X.min() >= 0.1 - 1e-9 and X.max() <= 0.9 + 1e-9, name
    print("ok  nine benchmark pools load with the paper's shapes")


def test_paper_results():
    import paper_results
    it = paper_results.iterations_to_target(paper_results.load_runs())
    med = it.groupby(['data_name', 'method'])['iterations'].median()
    for (data, method), value in {('AutoAM', 'KAN-DI-UCB-H'): 5, ('AutoAM', 'ZERO-EI'): 32,
                                  ('MOF_Td', 'KAN-DI-UCB-H'): 16, ('MOF_Td', 'ZERO-EI'): 42,
                                  ('MOF_Td', 'KAN-EI'): 128.5}.items():
        assert med[(data, method)] == value, (data, method, med[(data, method)])
    print('ok  paper_results.py reproduces the reported medians')


def test_analyze_example(out='ci_out'):
    out = os.path.join(ROOT, out)
    cmd = [sys.executable, os.path.join(ROOT, 'analyze.py'), os.path.join(ROOT, 'dat', 'small_feature', 'AutoAM.csv'),
           '--target', 'Score', '--fits', '1', '--out', out]
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    for f in ('report.txt', 'descriptor_importance.csv', 'summary.png'):
        assert os.path.isfile(os.path.join(out, f)), 'missing output ' + f
    table = pd.read_csv(os.path.join(out, 'descriptor_importance.csv'))
    ranking = table['descriptor'].tolist()
    print('ranking:', ', '.join('%s %.1f%%' % (d, 100 * s) for d, s in zip(ranking, table['AGE_share_mean'])))
    assert set(ranking[:2]) == {'X Offset Correction', 'Y Offset Correction'}, ranking
    print('ok  analyze.py on AutoAM ranks the two offset corrections first')


if __name__ == '__main__':
    test_imports()
    test_eval_func()
    test_benchmark_pools()
    test_paper_results()
    test_analyze_example()
    print('smoke test passed')
