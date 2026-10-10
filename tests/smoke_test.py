"""Fast smoke test, run by CI (.github/workflows/ci.yml) and runnable locally:

    python tests/smoke_test.py

1. the core modules import;
2. models.eval_func returns one value per row for a constant and a non-constant expression;
3. analyze.py runs end to end on examples/example_dataset.csv (one KAN fit, no replay) and ranks
   `temperature` first, the descriptor the synthetic yield depends on most.

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


def test_analyze_example(out='ci_out'):
    out = os.path.join(ROOT, out)
    cmd = [sys.executable, os.path.join(ROOT, 'analyze.py'), os.path.join(ROOT, 'examples', 'example_dataset.csv'),
           '--target', 'yield', '--fits', '1', '--out', out]
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    for f in ('report.txt', 'descriptor_importance.csv', 'summary.png'):
        assert os.path.isfile(os.path.join(out, f)), 'missing output ' + f
    table = pd.read_csv(os.path.join(out, 'descriptor_importance.csv'))
    ranking = table['descriptor'].tolist()
    print('ranking:', ', '.join('%s %.1f%%' % (d, 100 * s) for d, s in zip(ranking, table['AGE_share_mean'])))
    assert ranking[0] == 'temperature', 'expected temperature first, got %s' % ranking
    print('ok  analyze.py on the example dataset ranks temperature first')


if __name__ == '__main__':
    test_imports()
    test_eval_func()
    test_analyze_example()
    print('smoke test passed')
