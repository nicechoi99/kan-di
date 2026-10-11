import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import json
import traceback

import numpy as np
import pandas as pd

from datamanager import DATASET, load_dataset, load_results, get_random_indices, gen_dataset_few_initial
from models import DiscreteBO
from config import COMBINATIONS, FEATURE_RANGE, OUTPUT_RANGE
from utils import plt, checkexists, datadir, imgdir, cprint, timestamp, save, load
from putils import parallel_eval, ray   # serial stand-in for the cluster backend


def run_BO(inputs, params=None, i=None):
    if type(params) is ray._raylet.ObjectRef:
        params = ray.get(params)
    
    m, acquisition, data_name, seed = inputs
    dataset_name, datasets, hyperparams_map, n_initial, draw, showfig, savefig, parallel, verbose = params
    
    prefix = data_name + '_' + m + '_' + acquisition
    filedir = os.path.join(datadir, dataset_name, 'temp', prefix + '_' + str(seed) + '.pkl')
    
    try:
        cprint('Running case for', filedir)
        df = datasets[data_name]
        y = df['Y'].values
        y_subopt = OUTPUT_RANGE[0] + 0.9 * (OUTPUT_RANGE[1] - OUTPUT_RANGE[0])
        indices_initial = get_random_indices(y=y, lower=0.5, n_sample=n_initial, seed=seed)
        hyperparams = hyperparams_map.get(m)
        imgoutdir = os.path.join(imgdir, dataset_name, prefix, str(seed))
        
        agent = DiscreteBO(
            dataset_name, df, indices_initial=indices_initial, x_range=FEATURE_RANGE, y_range=OUTPUT_RANGE,
            m=m, kernel='GP', acquisition=acquisition, y_subopt=y_subopt, hyperparams=hyperparams, 
            draw=draw, showfig=showfig, savefig=savefig, imgoutdir=imgoutdir, parallel=parallel
        )
        
        log = agent.run(verbose=verbose)
        
        cprint('Job', filedir, 'done.', color='g')
        log['data_name'] = data_name
        log['seed'] = seed
        log['m'] = m
        log['acquisition'] = acquisition
        
    except Exception as e:
        cprint('Job', filedir, 'has been terminated with error.', color='y')
        error_info = {
            '__error__': True,
            'data_name': data_name,
            'seed': seed,
            'm': m,
            'acquisition': acquisition,
            'error_type': type(e).__name__,
            'error_msg': str(e),
            'traceback': traceback.format_exc(),
        }
        log = pd.DataFrame.from_records([error_info])
    
    # Save
    os.makedirs(os.path.dirname(filedir), exist_ok=True)
    log.to_pickle(filedir)
    return log


def handle_file(m, acquisition, dataset_name, data_name, seed, force=False):
    job = (m, acquisition, data_name, seed)
    prefix = f'{data_name}_{m}_{acquisition}'
    filedir = os.path.join(datadir, dataset_name, 'temp', f'{prefix}_{seed}.pkl')
    filename = os.path.basename(filedir)

    if not checkexists(filedir) or force:
        cprint(filename, 'has been included into jobs', color='w')
        return [job]
    
    log = pd.read_pickle(filedir)
    if '__error__' in log.columns:
        cprint(filename, 'has been terminated with error. Rerun the simulation case.', color='y')
        return [job]

    # cprint(filename, 'has been loaded without error.', color='g')
    return []


def load_kan_hyperparams(dataset_name):
    """KAN sparsity settings selected for the group (dat/<group>/best_param_KAN.json; a local .pkl wins)."""
    pkl = os.path.join(datadir, dataset_name, 'best_param_KAN.pkl')
    if os.path.exists(pkl):
        return load(pkl)
    with open(os.path.join(datadir, dataset_name, 'best_param_KAN.json')) as f:
        return json.load(f)


def run_BO_all(dataset_name='large_feature', data_names=None, n_seed=10, combinations=None, draw=False,
               force=False):
    # BO agent parameters
    n_initial = 10
    verbose = False  # whether to inspect training process, resulting functions, etc.
    showfig = False  # whether to show the monitoring panel
    savefig = True  # whether to save the monitoring panel (only when draw=True)
    combinations = COMBINATIONS if combinations is None else combinations
    
    # Parallel configuration
    parallel = False
    debug = False  # debugging for parallel=True mode
    target = 'local'  # choose between 'local' and 'dist'
    
    # Load datasets
    datasets = load_dataset(dataset_name)
    if data_names:
        datasets = {k: datasets[k] for k in data_names}
        
    # Normalize combinations to upper-case
    data_names = list(datasets.keys())
    seeds = list(range(n_seed))
    
    # COMBINATIONS.remove(('KAN', 'DI')) #! override
    
    # Define entire jobs
    jobs = []
    for m, acquisition in combinations:
        for data_name in data_names:
            for seed in seeds:
                job = (m, acquisition, data_name, seed)
                jobs.extend(handle_file(m, acquisition, dataset_name, data_name, seed, force=force))
    
    cprint('Total number of jobs:', len(jobs))
    
    # Build hyperparams map only for used models
    unique_m = sorted({m for m, _, _, _ in jobs})
    hyperparams_map = {}
    for m in unique_m:
        if m == 'KAN':
            hyperparams_map['KAN'] = load_kan_hyperparams(dataset_name)
        else:
            hyperparams_map[m] = None

    params = (dataset_name, datasets, hyperparams_map, n_initial, draw, showfig, savefig, parallel, verbose)

    # Run all jobs; per-job caching in run_BO makes re-runs cheap
    parallel_eval(run_BO, jobs, params=params, target=target, parallel=parallel, debug=debug)
    
    # Return every requested run, including the ones cached from earlier calls
    logs = []
    for m, acquisition in combinations:
        for data_name in data_names:
            for seed in seeds:
                filedir = os.path.join(datadir, dataset_name, 'temp', f'{data_name}_{m}_{acquisition}_{seed}.pkl')
                if os.path.exists(filedir):
                    logs.append(pd.read_pickle(filedir))
    return logs


def KAN_BO_test():
    draw = True
    showfig = True
    savefig = False
    parallel = False
    verbose = True
    
    m = 'KAN'
    acquisition = 'DI-EI'
    
    dataset_name = 'small_feature'
    datasets = load_dataset(dataset_name)
    data_name = 'AutoAM'
    n_initial = 10
    n_seed = 10
    seed = 0
    
    df = datasets[data_name]
    y = df['Y'].values
    y_subopt = OUTPUT_RANGE[0] + 0.9 * (OUTPUT_RANGE[1] - OUTPUT_RANGE[0])
    indices_initial = get_random_indices(y=y, lower=0.5, n_sample=n_initial, seed=seed)
    hyperparams = load_kan_hyperparams(dataset_name)
    
    prefix = data_name + '_' + m + '_' + acquisition
    imgoutdir = os.path.join(imgdir, dataset_name, prefix, str(seed))
    
    agent = DiscreteBO(
        dataset_name, df, indices_initial=indices_initial, x_range=FEATURE_RANGE, y_range=OUTPUT_RANGE,
        m=m, kernel='GP', acquisition=acquisition, y_subopt=y_subopt, hyperparams=hyperparams, 
        draw=draw, showfig=showfig, savefig=savefig, imgoutdir=imgoutdir, parallel=parallel
    )
    
    log = agent.run(verbose=verbose)
    return log


def summarize(logs):
    """Median iterations to the target per pool and method, counted as in the paper."""
    y_subopt = OUTPUT_RANGE[0] + 0.9 * (OUTPUT_RANGE[1] - OUTPUT_RANGE[0])
    rows = []
    for log in logs:
        if '__error__' in log.columns:
            continue
        hit = log[log['y_next'] > y_subopt].sort_values('k')
        rows.append({'data_name': log['data_name'].iloc[0], 'method': log['m'].iloc[0] + '-' + log['acquisition'].iloc[0],
                     'seed': log['seed'].iloc[0], 'iterations': int(hit['k'].iloc[0]) if len(hit) else np.nan})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df.groupby(['data_name', 'method'])['iterations'].agg(['median', 'count']).reset_index()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Benchmark sweep of the paper on the pools under dat/.')
    ap.add_argument('--dataset', default='large_feature', choices=sorted(DATASET),
                    help='pool group (default: large_feature)')
    ap.add_argument('--data', default=None, help='comma-separated pool names, e.g. AutoAM or MOF_Td (default: all)')
    ap.add_argument('--seeds', type=int, default=10, help='random initial designs per method (default: 10)')
    ap.add_argument('--methods', default=None,
                    help='comma-separated prior:acquisition pairs, e.g. KAN:DI-UCB-H,ZERO:EI (default: all in config)')
    ap.add_argument('--draw', action='store_true', help='save the monitoring panel of every iteration under img/')
    ap.add_argument('--force', action='store_true', help='rerun jobs whose result file already exists')
    args = ap.parse_args()
    combos = [tuple(c.split(':')) for c in args.methods.split(',')] if args.methods else None
    names = args.data.split(',') if args.data else None
    logs = run_BO_all(args.dataset, names, args.seeds, combos, draw=args.draw, force=args.force)
    table = summarize(logs)
    if table is not None:
        print(table.to_string(index=False))
