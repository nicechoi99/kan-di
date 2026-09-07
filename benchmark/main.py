import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
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
    
    # try:
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
        
    # except Exception as e:
    #     cprint('Job', filedir, 'has been terminated with error.', color='y')
    #     error_info = {
    #         '__error__': True,
    #         'data_name': data_name,
    #         'seed': seed,
    #         'm': m,
    #         'acquisition': acquisition,
    #         'error_type': type(e).__name__,
    #         'error_msg': str(e),
    #         'traceback': traceback.format_exc(),
    #     }
    #     log = pd.DataFrame.from_records([error_info])
    
    # # Save
    # log.to_pickle(filedir)
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


def run_BO_all():
    # BO agent parameters
    n_initial = 10
    n_seed = 10
    verbose = False  # whether to inspect training process, resulting functions, etc.
    draw = True  # whether to display the monitoring panel on the fly (#!you can generate fig from the log and then save)
    showfig = False  # whether to show the monitoring panel
    savefig = True  # whether to save the monitoring panel
    force = False  # whether to force re-run even if the result file exists 
    
    # Parallel configuration
    parallel = False
    debug = False  # debugging for parallel=True mode
    target = 'local'  # choose between 'local' and 'dist'
    
    # Run Bayesian optimization with the target prior/acquisition function combination
    dataset_name = 'large_feature'
    
    # Load datasets
    datasets = load_dataset(dataset_name)
        
    # Normalize combinations to upper-case
    data_names = list(datasets.keys())
    seeds = list(range(n_seed))
    
    # COMBINATIONS.remove(('KAN', 'DI')) #! override
    
    # Define entire jobs
    jobs = []
    for m, acquisition in COMBINATIONS:
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
            hyperparams_map['KAN'] = load(os.path.join(datadir, dataset_name, 'best_param_KAN.pkl'))
        else:
            hyperparams_map[m] = None

    params = (dataset_name, datasets, hyperparams_map, n_initial, draw, showfig, savefig, parallel, verbose)

    # Run all jobs; per-job caching in run_BO makes re-runs cheap
    logs = parallel_eval(run_BO, jobs, params=params, target=target, parallel=parallel, debug=debug)
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
    hyperparams = load(os.path.join(datadir, dataset_name, 'best_param_KAN.pkl'))
    
    prefix = data_name + '_' + m + '_' + acquisition
    imgoutdir = os.path.join(imgdir, dataset_name, prefix, str(seed))
    
    agent = DiscreteBO(
        dataset_name, df, indices_initial=indices_initial, x_range=FEATURE_RANGE, y_range=OUTPUT_RANGE,
        m=m, kernel='GP', acquisition=acquisition, y_subopt=y_subopt, hyperparams=hyperparams, 
        draw=draw, showfig=showfig, savefig=savefig, imgoutdir=imgoutdir, parallel=parallel
    )
    
    log = agent.run(verbose=verbose)
    return log


if __name__ == '__main__':
    # log = KAN_BO_test()
    
    logs = run_BO_all()
    
    a = 1
