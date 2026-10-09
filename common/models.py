import os
import platform
import warnings
import itertools
import random
from copy import deepcopy
from collections import Counter

import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import norm
import torch
from linear_operator.utils.warnings import NumericalWarning
from linear_operator.utils.errors import NotPSDError
from torch.distributions.normal import Normal
from torch.optim import Adam
from gpytorch.models import ExactGP
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.priors import GammaPrior
from gpytorch.means import Mean, ZeroMean, ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import cholesky_jitter, cholesky_max_tries
from kan import ex_round
from kan.custom import MultKAN, truncate_numbers, replace_half_powers
from kan.utils import create_dataset_from_data

from datamanager import get_X, get_Z, get_Y, get_Ymax, load_dataset, DATASET
from plotter import plot_data, plot_2D_contour, plot_regressor_accuracy, plot_XY, \
    plot_trajectory, plot_descriptor_transition, plot_descriptor_distribution, CMAP_IMPROVEMENT
from utils import plt, mpl, gridspec, imgdir, datadir, cprint, flatten_list, isarray, \
    cleanup_directory, zoomout, checkexists, flatten_list, load
from putils import ray, parallel_eval


warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')


def calc_distance(x1, x2, mode='L2-norm'):
    if mode == 'L2-norm':
        return np.sqrt(np.sum((x1 - x2)**2))
    else:
        raise NotImplementedError


def detach(value, to_list=False):
    """Convert torch.Tensor, numpy array, or scalar to a Python native type."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        # Return scalar if 0-dimensional tensor
        if value.ndim == 0:
            return value.item()
        # Return list if to_list=True, otherwise numpy array
        elif to_list:
            return value.numpy().tolist()
        else:
            return value.numpy()
    elif isinstance(value, np.ndarray):
        # Return scalar if single element
        if value.ndim == 0 or (value.ndim == 1 and value.size == 1):
            return float(value)
        # Return list if requested, otherwise keep as array
        elif to_list:
            return value.tolist()
        else:
            return value
    elif isinstance(value, (np.number, np.bool_)):
        return float(value)
    else:
        # Already a Python native type (int, float, list, etc.)
        return value


def tensorize(dataset):
    for name, data in dataset.items():
        if isinstance(data, torch.Tensor):
            pass
        elif isinstance(data, np.ndarray):
            dataset[name] = torch.from_numpy(data).float()
        else:
            raise RuntimeError("unexpected behavior")
    return dataset


def inverse_transformation(formula):  # x = exp(z) or equivalently z = log(x) transformed case
    new_expr = deepcopy(formula)
    subs = {}
    for sym in new_expr.free_symbols:
        subs[sym] = sp.log(sym)  # replace z -> log(x)
    new_expr = new_expr.xreplace(subs)
    return new_expr
    

def sqrt_clip(z, threshold=1e-6):
    z = np.asarray(z, dtype=float)
    return np.sqrt(np.clip(z, threshold, None))


def pow_clip(base, exp):
    base = np.asarray(base, dtype=float)
    try:
        if np.isscalar(exp) and float(exp) == 0.5:
            return sqrt_clip(base)
    except Exception:
        pass
    
    if isinstance(exp, (np.ndarray, list, tuple)) and np.allclose(np.asarray(exp, dtype=float), 0.5):
        return sqrt_clip(base)    
    return np.power(base, exp)
    

def log_clip(z, threshold=-10.0):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    mask = z > 0.0
    out[~mask] = float(threshold)
    out[mask] = np.log(z[mask])
    return out


clip_functions = {'sqrt': sqrt_clip, '**': pow_clip, 'pow': pow_clip, 'log': log_clip}
    
    
def eval_func(func, x):
    x = np.asarray(x, dtype=float)
    args = [x[:, i] for i in range(x.shape[1])]
    value = func(*args)
    
    # A derivative that simplifies to a constant comes back from lambdify as a scalar; broadcast it to one
    # value per row so callers can reshape it (otherwise AGE raised on, e.g., a linear KAN expression).
    n = x.shape[0]
    if isinstance(value, list):  # maybe input x is grid
        value = np.vstack([np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in value]).T.mean(axis=0)
    else:
        value = np.asarray(value, dtype=float)
        value = np.full(n, float(value)) if value.ndim == 0 else value.squeeze()
    return value


def apply_epsilon(x, x_range, eps):
    xmin, xmax = x_range

    if isinstance(x, np.ndarray):
        if np.any(x < xmin) or np.any(x > xmax):
            raise ValueError('Unexpected behavior: x out of xbounds')
        x_ = x.copy()
        x_[x_ == xmin] = xmin + eps
        x_[x_ == xmax] = xmax - eps
        return x_
    elif isinstance(x, torch.Tensor):
        if torch.any(x < xmin) or torch.any(x > xmax):
            raise ValueError('Unexpected behavior: x out of xbounds')
        x_ = x.clone()
        x_[x_ == xmin] = xmin + eps
        x_[x_ == xmax] = xmax - eps
        return x_
    else:
        raise TypeError('x must be a numpy.ndarray or torch.Tensor')


def plot_equation(f, x_range=(0.1, 0.9), eps=1e-6, n_points=100, n_discretization=5, figsize=(10, 3)):
    xvars = sorted(list(f.free_symbols), key=lambda s: s.name)
    input_dim = len(xvars)
    
    _f = sp.lambdify(xvars, f, modules=[clip_functions, 'numpy'])

    fig, axes = plt.subplots(1, input_dim, figsize=(10, 2.5))
    x_grid = np.linspace(*x_range, n_discretization)
    x_grid[0] += eps
    x_grid[-1] -= eps
    points = list(itertools.product(x_grid, repeat=input_dim-1))
    
    for i, xi in enumerate(xvars):
        ax = axes[i]
        xline = np.linspace(*x_range, n_points)
        
        for p in points:
            xothers = np.stack(p)
            Xothers = np.tile(xothers, (n_points, 1))
            X = np.insert(Xothers, i, xline, axis=1)
            yline = eval_func(_f, X)
            ax.plot(xline, yline, lw=1.0, alpha=0.9, color='k')

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xi)
        if i == 0:
            ax.set_ylabel('y(X)')
    
    fig.tight_layout()
    return fig, axes


def eval_AGE_numeric(df, X, xvars, x_range, eps=1e-6, n_samples_i=10, n_samples_j=1000, threshold=1e6):
    """Estimate AGE delta_i and lambda_i numerically via stratified sampling."""
    d = len(xvars)

    # Normalize xbounds to per-variable list of tuples
    if isinstance(x_range, tuple) and len(x_range) == 2:
        xbounds = [x_range] * d
    elif isinstance(x_range, (list, tuple)) and all(
            isinstance(t, tuple) and len(t) == 2 for t in x_range):
        if len(x_range) != d:
            raise ValueError('Length of x_range list must match number of xvars.')
        xbounds = list(x_range)
    else:
        raise TypeError('x_range must be (a, b) or [(a1, b1), ..., (ad, bd)].')
    
    # Vectorized deltas (keep original modules usage)
    deltas = [sp.lambdify(xvars, df[i], modules=[clip_functions, 'numpy']) for i in range(d)]

    # Per-variable low/high/span with boundary guard
    lows = np.array([a + eps for a, b in xbounds], dtype=float)
    highs = np.array([b - eps for a, b in xbounds], dtype=float)
    spans = np.maximum(highs - lows, eps)

    lambdas = np.zeros(d, dtype=float)
    N = n_samples_i * n_samples_j
    X_input = X  # preserve original input for delta evaluation

    for i in range(d):
        # Stratified grid along x_i
        z_i_grid = np.linspace(lows[i], highs[i], n_samples_i)

        # Random for others, overwrite column i with grid
        X_rand = lows + spans * np.random.rand(N, d)
        X_rand[:, i] = np.repeat(z_i_grid, repeats=n_samples_j)

        # Evaluate and reshape
        vals = eval_func(deltas[i], X_rand)
        vals = np.asarray(vals, dtype=float).reshape(n_samples_i, n_samples_j)

        # Robust averaging
        mask = np.isfinite(vals) & (np.abs(vals) < threshold)
        safe = np.where(mask, vals, np.nan)
        cj = np.nanmean(safe, axis=1)

        lambdas[i] = 0.0 if np.all(np.isnan(cj)) else np.nanmean(cj)

    deltas = np.vstack([eval_func(deltas[i], X_input) for i, delta in enumerate(deltas)]).T
    return deltas, lambdas


def eval_AGE_analytic(df, xvars, x_range):
    """Analytically compute AGE lambda_i via symbolic integration over a uniform domain."""
    d = len(xvars)

    # Normalize x_range
    if isinstance(x_range, tuple):
        xbounds = [x_range] * d
    elif isinstance(x_range, list) and all(isinstance(t, tuple) and len(t) == 2 for t in x_range):
        if len(x_range) != d:
            raise ValueError('Length of x_range list must match number of xvars.')
        xbounds = x_range
    else:
        raise TypeError('x_range must be either a tuple (a, b) or list of tuples [(a1, b1), ...].')

    # Compute total integration volume
    vol = np.prod([b - a for (a, b) in xbounds])

    lambdas = np.zeros(d, dtype=float)

    # Loop over each variable derivative expression
    for i, expr in enumerate(df):
        try:
            integ = expr
            # Integrate sequentially using possibly different xbounds per variable
            for j, x in enumerate(xvars):
                a, b = xbounds[j]
                integ = sp.integrate(integ, (x, a, b))
            lambdas[i] = float(sp.simplify(integ / vol))
        except Exception as e:
            print(f'Integration failed for variable {xvars[i]}: {e}')
            lambdas[i] = np.nan

    return lambdas


class KANMean(Mean):
    def __init__(self):
        super(KANMean, self).__init__()
    
    
    def plot_training_process(self, result):
        fig, ax = plt.subplots(1, 1)
        tr = np.array(result['train_loss']).ravel()
        te = np.array(result['test_loss']).ravel()
        ax.plot(tr, label='Train')
        ax.plot(te, label='Test')
        
        # Decoration
        ax.legend(frameon=False, loc='upper right')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (RMSE)')
        return
    
    
    def get_dimensions(self, X, y):
        input_dim = X.shape[-1]
        
        if y.ndim == 1:
            output_dim = 1
        else:
            output_dim = y.shape[-1]
        
        return input_dim, output_dim
    
    
    def set_seed(self, seed):  # randomness control
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        eval("setattr(torch.backends.cudnn, 'deterministic', True)")  # for parallelization
        eval("setattr(torch.backends.cudnn, 'benchmark', False)")
        torch.cuda.manual_seed_all(seed)
        return
    
    
    def get_x_range(self, dataset):
        X, y = self.concat_data(dataset)
        
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        assert len(torch.unique(X_min.values)) == 1
        assert len(torch.unique(X_max.values)) == 1
        
        X_min = X_min.values[0]
        X_max = X_max.values[0]
        return torch.stack([X_min, X_max])
    

    def get_y_range(self, dataset):
        X, y = self.concat_data(dataset)
        
        y_min, y_max = y.min(axis=0), y.max(axis=0)
        
        y_min = y_min.values[0]
        y_max = y_max.values[0]
        return torch.stack([y_min, y_max])
    
    
    def transform_data(self, dataset):  # transform x by z = log(x)
        X_train, X_test, _, _ = self.unpack_data(dataset)
        
        X_train = torch.log(X_train)
        X_test = torch.log(X_test)
        
        dataset['train_input'] = X_train
        dataset['test_input'] = X_test
        return dataset
    
    
    def unpack_data(self, dataset):
        X_train = dataset['train_input']
        X_test = dataset['test_input']
        y_train = dataset['train_label']
        y_test = dataset['test_label']
        return X_train, X_test, y_train, y_test
    
    
    def concat_data(self, dataset):
        X_train, X_test, y_train, y_test = self.unpack_data(dataset)
        X = torch.concat([X_train, X_test], axis=0)
        y = torch.concat([y_train, y_test], axis=0)
        return X, y
    
    
    def check_data(self, dataset):
        # Variable dimension should be the same for train and test data
        assert dataset['train_input'].shape[1] == dataset['test_input'].shape[1]
        
        # Prediction target (y) should be in (n,1) matrix
        for key in ['train_label', 'test_label']:
            if dataset[key].ndim == 1:
                dataset[key] = dataset[key].reshape(-1,1)
        return dataset
    
    
    def fit(self, dataset, x_range=None, y_range=None, mult_arity=2, grid=5, seed=0, 
            k=3, hidden_dim=[1, 1], sparse_init=False, update_grid=False, device='cpu', opt='LBFGS', 
            lr=0.1, check_initialization=True, log_transformation=True, pre_regularization=True, 
            lamb=0.01, lamb_coef=0.1, lamb_coefdiff=0.1, lamb_entropy=0.01, pruning_th=0.1,
            draw=False, verbose=True):
        self.dataset = dataset
        self.mult_arity = mult_arity
        self.grid = grid
        self.seed = seed
        self.k = k
        self.sparse_init = sparse_init
        self.update_grid = update_grid
        self.device = device
        self.opt = opt
        self.lr = lr
        self.check_initialization = check_initialization
        self.log_transformation = log_transformation
        self.pre_regularization = pre_regularization
        self.lamb = lamb
        self.lamb_entropy = lamb_entropy
        self.lamb_coef = lamb_coef
        self.lamb_coefdiff = lamb_coefdiff
        self.pruning_th = pruning_th
        self.draw = draw
        self.verbose = verbose
        
        # Check dataformat
        dataset = deepcopy(dataset)  # unlink
        dataset = self.check_data(dataset)
        
        # Randomness control
        self.set_seed(seed)
        
        # Data transformation
        dataset = tensorize(dataset)
        if log_transformation:
            dataset = self.transform_data(dataset)
        self.log_transformation = log_transformation
        
        # Get feature range
        if x_range is None:
            x_range = self.get_x_range(dataset)
        self.x_range = x_range
        
        if y_range is None:
            y_range = self.get_y_range(dataset)
        self.y_range = y_range
        
        # Get X and y for future use
        if (len(dataset['train_label']) == len(dataset['test_label'])) and \
            all(dataset['train_label'] == dataset['test_label']):  # in BO
            X = dataset['train_input']
            y = dataset['train_label']
        else:
            X, y = self.concat_data(dataset)
            
        self.X = X
        self.y = y
        
        # Set architecture dimension
        input_dim, output_dim = self.get_dimensions(X, y)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Set xvars
        xvars_string = ['x' + str(i) for i in range(input_dim)]
        xvars = sp.symbols(xvars_string)
        self.xvars = xvars
        
        # Set optimizer configuration
        assert opt in ['Adam', 'AdamW', 'LBFGS']
        if opt in ['Adam', 'AdamW']:
            steps_train = 500
            steps_retrain = 200
        elif opt == 'LBFGS':
            steps_train = 100
            steps_retrain = 50
        else:
            raise NotImplementedError
        
        # Generate network
        model = MultKAN(width=[input_dim, hidden_dim, output_dim], k=k, 
                        mult_arity=mult_arity, grid=grid, grid_range=x_range, 
                        seed=seed, sparse_init=sparse_init, device=device)
        self.model = model

        # Check initialized network
        if check_initialization and draw:
            model(X)
            model.plot()

        # Train
        cprint('Training KAN model...')
        if pre_regularization:
            result = model.fit(dataset, opt=opt, lr=lr, steps=steps_train, 
                                update_grid=update_grid, verbose=verbose,
                                lamb=lamb, lamb_entropy=lamb_entropy, 
                                lamb_coef=lamb_coef, lamb_coefdiff=lamb_coefdiff)
        else:
            result = model.fit(dataset, opt=opt, lr=lr, steps=steps_train, 
                                update_grid=update_grid, verbose=verbose)

        # Pruning
        cprint('Pruning node and edges of KAN model...')
        model_sparse = model.prune(node_th=pruning_th, edge_th=pruning_th)
        
        # Re-training with pruned model
        cprint('Re-training with pruned KAN model...')
        result_sparse = model_sparse.fit(dataset, opt=opt, lr=lr, steps=steps_retrain, 
                                         update_grid=update_grid, verbose=verbose,
                                         lamb=lamb, lamb_entropy=lamb_entropy, 
                                         lamb_coef=lamb_coef, lamb_coefdiff=lamb_coefdiff)
        self.model_sparse = model_sparse
        
        # Symbolification
        self.formula = self.get_formula(verbose=verbose)
        
        if len(self.formula.free_symbols) == 0:
            raise RuntimeError('Unexpected behavior! self.get_formula returned may have returned constant value')
        
        if draw:
            if verbose:
                self.plot_training_process(result)
                self.plot_training_process(result_sparse)
            
            self.model.plot()
            self.model_sparse.plot()
            
            self.plot_parity(use_formula=True)
            plt.show()
        
        return {'result': result, 'result_sparse': result_sparse}
    
    
    def get_formula(self, weight_simple=0.0, verbose=True, pretty=True):
        if self.log_transformation:
            lib = ['1/x', 'x', 'exp', 'gaussian']
        else:
            lib = ['1/x', 'x', 'exp', 'gaussian', 'sqrt', 'log']
        
        # Get symbolic functions
        self.model_sparse.auto_symbolic(lib=lib, weight_simple=weight_simple, verbose=verbose)
        equations = self.model_sparse.symbolic_formula(var=self.xvars)[0]
        assert len(equations) == 1
        formula = equations[0]
        
        # Get pretty symbolic function
        if pretty:
            formula = truncate_numbers(formula)
        
        # Replace x**0.5 to sqrt(x)
        formula = replace_half_powers(formula)
        
        # Get the inverse-transformed equation
        if self.log_transformation:
            formula = inverse_transformation(formula)
        
        if verbose:
            print(formula)
        return formula
    
    
    def get_complexity(self, verbose=True):
        complexity = self.model_sparse.eval_complexity()
        
        if verbose:
            print(complexity)
        return complexity
    
    
    def plot_parity(self, dataset=None, use_formula=True, use_sparse=True):
        """
        Make sure that "dataset" is intact one (is not scaled)
        """
        if dataset is None:  # use archived
            dataset = self.dataset
        
        # Parse data
        X_train, X_test, y_train, y_test = self.unpack_data(dataset)
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        
        # Get predictions
        self.model.eval()
        self.model_sparse.eval()
        with torch.no_grad():
            y_train_pred = detach(self.forward(X_train, use_formula=use_formula, use_sparse=use_sparse))
            y_test_pred = detach(self.forward(X_test, use_formula=use_formula, use_sparse=use_sparse))
        
        # Plot
        fig, ax = plt.subplots(1, 1)
        ax.scatter(y_train, y_train_pred, label='Train', facecolor='k', edgecolor='k', zorder=1)
        ax.scatter(y_test, y_test_pred, label='Test', facecolor='w', edgecolor='k', zorder=0)
        
        # Draw y=x line
        ax.plot([0, 1], [0, 1], color='k', linestyle='-', label='y(pred) = y(exp)', zorder=-2)
        
        # Decoration
        ax.set_xlabel('y(exp)')
        ax.set_ylabel('y(pred)')
        ax.legend(frameon=False, loc='upper left')
        return
    
    
    def forward(self, x, use_formula=True, use_sparse=True):
        if isinstance(x, torch.Tensor):
            x = x.float()
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            raise RuntimeError("unexpected behavior")
        
        if use_formula:  # formula is already inverse-transformed, so no need log-transformation
            f_KAN = sp.lambdify(self.xvars, self.formula, [clip_functions, 'numpy'])
            y = eval_func(f_KAN, x)
            y = torch.from_numpy(y).float()
        else:
            if self.log_transformation:
                x = torch.log(x)
                
            if use_sparse:
                y = self.model_sparse(x).squeeze()
            else:
                y = self.model(x).squeeze()
        
        if torch.isnan(y).any():
            raise RuntimeError("unexpected behavior")
        return y
                
    
class ExactGPModel(ExactGP):
    def __init__(self, x, y, likelihood, nu, prior_length, prior_alpha, mean=None):
        super(ExactGPModel, self).__init__(x, y, likelihood)
        
        # Mean
        self.mean = mean
        
        # Base kerel
        kernel = MaternKernel(nu=nu, ard_num_dims=x.shape[-1], lengthscale_prior=prior_length['param'])
        
        # Covraiance
        self.cov = ScaleKernel(kernel, outputscale_prior=prior_alpha['param'])
        self.cov.outputscale = prior_alpha['mode']  # initial value
        self.cov.base_kernel.lengthscale = prior_length['mode']  # initial value

    
    def forward(self, x):
        x_mu = self.mean(x)
        x_cov = self.cov(x)
        mu_and_sigma = MultivariateNormal(x_mu, x_cov)
        
        # Checker
        if torch.isnan(x_mu).any():
            raise RuntimeError("unexpected behavior")
        if torch.isnan(mu_and_sigma.mean).any():
            raise RuntimeError("unexpected behavior")
        if torch.isnan(mu_and_sigma.stddev).any():
            raise RuntimeError("unexpected behavior")
        return mu_and_sigma


class DiscreteBO:  #! convert df to solve maximization problem if necessary
    def __init__(self, dataset_name, df, indices_initial, x_range, y_range, m=0, kernel='GP',
                 acquisition='EI', y_subopt=0.8, nu=5/2, sigma_min=1e-5, device='cpu', hyperparams=None,
                 parallel=False, draw=True, showfig=True, savefig=True, imgoutdir=None):
        # Importing dataset
        self.dataset_name = dataset_name
        df.reset_index(drop=True, inplace=True)
        self.df = df
        self._X = get_X(df)
        self._Y = get_Y(df)
        
        self.indices = df.index.values.tolist()
        self.indices_initial = indices_initial
        
        self.input_dim = self._X.shape[-1]
        self.xvars = ['x' + str(i) for i in range(self.input_dim)]
        
        # Get y_subopt
        self.x_range = x_range
        self.y_range = y_range
        self.y_subopt = y_subopt
        self.indices_y_subopt = list(np.array(list(df.index))[self._Y >= self.y_subopt])
        
        # Set y_opt
        self.idx_opt = self.get_optimal_idx(df)
        self.y_opt = self._Y[self.idx_opt]
        
        # Configurations
        self.device = device
        self.parallel = parallel
        self.zlabel = df.attrs['embedder']
        self.draw = draw
        if parallel:
            showfig = False
        self.showfig = showfig
        self.savefig = savefig
        self.imgoutdir = imgoutdir
        
        # Set model hyperparameters
        self.kernel = kernel
        self.acquisition = acquisition
        self.nu = nu  # smoothness parameter
        self.sigma_min = sigma_min  # minimum noise
        self.prior_sigma = {'param': GammaPrior(0.5, 5.0), 'mode': 0.1}  # noise parameter / GammaPrior(alpha, beta)
        
        # Set m and kernel models
        if isinstance(m, Mean):
            self.m = m
        elif m == 0 or m.lower() == 'zero':
            self.m = ZeroMean()
        elif m == 'constant':
            self.m = ConstantMean()  # default
        elif m == 'KAN':
            self.m = KANMean()
            if hyperparams is None:
                self.hyperparams = load(os.path.join(datadir, self.dataset_name, 'best_param_KAN.pkl'))
            else:
                self.hyperparams = hyperparams
        else:
            raise NotImplementedError
        
        if kernel == 'GP':
            self.prior_alpha = {'param': GammaPrior(2.0, 0.5), 'mode': 2.0}  # kernal output scale parameter / GammaPrior(alpha, beta)
            self.prior_length = {'param': GammaPrior(2.0, 1.0), 'mode': 2.0}  # kernal length parameter
        else:
            raise NotImplementedError
        
        # Set logger
        self.log = self.initialize_log()
    
    
    @classmethod
    def set_figure_panel(self, acquisition, parallel=False):
        if acquisition in ['EI', 'UCB', 'TS']:
            fig = plt.figure(figsize=(6, 4))

            columns = fig.add_gridspec(
                1, 2,
                width_ratios=[1.0, 2.0],
                wspace=0.3,
                top=0.92, bottom=0.07, left=0.07, right=0.95
            )

            left  = columns[0, 0].subgridspec(2, 1, hspace=0.4)
            right = columns[0, 1].subgridspec(2, 2, hspace=0.4, wspace=0.3)

            axes = np.empty((2, 3), dtype=object)

            axes[0, 0] = fig.add_subplot(left[0, 0])
            axes[1, 0] = fig.add_subplot(left[1, 0])
            axes[0, 1] = fig.add_subplot(right[0, 0])
            axes[0, 2] = fig.add_subplot(right[0, 1])
            axes[1, 1] = fig.add_subplot(right[1, 0])
            axes[1, 2] = fig.add_subplot(right[1, 1])
        
        elif 'DI' in acquisition:
            fig = plt.figure(figsize=(7, 6))
            
            main = fig.add_gridspec(
                2, 1,
                height_ratios=[2, 0.9],
                hspace=0.2, top=0.92, bottom=0.07, left=0.07, right=0.95
            )

            top = main[0].subgridspec(1, 2, width_ratios=[1.0, 2.0], wspace=0.3)
            
            bottom = main[1].subgridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.2)
            bottom_left = bottom[0,0].subgridspec(1, 2, width_ratios=[2, 0.5], wspace=0.1)

            upper_left = top[0, 0].subgridspec(2, 1, hspace=0.4)
            upper_right = top[0, 1].subgridspec(2, 2, hspace=0.4, wspace=0.3)

            axes = np.empty((3, 3), dtype=object)
            
            axes[0, 0] = fig.add_subplot(upper_left[0, 0])
            axes[1, 0] = fig.add_subplot(upper_left[1, 0])
            
            axes[0, 1] = fig.add_subplot(upper_right[0, 0])
            axes[0, 2] = fig.add_subplot(upper_right[0, 1])
            axes[1, 1] = fig.add_subplot(upper_right[1, 0])
            axes[1, 2] = fig.add_subplot(upper_right[1, 1])
            
            axes[2, 0] = fig.add_subplot(bottom_left[0,0])
            axes[2, 1] = fig.add_subplot(bottom_left[0,1])
            axes[2, 2] = fig.add_subplot(bottom[0,1])
            
        else:
            raise NotImplementedError
        
        system = platform.system().lower()
        
        if system == 'windows' and not parallel:
            manager = plt.get_current_fig_manager()
            manager.window.wm_geometry("+0+0")  # set display location based on the monitor
        
        return fig, axes
    
    
    def run(self, post_iteration=10, verbose=True):
        df = self.df
        indices_initial = self.indices_initial
        indices_sampled = deepcopy(indices_initial)
        self.indices_sampled = indices_sampled
        
        K = len(df) - len(indices_initial)
        idx_prev = indices_sampled[-1]
        
        # Setup figures
        if self.draw:
            if self.savefig:
                assert self.imgoutdir
                cleanup_directory(self.imgoutdir)
                
            fig, axes = self.set_figure_panel(acquisition=self.acquisition, parallel=self.parallel)
        
        # Run BO
        counter = 0
        for k in range(K):
            self.k = k
            cprint('Running BO | Iter = (', k, '/', K, ')', color='c', inspect=False)
            
            # Construct the model and train it
            self.construct_model(indices_sampled, verbose=verbose)
            
            # Train the kernel
            self.fit(verbose=verbose)
            
            # Find current max
            y_max, idx_max = self.find_current_y_max(indices_sampled)
            self.y_max = y_max
            self.idx_max = idx_max
            
            # Propose next location
            idx_next, a_next, I_next, I_labels, descriptors, age_lambdas, kan_formula = self.propose_location(y_max, verbose=verbose)
            self.idx_next = idx_next
            
            # Conduct experiment
            y_next = self.virtual_experiment(idx_next)
            self.y_next = y_next
            indices_sampled.append(idx_next)
            self.indices_sampled = indices_sampled
        
            # Record progress
            log = self.record(k, idx_prev, idx_next, y_next, y_max, a_next, I_next, descriptors, age_lambdas, kan_formula)
            
            # Monitor progress
            if self.draw:
                self.monitor_progress(fig, axes, showfig=self.showfig, savefig=self.savefig)
            
            # Break or update
            self.update_priors()
            idx_prev = idx_next
            
            # Terminate
            if self.y_max > self.y_subopt:
                counter += 1
            
            if counter >= post_iteration:
                if 'DI' in self.acquisition:
                    if a_next < 0.1:
                        cprint('Termination condition reached.', color='g')
                        break
                else:
                    cprint('Termination condition reached.', color='g')
                    break
        return log
    
    
    def find_current_y_max(self, indices_sampled):
        y = get_Y(self.df)[indices_sampled]
        idx_max = indices_sampled[y.argmax()]
        y_max = y[y.argmax()]
        
        return y_max, idx_max
    
            
    def initialize_log(self):
        # prior_r2 / swing_ratio / prior_diverged added 2026-08-21. They record the state of the KAN
        # prior at each iteration, which nothing in the log captured before. Without prior_r2 there is
        # no way to tell afterwards whether a BO run improved because the prior became more accurate,
        # and without the swing columns there is no way to tell how many iterations ran on a diverged
        # symbolic formula. Both questions otherwise cost a full rerun to answer.
        return pd.DataFrame(columns=['k', 'x_prev', 'x_next', 'y_next', 'y_max', 'a_next', 'I_next',
                                     'descriptors', 'loss', 'noise', 'outputscale', 'lengthscale',
                                     'd_xnext_xprev', 'd_xnext_xopt',
                                     'age_lambdas', 'kan_formula',
                                     'prior_r2', 'swing_ratio', 'prior_diverged'])
    
    
    def get_optimal_idx(self, df):
        return int(np.argmax(df['Y'].values))
    
    
    def construct_model(self, indices_sampled, verbose=True):
        indices_unsampled = list(set(self.indices) - set(indices_sampled))
        
        # Parse data
        X = get_X(self.df, indices_sampled)
        y = get_Y(self.df, indices_sampled)
        _X = get_X(self.df, indices_unsampled)
        _y = get_Y(self.df, indices_unsampled)
        
        self.X = X
        self.y = y
        
        # Construct dataset
        dataset = {}
        dataset['train_input'] = X
        dataset['test_input'] = _X
        dataset['train_label'] = y.reshape(-1, 1)
        dataset['test_label'] = _y.reshape(-1, 1)
        
        # Update mu
        if isinstance(self.m, KANMean):
            self.m.train()
            for param in self.m.parameters():
                param.requires_grad = True
            
            self.m.fit(dataset, x_range=self.x_range, device=self.device, verbose=verbose, **self.hyperparams)
            
            self.m.eval()
            for param in self.m.parameters():
                param.requires_grad = False
        elif isinstance(self.m, (ZeroMean, ConstantMean)):
            pass
        else:
            raise NotImplementedError
        
        # Likelihood
        self.likelihood = GaussianLikelihood(noise_prior=self.prior_sigma['param'],
                                             noise_constraint=GreaterThan(self.sigma_min))
        self.likelihood.noise = torch.tensor([float(self.prior_sigma['mode'])])  # initial value
        
        # Posterior
        model = ExactGPModel(self.X, self.y, likelihood=self.likelihood, nu=self.nu, 
                             prior_length=self.prior_length, prior_alpha=self.prior_alpha, 
                             mean=self.m)
        
        # Set computation mode
        if torch.cuda.is_available() and self.device in ['gpu', 'cuda']:
            model = model.cuda()
        
        # For future use
        self.model = model
        return
    

    def fit(self, lr=0.1, n_iters=100, verbose=True):  # train the model
        self.model.train()
        self.likelihood.train()
        
        # Load data archived during model construction
        X = self.X
        y = self.y
        
        # Set optimizer
        optimizer = Adam(self.model.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(likelihood=self.likelihood, model=self.model)
        
        # Fit hyperparameters
        cprint('Updating parameters...')
        for i in range(n_iters):
            optimizer.zero_grad()
            yhat = self.forward(X)
            
            mean = yhat.mean
            cov = yhat.lazy_covariance_matrix
            
            warnings.filterwarnings("ignore", category=NumericalWarning)
            last_err = None
            for jitter in (1e-6, 1e-5, 1e-4, 1e-3):
                try:
                    with cholesky_jitter(jitter), cholesky_max_tries(5):
                        yhat = MultivariateNormal(mean, cov)
                        loss = -mll(yhat, y)
                    break
                except NotPSDError as e:
                    last_err = e
                    continue
            else:
                raise last_err
            
            loss.backward()
            
            loss = np.round(loss.item(), 3)
            noise = np.round(self.model.likelihood.noise.item(), 3)
            outputscale = np.round(self.model.cov.outputscale.item(), 2)
            lengthscale = np.round(self.model.cov.base_kernel.lengthscale.squeeze().tolist(), 2)
            
            if verbose and (i+1) % (n_iters/10) == 0:  # check convergence
                cprint('    Iter', i+1, '/', n_iters, '| loss:', loss, '| noise:', noise, '| outputscale:', outputscale, '| lengthscale:', lengthscale, color='w', inspect=False)
            
            optimizer.step()
        
        # Record states
        self.loss = loss
        self.noise = noise
        self.outputscale = outputscale
        self.lengthscale = lengthscale
        
        self.model.eval()
        self.likelihood.eval()
        return
    
    
    def forward(self, x):
        warnings.filterwarnings("ignore", category=NumericalWarning)
        last_err = None
        for jitter in (1e-6, 1e-5, 1e-4, 1e-3):
            try:
                with cholesky_jitter(jitter), cholesky_max_tries(5):
                    mu_and_sigma = self.model(x)
                    yhat = self.likelihood(mu_and_sigma)
                break
            except NotPSDError as e:
                last_err = e
                continue
        else:
            raise last_err

        return yhat


    def predict(self, x, tensor=True):
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            yhat = self.forward(x)
            
            mu = yhat.mean
            var = yhat.variance
            
        if tensor:
            return mu, var
        else:
            return detach(mu), detach(var)


    def sample_posterior(self, x, n_samples):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            posterior = self.forward(x)  # MultivariateNormal
            samples = posterior.sample(torch.Size([n_samples]))  # (n_samples, K)

            if n_samples == 1:
                samples = samples.squeeze(0)  # (K,)
        return samples
    
    
    def propose_location(self, y_max, verbose=True):
        indices_unsampled = list(set(self.indices) - set(self.indices_sampled))

        x = get_X(self.df, indices_unsampled)
        result = self.eval_acquisition(x, y_max, verbose=verbose)
        if len(result) == 6:
            a, I, I_labels, descriptors, age_lambdas, kan_formula = result
        else:
            a, I, I_labels, descriptors = result
            age_lambdas, kan_formula = None, None

        idx = int(torch.argmax(a))
        idx_next = indices_unsampled[idx]
        a_next = detach(a[idx])
        I_next = detach(I[idx,:])
        return idx_next, a_next, I_next, I_labels, descriptors, age_lambdas, kan_formula
    
    
    def record(self, k, idx_prev, idx_next, y_next, y_max, a_next, I_next, descriptors, age_lambdas=None, kan_formula=None):
        loss = self.loss
        noise = self.noise
        outputscale = self.outputscale
        lengthscale = self.lengthscale

        # Convert to Python native types for DataFrame storage
        y_next = detach(y_next)
        y_max = detach(y_max)

        X = get_X(self.df, tensor=False)
        x_prev = X[idx_prev,:]
        x_next = X[idx_next,:]
        x_opt = X[self.idx_opt,:]
        d_xnext_xprev = calc_distance(x_next, x_prev)
        d_xnext_xopt = calc_distance(x_next, x_opt)

        prior_r2, swing_ratio, prior_diverged = self.probe_prior()

        row = [k, x_prev, x_next, y_next, y_max, a_next, I_next, descriptors, loss, noise,
               outputscale, lengthscale, d_xnext_xprev, d_xnext_xopt,
               age_lambdas, kan_formula,
               prior_r2, swing_ratio, prior_diverged]
        self.log.loc[k] = row

        return self.log


    def probe_prior(self):
        """State of the KAN prior at this iteration: generalization and whether the formula blew up.

        prior_r2 is scored on the candidates not yet queried, which is a genuine holdout in an in-silico
        replay because their responses are known but were not used to fit. swing_ratio is the formula's
        output range over the whole pool divided by the response range; the symbolic stage intermittently
        produces formulas swinging many times the response, and neither R2 nor the acquisition reveals it.

        Returns NaNs for non-KAN priors and never raises, so a probe failure cannot end a BO run.
        """
        try:
            if not isinstance(self.m, KANMean) or getattr(self.m, 'formula', None) is None:
                return np.nan, np.nan, False
            X = get_X(self.df, tensor=False)
            y = np.asarray(get_Y(self.df, tensor=False), float)
            f = self.m.formula
            zv = sorted(f.free_symbols, key=lambda s: s.name)
            keep = [int(v.name.split('x')[1]) for v in zv]
            if not keep:
                return np.nan, 0.0, False
            # lambdify compiles, and a [10,5] formula is large enough that recompiling it once per
            # iteration costs seconds. The formula only changes when the prior is refitted, so cache on
            # its string form and reuse across iterations that share it.
            key = str(f)
            cache = getattr(self, '_probe_cache', None)
            if cache is None or cache[0] != key:
                cache = (key, sp.lambdify(zv, f, 'numpy'), keep)
                self._probe_cache = cache
            g, keep = cache[1], cache[2]
            pred = np.asarray(g(*[X[:, j] for j in keep]), float).ravel() * np.ones(len(X))

            fin = np.isfinite(pred)
            yr = float(np.nanmax(y) - np.nanmin(y))
            swing = float(pred[fin].max() - pred[fin].min()) if fin.any() else np.inf
            ratio = swing / yr if yr > 0 else np.inf

            held = np.setdiff1d(np.arange(len(X)), np.asarray(self.indices_sampled, dtype=int))
            ok = held[np.isfinite(pred[held]) & np.isfinite(y[held])] if len(held) else held
            if len(ok) > 2 and np.var(y[ok]) > 0:
                r2 = float(1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))
            else:
                r2 = np.nan
            return r2, ratio, bool(ratio > 3.0)
        except Exception:
            return np.nan, np.nan, False
        
    
    def eval_DI(self, x, y_max, acquisition, eps=1e-6):
        # Unpack
        X = self.X
        y = self.y
        y_best = torch.max(y)
        y_subopt = self.y_subopt
        x_range = self.x_range
        y_range = self.y_range
        
        # Explorative improvement function
        if acquisition == 'DI-EI':
            mu, var = self.predict(x)
            sigma = torch.sqrt(var)
            
            diff = mu - y_max
            z = diff/sigma
            normal = Normal(0, 1)
            _I_exploitation = diff * normal.cdf(z)
            _I_exploration = sigma * normal.log_prob(z).exp()
            yhat = _I_exploitation + _I_exploration
            yhat[sigma <= 0.] = 0.
            
        elif acquisition == 'DI-UCB-L':
            mu, var = self.predict(x)
            sigma = torch.sqrt(var)
            yhat = mu + 0.1*sigma
            
        elif acquisition == 'DI-UCB-H':
            mu, var = self.predict(x)
            sigma = torch.sqrt(var)
            yhat = mu + 1.0*sigma
            
        elif acquisition == 'DI-TS':
            yhat = self.sample_posterior(x, n_samples=1)
            
        elif acquisition == 'DI':
            yhat = y_range[0]*torch.ones(x.shape[0])
            
        else:
            raise NotImplementedError
        
        gate_e = (y_subopt >= y_best).float()
        
        if acquisition == 'DI-EI':
            I_e = yhat/(y_range[1] - y_range[0]) * (y_subopt/y_best) * gate_e
        else:
            I_e = (yhat - y_range[0])/(y_range[1] - y_range[0]) * (y_subopt/y_best) * gate_e
        
        # Discriminative improvement function
        f_KAN = self.m.formula
        zvars = sorted(f_KAN.free_symbols, key=lambda s: s.name)
        
        df = [sp.diff(f_KAN, zvar)**2 for zvar in zvars]  # measure for curvature
        df = [replace_half_powers(_df) for _df in df]
        zindices = [int(zvar.name.split('x')[1]) for zvar in zvars]
        z = x[:, zindices]
        Z = X[:, zindices]
        
        deltas, lambdas = eval_AGE_numeric(df, z, zvars, x_range, eps=eps, n_samples_i=10, n_samples_j=1000)
        Deltas, Lambdas = eval_AGE_numeric(df, Z, zvars, x_range, eps=eps, n_samples_i=10, n_samples_j=1000)
        indices = np.argsort(lambdas)
        
        if len(indices) > 2:  # three or more active descriptors: I_d compares the top two (the third is read but not used)
            j1 = indices[-1]  # best descriptor
            j2 = indices[-2]  # runner-up
            j3 = indices[-3]
            z1 = zvars[j1]
            z2 = zvars[j2]
            descriptors = [z1.name, z2.name]
            
            deltas = torch.tensor(deltas).float()
            delta1 = deltas[:, j1]
            delta2 = deltas[:, j2]
            
            Deltas = torch.tensor(Deltas).float()
            Delta1 = Deltas[:, j1]
            Delta2 = Deltas[:, j2]
            
            Lambdas = torch.tensor(Lambdas).float()
            Lambda1 = Lambdas[j1]
            Lambda2 = Lambdas[j2]
            Lambda3 = Lambdas[j3]
            
            gate_d = (delta1.max() > Delta1.max()).float()
            I_d = delta1/(delta1 + delta2 + eps) * gate_d

        elif len(indices) == 2:  # exactly two active descriptors: top-2 comparison, no third
            j1 = indices[-1]  # best descriptor
            j2 = indices[-2]  # runner-up
            z1 = zvars[j1]
            z2 = zvars[j2]
            descriptors = [z1.name, z2.name]

            deltas = torch.tensor(deltas).float()
            delta1 = deltas[:, j1]
            delta2 = deltas[:, j2]

            Deltas = torch.tensor(Deltas).float()
            Delta1 = Deltas[:, j1]
            Delta2 = Deltas[:, j2]

            Lambdas = torch.tensor(Lambdas).float()
            Lambda1 = Lambdas[j1]
            Lambda2 = Lambdas[j2]

            gate_d = (delta1.max() > Delta1.max()).float()
            I_d = delta1/(delta1 + delta2 + eps) * gate_d

        elif len(indices) == 1:
            j1 = indices[0]
            z1 = zvars[j1]
            descriptors = [z1.name, z1.name]
            
            I_d = I_e * 0  # no need to discriminate
        else:
            raise RuntimeError("unexpected behavior")
            
        # Uniform improvement function
        # Known issue (see README): j1 is a position in zvars (formula symbols sorted by name), not an input
        # column; the leading descriptor's column is zindices[j1]. KANDOE_IU_FIX=1 selects that column.
        # Unset keeps x[:, j1], the computation behind the results reported in the paper.
        if os.environ.get('KANDOE_IU_FIX', '0') == '1':
            z = x[:, zindices[j1]]
            Z = X[:, zindices[j1]]
        else:
            z = x[:, j1]
            Z = X[:, j1]
        d_min = torch.min(np.abs(z[:,None] - Z[None,:]), axis=1).values
        I_u = d_min/(x_range[1] - x_range[0])
        
        # Evaluate aquisition function
        a = I_e + I_d + I_u
        
        # Checker
        if torch.isnan(a).any():
            raise RuntimeError("unexpected behavior")
        # Build per-feature AGE map (full feature space)
        n_feat = x.shape[1]
        age_full = np.zeros(n_feat)
        for iloc, col in enumerate(zindices):
            if col < n_feat:
                age_full[col] = lambdas[iloc]

        return a, (I_e, I_d, I_u), descriptors, age_full, str(f_KAN)
    
    
    def eval_EI(self, x, y_max):
        mu, var = self.predict(x)
        sigma = torch.sqrt(var)
        
        diff = mu - y_max
        z = diff/sigma
        normal = Normal(0, 1)
        I_exploitation = diff * normal.cdf(z)
        I_exploration = sigma * normal.log_prob(z).exp()
        a = I_exploitation + I_exploration
        a[sigma <= 0.] = 0.
        
        return a, (I_exploitation, I_exploration)
    
    
    def eval_acquisition(self, x, y_max, verbose=True):
        """
        NOTE: Returning descriptors should be evaluated using self.X, not x.
        """
        acquisition = self.acquisition
        
        if acquisition == 'EI':
            a, I = self.eval_EI(x, y_max)
            I_exploitation, I_exploration = I
            
            I = torch.vstack((I_exploitation, I_exploration)).T
            I_labels = ['$I_{exploit}$', '$I_{explore}$']
            
            descriptors = None
            return a, I, I_labels, descriptors
            
        elif acquisition == 'UCB':
            mu, var = self.predict(x)
            sigma = torch.sqrt(var)
            kappa = 2.576
            a = mu + kappa * sigma
            I_exploitation = mu
            I_exploration = kappa * sigma

            I = torch.vstack((I_exploitation, I_exploration)).T
            I_labels = ['$I_{exploit}$', '$I_{explore}$']
            
            descriptors = None
            return a, I, I_labels, descriptors
        
        elif acquisition == 'TS':
            yhat = self.sample_posterior(x, n_samples=1)  # return (x.shape[0], n_samples)
            a = I_exploitation = yhat
            
            I = I_exploitation.reshape(-1, 1)
            I_labels = ['$I_{exploit}$']
            
            descriptors = None
            return a, I, I_labels, descriptors
    
        elif 'DI' in acquisition:  # descriptor identification
            a, I, descriptors, age_lambdas, kan_formula = self.eval_DI(x, y_max, acquisition)
            I_e, I_d, I_u = I

            I = torch.vstack((I_e, I_d, I_u)).T
            I_labels = ['$I_{e}$', '$I_{d}$', '$I_{u}$']
            return a, I, I_labels, descriptors, age_lambdas, kan_formula
            
        else:
            raise NotImplementedError
    
    
    def update_priors(self):
        self.prior_alpha['mode'] = self.model.cov.outputscale.detach()
        self.prior_length['mode'] = self.model.cov.base_kernel.lengthscale.detach()[0]
        self.prior_sigma['mode'] = self.model.likelihood.noise.detach()[0]
        return
    
    
    def virtual_experiment(self, idx):
        df = self.df
        return get_Y(df, tensor=False)[idx]
    
    
    def monitor_progress(self, fig, axes, showfig=True, savefig=True):
        # Parse data
        X = get_X(self.df, tensor=True)
        Y = get_Y(self.df, tensor=False)
        Z = get_Z(self.df, tensor=False)
        x_range = self.x_range
        y_range = self.y_range
        y_max = self.y_max
        
        # Evaluate model prediction
        Y_mean, Y_cov = self.predict(X, tensor=False)
        
        # Evaluate acquisition function values
        A, _, I_labels, descriptors = self.eval_acquisition(X, y_max, verbose=False)  # use full record
        
        # Tensor to numpy array
        X = detach(X)
        A = detach(A)
        
        # Get indices
        indices = self.indices
        indices_initial = self.indices_initial
        indices_sampled = self.indices_sampled
        indices_unsampled = list(set(indices) - set(indices_sampled))
        indices_y_subopt = self.indices_y_subopt
        idx_next = self.idx_next
        idx_opt = self.idx_opt
        idx_max = self.idx_max
        zlabel = self.zlabel
        
        # Plot acquisition function
        plot_2D_contour(fig, axes[0,0], Z, Y_mean, indices_initial, 
                        indices_sampled, indices_unsampled, indices_y_subopt,
                        idx_next, idx_max, idx_opt, xlabel=zlabel, ylabel=r'$\hat{Y}$')
        
        # Plot objective surface
        plot_2D_contour(fig, axes[1,0], Z, A, indices_initial, 
                        indices_sampled, indices_unsampled, indices_y_subopt,
                        idx_next, idx_max, idx_opt, xlabel=zlabel, ylabel='a(x)', legend=False)
        
        # Plot regressor accuracy
        plot_regressor_accuracy(axes[0,1], Y, Y_mean, indices_initial, 
                                indices_sampled, indices_unsampled, indices_y_subopt,
                                idx_next, idx_max, idx_opt, x_range=x_range, y_range=y_range,
                                xlabel='$Y(x)$', ylabel=r'$\mu(x)$', legend=False)
        
        # Plot improvement function
        K = self.log.index.values
        I = np.vstack(self.log['I_next'].values)
        colors = np.vstack(list(CMAP_IMPROVEMENT.values()))
        plot_trajectory(axes[1,1], K, I, colors=colors, xlabel='Iterations', ylabel='$I(x_{next})$', labels=I_labels)
        
        # Plot distances
        d_xnext_xopt = np.vstack(self.log['d_xnext_xopt'].values)
        d_xnext_xprev = np.vstack(self.log['d_xnext_xprev'].values)
        
        plot_trajectory(axes[0,2], K, np.hstack((d_xnext_xopt, d_xnext_xprev)), 
                        xlabel='Iterations', ylabel='Distance', 
                        labels=['d($x_{next}$, $x^*$)', 'd($x_{next}$, $x_{prev}$)'])
        
        if self.acquisition in ['EI', 'UCB', 'TS']:
            # Plot kernel parameters
            noise = np.vstack(self.log['noise'].values)
            outputscale = np.vstack(self.log['outputscale'].values)
            lengthscale = np.vstack(self.log['lengthscale'].values)
            length_scale_labels = [rf'$\ell_{i}$' for i in range(lengthscale.shape[-1])]
            
            plot_trajectory(axes[1,2], K, np.hstack((lengthscale, outputscale, noise)), 
                            xlabel='Iterations', ylabel='Measures', 
                            labels=[*length_scale_labels, r'$\sigma^{2}_{f}$', r'$\sigma^{2}_{n}$'])
            
        elif 'DI' in self.acquisition:
            # Plot XY (x = descriptors)
            if len(descriptors) == 2:
                pass
            elif len(descriptors) == 1:
                descriptors = [descriptors[0], descriptors[0]]
            else:
                raise RuntimeError("unexpected behavior")
            
            xvar1, xvar2 = descriptors[0], descriptors[1]
            idx1, idx2 = int(xvar1.split('x')[1]), int(xvar2.split('x')[1])
            X1 = X[:,idx1].squeeze()
            X2 = X[:,idx2].squeeze()
            xvarname1 = xvar1[0] + str(idx1 + 1)  # 1-indexing
            xvarname2 = xvar2[0] + str(idx2 + 1)
            xlabel1 = 'Best descriptor (' + xvarname1 + ')'
            xlabel2 = 'Runner up (' + xvarname2 + ')'
            plot_XY(axes[1,2], X1, Y, indices_initial, indices_sampled, indices_unsampled, indices_y_subopt,
                    idx_next, idx_max, idx_opt, x_range=x_range, y_range=y_range, 
                    xlabel=xlabel1, ylabel='Y(x)', legend=False)
            plot_XY(axes[2,2], X2, Y, indices_initial, indices_sampled, indices_unsampled, indices_y_subopt,
                    idx_next, idx_max, idx_opt, x_range=x_range, y_range=y_range, 
                    xlabel=xlabel2, ylabel='Y(x)', legend=False)
            
            # Plot selected descriptors
            K = self.log.index.values
            D = np.vstack(self.log['descriptors'])
            plot_descriptor_transition(axes[2,0], K, D, I, xlabel='Iterations', xvars=self.xvars)
            plot_descriptor_distribution(axes[2,1], K, D, xvars=self.xvars)
        
        if showfig:
            plt.pause(0.1)
        
        if savefig:
            filedir =  os.path.join(self.imgoutdir, str(K[-1]) + '.svg')
            fig.savefig(filedir)
        return
