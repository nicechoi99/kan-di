import os
import warnings

import torch
from torch.optim import Adam
import numpy as np
import sympy as sp
import pandas as pd
from scipy.stats import gamma
from matplotlib.colors import to_rgba
from kan.custom import replace_half_powers
from sympy.functions.special.bsplines import bspline_basis_set

from datamanager import load_dataset, get_X, get_Y
from models import KANMean, ZeroMean, GammaPrior, GaussianLikelihood, ExactGPModel, GreaterThan, \
    ExactMarginalLogLikelihood, MultivariateNormal, cholesky_jitter, cholesky_max_tries, \
    eval_average_gradient_energy_numeric, NumericalWarning
from plotter import plot_data, spline_regression
from utils import plt, cprint, checkexists, datadir, imgdir, MaxNLocator, NullLocator


def gen_demo_data(n=300, biased=False):
    """
    Generate synthetic demo data for testing.

    Args:
    - n: Number of samples to generate.
    - biased: If True, sample 2/3 of data from the top 10% of y-values and 1/3 from the rest,
              with x2 domain biased toward smaller values.

    Returns:
    - x1, x2, y: Arrays of input and output variables.
    """
    np.random.seed(0)
    x1 = np.random.uniform(0.1, 5.0, n)
    # base distribution for x2 (uniform small domain)
    x2 = np.random.uniform(0.0, 0.1, n)
    y = gamma.pdf(x1, a=2.0, scale=1.0) + np.random.randn(n) * x2

    if biased:
        # Determine top 10% threshold
        threshold = np.percentile(y, 90)
        idx_top = np.where(y >= threshold)[0]
        idx_rest = np.where(y < threshold)[0]

        n_top = int(2 * n / 3)
        n_rest = n - n_top

        # Sample indices
        top_sample = np.random.choice(idx_top, size=n_top, replace=len(idx_top) < n_top)
        rest_sample = np.random.choice(idx_rest, size=n_rest, replace=len(idx_rest) < n_rest)

        idx_selected = np.concatenate([top_sample, rest_sample])
        np.random.shuffle(idx_selected)

        # Re-sample x2 with bias toward smaller values (e.g., quadratic bias)
        # This makes the x2 distribution right-skewed (more dense near 0)
        x2_biased = np.random.beta(a=1.0, b=5.0, size=n) * 0.1  # skewed toward 0

        # Apply selection
        x1, y = x1[idx_selected], y[idx_selected]
        x2 = x2_biased

    return x1, x2, y


def plot_demo_data(n=300, biased=False, elev=10, azim=-74, var=None, save=True, show=True, filename='demo_data.svg'):
    x1, x2, y = gen_demo_data(n=n, biased=biased)

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, x2, y, c=y, cmap='viridis', s=20, alpha=0.8)

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(4))

    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.zaxis.set_minor_locator(NullLocator())

    ax.grid(True)

    ax.xaxis._axinfo['grid']['color'] = (0, 0, 0, 0.1)
    ax.yaxis._axinfo['grid']['color'] = (0, 0, 0, 0.1)
    ax.zaxis._axinfo['grid']['color'] = (0, 0, 0, 0.1)
    ax.xaxis.pane.set_facecolor((0.9, 0.9, 0.9, 1))
    ax.yaxis.pane.set_facecolor((0.5, 0.5, 0.5, 1))
    ax.zaxis.pane.set_facecolor((0.3, 0.3, 0.3, 1))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((0, 0, 0, 0))  # make the axis line fully transparent

    ax.tick_params(axis='x', which='both', length=0, width=0, color='w', labelbottom=False)
    ax.tick_params(axis='y', which='both', length=0, width=0, color='w', labelleft=False)
    ax.tick_params(axis='z', which='both', length=0, width=0, color='w', labelleft=False)

    ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()

    if save:
        if var is None:
            filedir = os.path.join(imgdir, filename)
        else:
            filedir = os.path.join(imgdir, filename.replace('.', '_' + var + '.'))
        fig.savefig(filedir)

    if show:
        plt.show()
    return fig, ax


def plot_demo_data_with_zero_prior(n=30, save=True, show=True):
    x1, x2, y = gen_demo_data(n=n)
    fig, ax = plot_demo_data(n=n, save=False, show=False)

    # Overlay a plane
    x1g, x2g = np.meshgrid(np.linspace(x1.min(), x1.max(), 30),
                        np.linspace(x2.min(), x2.max(), 30))
    y0 = np.mean(y)
    y_plane = np.full_like(x1g, y0)
    
    ax.plot_surface(x1g, x2g, y_plane, color='k', alpha=0.8, linewidth=0)
    ax.plot3D([x1.min(), x1.max(), x1.max(), x1.min(), x1.min()],
              [x2.min(), x2.min(), x2.max(), x2.max(), x2.min()],
              [y0, y0, y0, y0, y0], color='black', lw=2.0)

    fig.tight_layout()

    if save:
        filedir = os.path.join(imgdir, 'demo_data_with_plane.svg')
        fig.savefig(filedir)

    if show:
        plt.show()
    return fig, ax


def plot_demo_data_with_KAN(n=30, save=True, show=True, var='x1', filename='demo_data_with_KAN.svg'):
    x1, x2, y = gen_demo_data(n=n)
    fig, ax = plot_demo_data(n=n, save=False, show=False)

    if var == 'x1':
        z = x1
        elev = 0
        azim = -90
    elif var == 'x2':
        z = x2
        elev = 0
        azim = 0
    else:
        raise RuntimeError('Unexpected behavior')

    ax.view_init(elev=elev, azim=azim)

    spline = spline_regression(z, y, show=False)
    xhat = np.linspace(np.min(z), np.max(z), 400)
    yhat = spline(xhat)

    if var == 'x1':
        ax.plot(xhat, -0.01*np.ones_like(xhat), yhat, color='k', lw=3.0, zorder=10)
    elif var == 'x2':
        ax.plot(-0.01*np.ones_like(xhat), xhat, yhat, color='k', alpha=0.8, lw=2.0, zorder=10)
    else:
        raise RuntimeError('Unexpected behavior')

    ax.tick_params(direction='in', length=3, width=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        filedir = os.path.join(imgdir, filename.replace('KAN', 'KAN_' + var))
        fig.savefig(filedir)

    if show:
        plt.show()
    return fig, ax


def gen_benchmark_functions():
    """
    Construct a set of symbolic benchmark functions for Bayesian optimization.

    Returns:
    - benchmarks: Dictionary of benchmark problems with SymPy expressions and domain bounds.
    """
    # Common symbol definitions (up to 6D)
    x1, x2, x3, x4, x5, x6 = sp.symbols('x1 x2 x3 x4 x5 x6', real=True)

    # -------------------------------------------------------------------------
    # Rastrigin (5D scalable)
    # -------------------------------------------------------------------------
    x = sp.symbols('x1 x2 x3 x4 x5', real=True)
    A = 10
    rastrigin_expr = A * 5 + sum(xi**2 - A * sp.cos(2 * sp.pi * xi) for xi in x)

    # -------------------------------------------------------------------------
    # Hartmann-6
    # -------------------------------------------------------------------------
    alpha = [1.0, 1.2, 3.0, 3.2]
    A_mat = [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ]
    P = 1e-4 * sp.Matrix([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    hartmann_expr = -sum(
        alpha[k] * sp.exp(-sum(A_mat[k][j] * (sp.Symbol(f'x{j+1}') - P[k, j])**2 for j in range(6)))
        for k in range(4)
    )

    # -------------------------------------------------------------------------
    # Griewank (5D)
    # -------------------------------------------------------------------------
    sum_term = sum(xi**2 for xi in x) / 4000
    prod_term = sp.prod(sp.cos(x[i] / sp.sqrt(i + 1)) for i in range(5))
    griewank_expr = sum_term - prod_term + 1

    # -------------------------------------------------------------------------
    # Rosenbrock (5D)
    # -------------------------------------------------------------------------
    rosen_expr = sum(100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(4))

    # -------------------------------------------------------------------------
    # Ackley (5D scalable)
    # -------------------------------------------------------------------------
    a, b, c = 20, 0.2, 2 * sp.pi
    n = 5
    sum_sq = sum(xi**2 for xi in x)
    sum_cos = sum(sp.cos(c * xi) for xi in x)
    ackley_expr = -a * sp.exp(-b * sp.sqrt(sum_sq / n)) - sp.exp(sum_cos / n) + a + sp.E

    # -------------------------------------------------------------------------
    # Collect into dictionary
    # -------------------------------------------------------------------------
    benchmarks = {
        'rastrigin': {
            'expr': rastrigin_expr,
            'domain': {
                'x1': (-5.12, 5.12),
                'x2': (-5.12, 5.12),
                'x3': (-5.12, 5.12),
                'x4': (-5.12, 5.12),
                'x5': (-5.12, 5.12)
            }
        },

        'hartmann6': {
            'expr': hartmann_expr,
            'domain': {
                'x1': (0.0, 1.0),
                'x2': (0.0, 1.0),
                'x3': (0.0, 1.0),
                'x4': (0.0, 1.0),
                'x5': (0.0, 1.0),
                'x6': (0.0, 1.0)
            }
        },

        'griewank': {
            'expr': griewank_expr,
            'domain': {
                'x1': (-600.0, 600.0),
                'x2': (-600.0, 600.0),
                'x3': (-600.0, 600.0),
                'x4': (-600.0, 600.0),
                'x5': (-600.0, 600.0)
            }
        },

        'rosenbrock': {
            'expr': rosen_expr,
            'domain': {
                'x1': (-2.0, 2.0),
                'x2': (-2.0, 2.0),
                'x3': (-2.0, 2.0),
                'x4': (-2.0, 2.0),
                'x5': (-2.0, 2.0)
            }
        },

        'ackley': {
            'expr': ackley_expr,
            'domain': {
                'x1': (-32.768, 32.768),
                'x2': (-32.768, 32.768),
                'x3': (-32.768, 32.768),
                'x4': (-32.768, 32.768),
                'x5': (-32.768, 32.768)
            }
        }
    }

    return benchmarks


def add_gradient(ax, bars, base_color=(191/255, 191/255, 191/255), highlight_color=(38/255, 166/255, 119/255),
                 threshold=0.6, base_alpha=0.35):
    """
    Overlay vertical alpha gradient per bar without breaking xlim/xticks.
    Bars with height > threshold are highlighted with a separate gradient color.
    The gradient is darker at the top and lighter at the bottom.

    Args:
    - ax: Matplotlib Axes object.
    - bars: BarContainer returned by ax.bar.
    - base_color: Base color for bars with height <= threshold.
    - highlight_color: RGB tuple for bars with height > threshold.
    - threshold: Height cutoff above which highlight_color is used.
    - base_alpha: Base transparency for the bar face before gradient overlay.

    Returns:
    - None
    """
    rgb_base = to_rgba(base_color)
    rgb_high = to_rgba(highlight_color)

    for rect in bars:
        h = rect.get_height()
        color = rgb_high if h > threshold else rgb_base
        rect.set_facecolor((color[0], color[1], color[2], base_alpha))

    xlim0 = ax.get_xlim()
    ylim0 = ax.get_ylim()

    for rect in bars:
        x, y = rect.get_x(), rect.get_y()
        w, h = rect.get_width(), rect.get_height()
        if h <= 0:
            continue

        rgb = rgb_high if h > threshold else rgb_base

        # top darker → reverse gradient direction
        grad = np.linspace(0, 1, 256).reshape(-1, 1)
        rgba = np.ones((256, 1, 4), dtype=float)
        rgba[..., :3] = rgb[:3]
        rgba[..., 3] = grad

        im = ax.imshow(
            rgba, extent=[x, x + w, y, y + h], aspect='auto', origin='lower',
            interpolation='bilinear', zorder=rect.get_zorder() + 1,
            transform=ax.transData, clip_on=True
        )
        im.set_clip_path(rect)

    ax.set_xlim(xlim0)
    ax.set_ylim(ylim0)
    return


def plot_benchmark_age(save=True, show=True):
    benchmarks = gen_benchmark_functions()
    
    ncols = len(benchmarks)
    fig, axes = plt.subplots(1, ncols, figsize=(2*ncols, 2))

    if ncols == 1:
        axes = [axes]

    for ax, (name, problem) in zip(axes, benchmarks.items()):
        f = problem['expr']
        domain = problem['domain']

        xvars = sorted(f.free_symbols, key=lambda s: s.name)
        df = [sp.diff(f, x)**2 for x in xvars]
        df = [replace_half_powers(_df) for _df in df]

        age = eval_average_gradient_energy_numeric(df, xvars,
                                                   feature_range=list(domain.values()))
        age_norm = age/age.max()

        xnames = [x.name for x in xvars]
        order = np.argsort(-age_norm)
        xsorted = np.array(xnames)[order]
        ysorted = age_norm[order]

        bars = ax.bar(xsorted, ysorted, color='none', edgecolor='black', linewidth=0.8, zorder=2)

        add_gradient(ax, bars)

        # remove minor ticks + increase label font
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='x', which='minor', length=0)

        ax.set_title(name.capitalize(), fontweight='bold', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)

    if save:
        filedir = os.path.join(imgdir, 'benchmark_AGE_distribution.svg')
        plt.savefig(filedir)

    if show:
        plt.show()

    return


def plot_small_feature_dataset_age(save=True, show=True):
    """
    Plot normalized AGE distributions for small-feature datasets in a single-row subplot layout.

    Args:
    - save:  Whether to save the combined AGE figure.
    - show:  Whether to display the figure.

    Returns:
    - None.
    """
    dataset_name = 'small_feature'
    datasets = load_dataset(dataset_name)

    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(2*ncols, 2))

    if ncols == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, datasets.items()):
        X = get_X(df)
        y = get_Y(df)

        d = X.shape[1]
        xvars = sp.symbols(' '.join([f'x{i+1}' for i in range(d)]), real=True)
        if d == 1:
            xvars = (xvars,)

        splines = []
        for i in range(d):
            xi = X[:, i]
            spline = spline_regression(xi, y, show=False)
            splines.append(spline)

        exprs = []
        bounds = []
        for i in range(d):
            expr_i = scipy_spline_to_sympy(splines[i], xvars[i], as_piecewise=True)
            exprs.append(expr_i)

            xmin, xmax = np.nanmin(X[:, i]), np.nanmax(X[:, i])
            if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
                xmin, xmax = 0.0, 1.0
            bounds.append((float(xmin), float(xmax)))

        df_list = [sp.diff(exprs[i], xvars[i])**2 for i in range(d)]
        df_list = [replace_half_powers(e) for e in df_list]

        age = eval_average_gradient_energy_numeric(df_list, list(xvars), feature_range=bounds)
        age_norm = age/age.max()

        xnames = np.array([str(v) for v in xvars])
        order = np.argsort(-age_norm)
        xsorted = xnames[order]
        ysorted = age_norm[order]

        bars = ax.bar(xsorted, ysorted, color='none', edgecolor='black',
                      linewidth=0.8, zorder=2)

        add_gradient(ax, bars, highlight_color=(27/255, 93/255, 69/255))

        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='x', which='minor', length=0)

        ax.set_title(name, fontweight='bold', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)

    if save:
        filedir = os.path.join(imgdir, f'{dataset_name}_AGE_distribution.svg')
        plt.savefig(filedir)

    if show:
        plt.show()

    return


def scipy_spline_to_sympy(spline, x, as_piecewise=True):
    """
    Convert a SciPy UnivariateSpline/LSQUnivariateSpline to a SymPy expression.

    Args:
    - spline: SciPy spline object.
    - x: SymPy symbol for the variable.
    - as_piecewise: Whether to rewrite the result as Piecewise.

    Returns:
    - expr: SymPy expression equivalent to the SciPy spline.
    """
    tck = getattr(spline, '_eval_args', None)
    if tck is None and hasattr(spline, 'tck'):
        tck = spline.tck
    if tck is None:
        raise TypeError('Unsupported spline type: cannot extract (t, c, k).')

    t, c, k = tck
    t = sp.Tuple(*[sp.Float(v) for v in np.asarray(t, float).tolist()])
    c = [sp.Float(v) for v in np.asarray(c, float).tolist()]
    k = int(k)

    # Compute valid number of bases and align coefficients
    n_bases = len(t) - k - 1
    if n_bases <= 0:
        raise ValueError('Invalid knot/degree configuration: len(t) - k - 1 <= 0.')

    # Basis set length is guaranteed == n_bases
    basis = bspline_basis_set(k, t, x)
    if len(basis) != n_bases:
        raise ValueError('Unexpected basis size mismatch.')

    # Align coefficient length to n_bases (truncate or pad with zeros if needed)
    if len(c) < n_bases:
        c = c + [sp.Float(0.0)] * (n_bases - len(c))
    elif len(c) > n_bases:
        c = c[:n_bases]

    expr = sp.Add(*[ci * bi for ci, bi in zip(c, basis)])

    if as_piecewise:
        expr = sp.simplify(expr.rewrite(sp.Piecewise))  # takes long time

    return expr


def forward(model, likelihood, x):
    warnings.filterwarnings("ignore", category=NumericalWarning)
    last_err = None
    for jitter in (1e-6, 1e-5, 1e-4, 1e-3):
        try:
            with cholesky_jitter(jitter), cholesky_max_tries(5):
                mu_and_sigma = model(x)
                yhat = likelihood(mu_and_sigma)
            break
        except NotPSDError as e:
            last_err = e
            continue
    else:
        raise last_err

    return yhat


def plot_demo_data_with_posterior(n=30, save=True, show=True):
    x1, x2, y = gen_demo_data(n=n)
    X = np.vstack((x1, x2)).T

    X = torch.tensor(X)
    y = torch.tensor(y)

    # Generate Gaussian process model
    prior_sigma = {'param': GammaPrior(0.5, 5.0), 'mode': 0.1}
    sigma_min = 1e-5
    likelihood = GaussianLikelihood(noise_prior=prior_sigma['param'], noise_constraint=GreaterThan(sigma_min))
    likelihood.noise = torch.tensor([float(prior_sigma['mode'])])
    
    nu = 5/2
    prior_length = {'param': GammaPrior(2.0, 1.0), 'mode': 2.0}
    prior_alpha = {'param': GammaPrior(2.0, 0.5), 'mode': 2.0}
    m = ZeroMean()
    model = ExactGPModel(X, y, likelihood=likelihood, nu=nu, prior_length=prior_length, prior_alpha=prior_alpha, mean=m)
    
    # Train model
    model.train()
    likelihood.train()
    
    lr = 0.1
    n_iters = 100
    verbose = True
    optimizer = Adam(model.parameters(), lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood=likelihood, model=model)
    
    # Fit hyperparameters
    cprint('Updating parameters...')
    for i in range(n_iters):
        optimizer.zero_grad()
        yhat = forward(model, likelihood, X)
        
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
        noise = np.round(model.likelihood.noise.item(), 3)
        outputscale = np.round(model.cov.outputscale.item(), 2)
        lengthscale = np.round(model.cov.base_kernel.lengthscale.squeeze().tolist(), 2)
        
        if verbose and (i+1) % (n_iters/10) == 0:  # check convergence
            cprint('    Iter', i+1, '/', n_iters, '| loss:', loss, '| noise:', noise, '| outputscale:', outputscale, '| lengthscale:', lengthscale, color='w', inspect=False)
        
        optimizer.step()
    
    model.eval()
    likelihood.eval()
    
    # Plot posterior mean over the demo data (3D)
    fig1, ax1 = plot_demo_data(n=n, var='x1', save=False, show=False)

    # Build a grid on the x1-x2 domain
    n_grid = 60
    x1_lin = np.linspace(np.min(x1), np.max(x1), n_grid)
    x2_lin = np.linspace(np.min(x2), np.max(x2), n_grid)
    x1g, x2g = np.meshgrid(x1_lin, x2_lin)
    Xg = np.column_stack([x1g.ravel(), x2g.ravel()])
    Xg_t = torch.tensor(Xg)

    # Predict posterior on the grid
    with torch.no_grad():
        post = forward(model, likelihood, Xg_t)
        mean_g = post.mean.detach().cpu().numpy()
        var_g = post.variance.detach().cpu().numpy()

    mean_surf = mean_g.reshape(n_grid, n_grid)

    # Overlay posterior mean surface
    ax1.plot_surface(x1g, x2g, mean_surf, color='k', linewidth=3.0, alpha=0.8)

    fig1.tight_layout()

    # Plot posterior distribution: mean ± 2·std (3D)
    fig2, ax2 = plot_demo_data(n=n, var='x1', save=False, show=False)

    # Plot mean, upper, and lower surfaces
    std_g = np.sqrt(np.maximum(var_g, 0.0)).reshape(n_grid, n_grid)
    up_surf = mean_surf + 2.0 * std_g
    lo_surf = mean_surf - 2.0 * std_g

    # Mean surface
    ax2.plot_surface(x1g, x2g, mean_surf, color='k', linewidth=3.0, alpha=0.8)

    # Upper and lower credible surfaces
    ax2.plot_surface(x1g, x2g, up_surf, color='tomato', linewidth=0.4, alpha=0.8)
    ax2.plot_surface(x1g, x2g, lo_surf, color='tomato', linewidth=0.4, alpha=0.8)

    fig2.tight_layout()

    if save:
        filedir = os.path.join(imgdir, 'demo_data_with_posterior_mean.svg')
        fig1.savefig(filedir)
        
        filedir = os.path.join(imgdir, 'demo_data_with_posterior_distribution.svg')
        fig2.savefig(filedir)

    if show:
        plt.show()
    return



def plot_demo_data_with_KAN_posterior(n=30, save=True, show=True):
    x1, x2, y = gen_demo_data(n=n)
    X = np.vstack((x1, x2)).T

    X = torch.tensor(X)
    y = torch.tensor(y)
    dataset = {}
    dataset['train_input'] = dataset['test_input'] = X.float()
    dataset['train_label'] = dataset['test_label'] = y.float().reshape(-1, 1)
    feature_range = (X.min(), X.max())

    # Generate Gaussian process model
    prior_sigma = {'param': GammaPrior(0.5, 5.0), 'mode': 0.1}
    sigma_min = 1e-5
    likelihood = GaussianLikelihood(noise_prior=prior_sigma['param'], noise_constraint=GreaterThan(sigma_min))
    likelihood.noise = torch.tensor([float(prior_sigma['mode'])])
    
    nu = 5/2
    prior_length = {'param': GammaPrior(2.0, 1.0), 'mode': 2.0}
    prior_alpha = {'param': GammaPrior(2.0, 0.5), 'mode': 2.0}
    
    m = KANMean()
    m.train()
    m.fit(dataset, feature_range=feature_range, verbose=True)
    
    model = ExactGPModel(X, y, likelihood=likelihood, nu=nu, prior_length=prior_length, prior_alpha=prior_alpha, mean=m)
    
    # Train model
    model.train()
    likelihood.train()
    
    lr = 0.1
    n_iters = 100
    verbose = True
    optimizer = Adam(model.parameters(), lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood=likelihood, model=model)
    
    # Fit hyperparameters
    cprint('Updating parameters...')
    for i in range(n_iters):
        optimizer.zero_grad()
        yhat = forward(model, likelihood, X)
        
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
        noise = np.round(model.likelihood.noise.item(), 3)
        outputscale = np.round(model.cov.outputscale.item(), 2)
        lengthscale = np.round(model.cov.base_kernel.lengthscale.squeeze().tolist(), 2)
        
        if verbose and (i+1) % (n_iters/10) == 0:  # check convergence
            cprint('    Iter', i+1, '/', n_iters, '| loss:', loss, '| noise:', noise, '| outputscale:', outputscale, '| lengthscale:', lengthscale, color='w', inspect=False)
        
        optimizer.step()
    
    model.eval()
    likelihood.eval()
    
    # Plot posterior mean over the demo data (3D)
    fig, ax = plot_demo_data(n=n, var='x1', save=False, show=False)

    # Build a grid on the x1-x2 domain
    n_grid = 60
    x1_lin = np.linspace(np.min(x1), np.max(x1), n_grid)
    x2_lin = np.linspace(np.min(x2), np.max(x2), n_grid)
    x1g, x2g = np.meshgrid(x1_lin, x2_lin)
    Xg = np.column_stack([x1g.ravel(), x2g.ravel()])
    Xg_t = torch.tensor(Xg)

    # Predict posterior on the grid
    with torch.no_grad():
        post = forward(model, likelihood, Xg_t)
        mean_g = post.mean.detach().cpu().numpy()
        var_g = post.variance.detach().cpu().numpy()

    mean_surf = mean_g.reshape(n_grid, n_grid)

    # Overlay posterior mean surface
    ax.plot_surface(x1g, x2g, mean_surf, color='k', linewidth=3.0, alpha=0.8)

    fig.tight_layout()

    if save:
        filedir = os.path.join(imgdir, 'demo_data_with_KAN_posterior_mean.svg')
        fig.savefig(filedir)

    if show:
        plt.show()
    return


if __name__ == '__main__':
    # plot_demo_data(n=300)
    # plot_demo_data(n=300, elev=3, azim=-88, var='x1')
    # plot_demo_data(n=300, elev=0, azim=0, var='x2')
    
    # plot_benchmark_age()
    plot_small_feature_dataset_age()
    
    n = 30
    # plot_demo_data(n=n, filename='demo_data_scarce.svg')
    # plot_demo_data_with_zero_prior(n=n)
    plot_demo_data_with_KAN(n=n, var='x1')
    plot_demo_data_with_KAN(n=n, var='x2')
    # plot_demo_data_with_posterior(n=n)
    # plot_demo_data_with_KAN_posterior(n=n)

    # plot_demo_data(n=30, elev=3, azim=-88, biased=True, filename='demo_data_biased_x1.svg')
    # plot_demo_data(n=30, elev=0, azim=0, biased=True, filename='demo_data_biased_x2.svg')
    # plot_demo_data(n=30, elev=3, azim=-88, biased=False, filename='demo_data_unbiased_x1.svg')
    # plot_demo_data(n=30, elev=0, azim=0, biased=False, filename='demo_data_unbiased_x2.svg')
    a = 1
    