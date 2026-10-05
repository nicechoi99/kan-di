import os
from copy import deepcopy

import numpy as np
import sympy as sp
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata, Rbf, RBFInterpolator, UnivariateSpline, interp1d

from config import DATASET, DESCRIPTOR_INDICES, ALIAS
from utils import datadir, imgdir, plt, get_carray, set_colorbar, cprint, to_rgb, \
    MaxNLocator, ListedColormap, BoundaryNorm
from datamanager import load_dataset, get_X, get_Y


legend_order = ['Unsampled', 'Initial', 'Sampled', 'Current best', 'Next query', 'Optimal']


CMAP_IMPROVEMENT = {
    'exploration': [0.7, 0.7, 0.7],
    'discrimination': [0.15, 0.39, 0.30],
    'uniform': [0.138, 0.66, 0.48]
}


def spline_regression(x, y, smooth=0.7, degree=3, show=True):
    """Fit a smoothing spline after averaging duplicate x values."""
    # Convert to arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Average duplicates efficiently
    ux, inv = np.unique(x, return_inverse=True)
    ysum = np.bincount(inv, weights=y)
    ycnt = np.bincount(inv)
    uy = ysum / np.maximum(ycnt, 1)

    # Sort by x (np.unique already returns sorted ux)
    m = ux.size

    # Handle small-m cases safely
    if m == 0:
        raise ValueError('Empty input.')
    if m == 1:
        c = float(uy[0])

        def const_fun(t):
            # returns constant array with the same shape as t
            tt = np.asarray(t, dtype=float)
            return np.full_like(tt, c, dtype=float)

        if show:
            plt.scatter(x, y, s=15, alpha=0.6, label='raw')
            plt.scatter(ux, uy, s=40, label='avg')
            plt.axhline(c, lw=2, label='constant fit')
            plt.legend()
            plt.show()
        return const_fun

    if m == 2:
        spline = interp1d(ux, uy, kind='linear', fill_value='extrapolate', assume_sorted=True)
        if show:
            plt.scatter(x, y, s=15, alpha=0.6, label='raw')
            plt.scatter(ux, uy, s=40, label='avg')
            xs = np.linspace(ux[0], ux[-1], 200)
            plt.plot(xs, spline(xs), lw=2, label='linear fit')
            plt.legend()
            plt.show()
        return spline

    # m >= 3: choose valid spline degree k <= m-1 and within [1, 5]
    k = int(max(1, min(degree, m - 1)))

    # Fit smoothing spline
    s = float(smooth) * m
    spline = UnivariateSpline(ux, uy, s=s, k=k)

    if show:
        plt.scatter(x, y, s=15, color='gray', alpha=0.6, label='raw')
        plt.scatter(ux, uy, s=35, label='avg (by x)')
        xs = np.linspace(ux[0], ux[-1], 400)
        plt.plot(xs, spline(xs), lw=2, label=f'spline fit (k={k}, s={s:.2f})')
        plt.legend()
        plt.show()

    return spline


def plot_y_distributions(dataset_name):
    datasets = load_dataset(dataset_name)
    
    for name, df in datasets.items():
        y = get_Y(df, tensor=False)
        fig = plot_y_distribution(y, title=name, draw=False)

        filedir = os.path.join(imgdir, dataset_name, 'y_distribution_' + name + '.svg')
        fig.savefig(filedir)
    
    plt.show()
    return


def plot_y_distribution(y, title=None, ratio=0.5, draw=True):
    if y.ndim == 1:
        pass
    elif y.ndim == 2 and y.shape[-1] == 1:
        y = y.flatten()
    else:
        raise RuntimeError('Not allowed!')
        
    y_span = y.max() - y.min()
    y_lower = y.min() + y_span * ratio
    y_subopt = y.min() + 0.9 * y_span

    # Get counts
    n_counts_below_halfspan = np.sum(y <= (y_lower))

    kde = gaussian_kde(y.flatten())
    y_grid = np.linspace(y.min(), y.max(), 1000)
    pdf = kde(y_grid)

    # Plot
    fig, ax = plt.subplots(1, 1)

    ax.hist(y, bins=30, density=True, alpha=0.3, label='histogram')
    ax.plot(y_grid, pdf, 'r-', lw=2, label='KDE')
    ax.axvline(y_lower, color='k', ls='--',  label='$y_{lower}$')
    ax.axvline(y_subopt, color='g', ls='--',  label='$y^{*}_{\\alpha}$')
    ax.text(0.05, 0.45, 'Total counts = ' + str(len(y)),
            transform=plt.gca().transAxes, ha='left', va='top', fontsize=8, fontweight='bold')
    ax.text(0.05, 0.4, 'Count(y $\leq y_{lower}$) = ' + str(n_counts_below_halfspan),
            transform=plt.gca().transAxes, ha='left', va='top', fontsize=8)
    ax.set_xlabel('y')
    ax.set_ylabel('Density')
    ax.legend(frameon=False)
    if title:
        title = ALIAS.get(title, title)
        ax.set_title(title, fontsize=10, fontweight='bold')
        fig.subplots_adjust(top=0.9)
    
    if draw:
        plt.show()
    return fig


def plot_data(X, y, x_spans=None, axes=None, facecolor='0.6', edgecolor='0.6', xlabelcolor='k', xvars=None, yvar=None, descriptors=None, title=None, show=True):
    # Get data dimension
    d = X.shape[-1]
    
    # Set domain span
    if x_spans is None:
        x_mins = np.min(X, axis=0)
        x_maxs = np.max(X, axis=0)
    else:
        x_mins, x_maxs = x_spans

    pad = 0.01 * (x_maxs - x_mins)
    x_mins -= pad
    x_maxs += pad
    
    if axes is None:
        fig, axes = plt.subplots(1, d, figsize=(1.5*d, 2), sharey=True)
        inline = True
    else:
        inline = False
    
    if xvars is None:
        xvars = [f'$x_{{{i+1}}}$' for i in range(d)]
    
    for i, xvar in enumerate(xvars):
        ax = axes[i]
        ax.scatter(X[:, i], y, s=25, facecolor=facecolor, edgecolor=edgecolor, alpha=0.7, linewidth=1.0)
        
        # Decoration
        ax.set_xlim([x_mins[i], x_maxs[i]])
        
        if descriptors and i in descriptors:
            ax.set_xlabel(xvar, color=xlabelcolor, fontweight='bold')
        else:
            ax.set_xlabel(xvar)
            
        if yvar and i == 0:
            ax.set_ylabel(yvar)
        
        ax.minorticks_off()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    
    if inline:
        if title:
            fig.suptitle(title, fontsize=10)
        
        fig.subplots_adjust(left=0.04, right=0.97, top=0.9, bottom=0.14)
        fig.tight_layout()
        fig.adjust(top=0.9)
    
    if show:
        plt.show()
    
    if inline:
        return fig, axes
    else:
        return


def plot_kde(x, x_span=None, bandwidth=None, color='0.6', linewidth=0.5, alpha=0.8, 
             display_spine=True, spine_height=0.4, spine_color='k', spine_alpha=0.8, 
             spine_lw=1.0, ax=None):
    # Ensure numpy array
    x = np.asarray(x)

    # Prepare axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    # Set domain span
    if x_span is None:
        x_min = np.min(x)
        x_max = np.max(x)
    else:
        x_min, x_max = x_span

    pad = 0.01 * (x_max - x_min)
    x_min -= pad
    x_max += pad
    
    # Build KDE
    kde = gaussian_kde(x)
    if bandwidth is not None:
        kde.set_bandwidth(bw_method=bandwidth)

    # Grid for plotting
    grid = np.linspace(x_min, x_max, 100)
    density = kde(grid)

    # Plot
    ax.plot(grid, density, color=color, alpha=alpha, linewidth=linewidth)
    
    # Fill under the curve
    ax.fill_between(grid, density, color=color, alpha=alpha/10)
    
    # Spines
    if display_spine:
        mask = (x >= x_min) & (x <= x_max)
        xs = x[mask]

        ymin, ymax = ax.get_ylim()
        y0 = ymin
        y1 = ymin + (ymax - ymin) * spine_height

        ax.vlines(xs, y0, y1, color=spine_color, alpha=spine_alpha, linewidth=spine_lw)
        
    # Decoration
    ax.set_xlim([x_min, x_max])
    return ax


def plot_scatter(ax, X, Y, ctype, label, facecolor=None, edgecolor=None, 
                 s=12, linewidth=0.4, alpha=1, xlabel=None, ylabel=None):
    # Plot optimum point
    idx = int(np.argmax(Y))
    xs = X[idx]
    ys = Y[idx]
    ax.scatter(xs, ys, s=s*2, linewidth=linewidth*2, facecolor=facecolor, edgecolor=edgecolor)
    
    # Vertical line
    ymin = np.min(Y)
    ax.plot([xs, xs], [-5, ys], ':k', zorder=-5)
    
    # Plot points other than the optimum
    ax.scatter(X, Y, s=s, linewidth=linewidth, facecolor=facecolor, edgecolor=edgecolor, label=label, alpha=alpha)
    
    # Decoration  
    ax.set_ylim([-5, 3])
    ax.set_title(ctype.capitalize())
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return


def plot_2D_contour_simple(Z, Y, xlabel='Z', cmap='winter', n_grids=100, kernel='gaussian', 
                           neighbors=30, smoothing=1e-4, epsilon=1e-6, levels=15):
    fig, ax = plt.subplots(figsize=(4, 3))

    if len(Y) < neighbors:
        neighbors = max(5, len(Y)//5)
    
    # Set grids
    zmin, zmax = np.min(Z, axis=0), np.max(Z, axis=0)
    _Z1, _Z2 = np.meshgrid(
        np.linspace(zmin[0], zmax[0], n_grids),
        np.linspace(zmin[1], zmax[1], n_grids)
    )
    _Z = np.c_[_Z1.ravel(), _Z2.ravel()]

    # Filled contour plot
    n = len(Z)
    rbf = RBFInterpolator(Z, Y, kernel=kernel, neighbors=neighbors,
                          smoothing=float(smoothing), epsilon=epsilon)  # keep None to auto-tune; or pass a float
    
    _Y = rbf(_Z).reshape(_Z1.shape)
    contour = ax.contourf(_Z1, _Z2, _Y, levels=15, cmap=cmap, zorder=-10)
    
    # Scatter original points on top
    ax.scatter(Z[:, 0], Z[:, 1], c='w', edgecolors='k', s=30)

    # Colorbar & labels
    fig.colorbar(contour, ax=ax, label='${Y}$')
    ax.set_xlabel(f'${xlabel}_1$')
    ax.set_ylabel(f'${xlabel}_2$')

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_2D_contour(fig, ax, Z, Y, indices_initial, indices_sampled, indices_unsampled,
                    indices_y_subopt, idx_next, idx_max, idx_opt, xlabel='Z', ylabel=r'$\hat{Y}$', 
                    cmap='winter', s=15, n_grids=100, clear=True, legend=True, kernel='gaussian', 
                    neighbors=10, smoothing=1e-4, epsilon=1e-6, levels=15):
    if clear:
        ax.clear()
        
    # Set grids
    zmin, zmax = np.min(Z, axis=0), np.max(Z, axis=0)
    _Z1, _Z2 = np.meshgrid(
        np.linspace(zmin[0], zmax[0], n_grids),
        np.linspace(zmin[1], zmax[1], n_grids)
    )
    _Z = np.c_[_Z1.ravel(), _Z2.ravel()]
    
    # Filled contour plot
    n = len(Z)
    rbf = RBFInterpolator(Z[indices_sampled, :], Y[indices_sampled], kernel=kernel, neighbors=neighbors,
                          smoothing=float(smoothing), epsilon=epsilon)  # keep None to auto-tune; or pass a float
    
    _Y = rbf(_Z).reshape(_Z1.shape)
    contour = ax.contourf(_Z1, _Z2, _Y, levels=15, cmap=cmap, zorder=-10)
    
    indices_sampled = deepcopy(indices_sampled)  # unlink
    indices_sampled = list(set(indices_sampled) - set(indices_initial))
    
    indices_optimal = list(set(indices_y_subopt) | set([idx_opt]))
    
    # Scatter
    ax.scatter(Z[indices_unsampled, 0], Z[indices_unsampled, 1], s=s, marker='o', c='w', edgecolors='k', label='Unsampled', zorder=0)
    ax.scatter(Z[indices_initial, 0], Z[indices_initial, 1], s=s, marker='o', c='0.5', edgecolors='k', label='Initial', zorder=1)
    ax.scatter(Z[indices_sampled, 0], Z[indices_sampled, 1], s=s, marker='o', c='k', edgecolors='k', label='Sampled', zorder=4)
    ax.scatter(Z[indices_optimal, 0], Z[indices_optimal, 1], s=s*3, marker='*', c='y', edgecolors='k', label='Optimal', zorder=2)
    ax.scatter(Z[idx_max, 0], Z[idx_max, 1], s=s*1.5, marker='s', c='b', edgecolors='k', label='Current best', zorder=5)
    ax.scatter(Z[idx_next, 0], Z[idx_next, 1], s=s*1.5, marker='s', c='r', edgecolors='k', label='Next query', zorder=6)
    
    # Decoration
    pos = ax.get_position()
    set_colorbar(fig, pos=[pos.x1 + 0.01, pos.y0, 0.01, pos.y1 - pos.y0], values=_Y, label=ylabel, remove_ticks=True)
    ax.set_xlabel('$' + xlabel + '_{1}$')
    ax.set_ylabel('$' + xlabel + '_{2}$')
    
    if legend and len(fig.legends) == 0:
        handles, labels = ax.get_legend_handles_labels()
        
        handles = [handles[labels.index(idx)] for idx in legend_order]
        labels  = [order for order in legend_order]
        
        fig.legend(handles, labels, frameon=False, loc='lower center', ncol=7, bbox_to_anchor=(0.5, 0.95), fontsize=8)
    return


def plot_regressor_accuracy(ax, Y, Y_mean, indices_initial, indices_sampled, indices_unsampled, 
                            indices_y_subopt, idx_next, idx_max, idx_opt, x_range=None, y_range=None, s=15, 
                            xlabel='$Y$', ylabel=r'$\hat{Y}$', clear=True, legend=True):
    if clear:
        ax.clear()
        
    # Diagonal (x=y) line
    xy = np.linspace(np.min(Y), np.max(Y), 10)
    ax.plot(xy, xy, ':k', zorder=-1)
    
    indices_sampled = deepcopy(indices_sampled)  # unlink
    indices_sampled = list(set(indices_sampled) - set(indices_initial))
    
    indices_optimal = list(set(indices_y_subopt) | set([idx_opt]))
    
    # Prediction
    ax.scatter(Y[indices_unsampled], Y_mean[indices_unsampled], s=s, marker='o', c='w', edgecolors='k', label='Unsampled', zorder=0)
    ax.scatter(Y[indices_initial], Y_mean[indices_initial], s=s, marker='o', c='0.5', edgecolors='k', label='Initial', zorder=1)
    ax.scatter(Y[indices_sampled], Y_mean[indices_sampled], s=s, marker='o', c='k', edgecolors='k', label='Sampled', zorder=4)
    ax.scatter(Y[indices_optimal], Y_mean[indices_optimal], s=s*3, marker='*', c='y', edgecolors='k', label='Optimal', zorder=2)
    ax.scatter(Y[idx_max], Y_mean[idx_max], s=s*1.5, marker='s', c='b', edgecolors='k', label='Current best', zorder=5)
    ax.scatter(Y[idx_next], Y_mean[idx_next], s=s*1.5, marker='s', c='r', edgecolors='k', label='Next query', zorder=6)
        
    # Decorate
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if x_range:
        x_span = x_range[1] - x_range[0]
        alpha = x_span/8
        ax.set_xlim(x_range[0] - alpha, x_range[1] + alpha)
    
    if y_range:
        y_span = y_range[1] - y_range[0]
        alpha = y_span/8
        ax.set_ylim(x_range[0] - alpha, x_range[1] + alpha)
        
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        
        handles = [handles[labels.index(idx)] for idx in legend_order]
        labels  = [order for order in legend_order]
        
        ax.legend(handles, labels, frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    return


def get_uY_with_bounds(Y, Y_mean, Y_std):
    indices = np.argsort(Y)
    return Y[indices], Y_mean[indices], Y_std[indices]


def plot_XY(ax, X, Y, indices_initial, indices_sampled, indices_unsampled, indices_y_subopt, 
            idx_next, idx_max, idx_opt, x_range=None, y_range=None, s=15, xlabel=None, ylabel=None, 
            labels=None, clear=True, legend=True):
    if clear:
        ax.clear()
        
    indices_sampled = deepcopy(indices_sampled)  # unlink
    indices_sampled = list(set(indices_sampled) - set(indices_initial))
    
    indices_optimal = list(set(indices_y_subopt) | set([idx_opt]))
    
    ax.scatter(X[indices_unsampled], Y[indices_unsampled], s=s, marker='o', c='w', edgecolors='k', label='Unsampled', zorder=0)
    ax.scatter(X[indices_initial], Y[indices_initial], s=s, marker='o', c='0.5', edgecolors='k', label='Initial', zorder=1)
    ax.scatter(X[indices_sampled], Y[indices_sampled], s=s, marker='o', c='k', edgecolors='k', label='Sampled', zorder=4)
    ax.scatter(X[indices_optimal], Y[indices_optimal], s=s*3, marker='*', c='y', edgecolors='k', label='Optimal', zorder=2)
    ax.scatter(X[idx_max], Y[idx_max], s=s*1.5, marker='s', c='b', edgecolors='k', label='Current best', zorder=5)
    ax.scatter(X[idx_next], Y[idx_next], s=s*1.5, marker='s', c='r', edgecolors='k', label='Next query', zorder=6)
    
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.xaxis.set_ticklabels([])
    
    ax.set_ylabel(ylabel)
    
    if x_range:
        x_span = x_range[1] - x_range[0]
        alpha = x_span/8
        ax.set_xlim(x_range[0] - alpha, x_range[1] + alpha)
    
    if y_range:
        y_span = y_range[1] - y_range[0]
        alpha = y_span/8
        ax.set_ylim(y_range[0] - alpha, y_range[1] + alpha)
        
    # Plot sample distribution
    indices_sampled = list(set(indices_sampled) | set(indices_initial))
    ax.eventplot(X[indices_sampled], lineoffsets=x_range[0] - alpha, linelengths=alpha, orientation='horizontal', colors='m')
    
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        
        handles = [handles[labels.index(idx)] for idx in legend_order]
        labels  = [order for order in legend_order]
        
        ax.legend(handles, labels, frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    return


def plot_descriptor_transition(ax, x, y, z, xlabel=None, xvars=None, clear=True):
    if clear:
        ax.clear()

    x = np.asarray(x).reshape(-1)
    n = x.shape[0]
    y = np.asarray(y)

    # Generate mapping
    name_to_idx = {name: i for i, name in enumerate(xvars)}
    idx_mat = np.vectorize(lambda v: name_to_idx[v])(y)  # (n,2)

    # Base binary heatmap
    H = np.zeros((len(xvars), n), dtype=int)
    for i in range(n):
        for j in range(2):
            H[idx_mat[i, j], i] = 1
    
    # Get heat map
    V = np.ones((len(xvars), n, 3))
    for i in range(n):
        idx = np.argmax(z[i, :])
        dominant = list(CMAP_IMPROVEMENT.keys())[idx]
        color = CMAP_IMPROVEMENT[dominant]
        V[H[:, i] == 1, i, :] = color

    # plot
    ax.imshow(V, aspect='auto', origin='upper', extent=[x.min()-0.5, x.max()+0.5, len(xvars)-0.5, -0.5], zorder=0)
    
    # Draw phase transition lines
    for i in range(1, n):
        if np.argmax(z[i, :]) != np.argmax(z[i-1, :]):
            ax.plot([x[i]-0.5, x[i]-0.5], [-0.5, len(xvars)-0.5], ':k', lw=1.0, zorder=10)
    
    # Draw grids
    for i in range(n):
        ax.plot([x[i]-0.5, x[i]-0.5], [-0.5, len(xvars)-0.5], '-', color=[0.8, 0.8, 0.8], lw=0.5, zorder=2)
    
    for i in range(len(xvars)):
        ax.plot([x.min()-0.5, x.max()+0.5], [i-0.5, i-0.5], '-', color=[0.8, 0.8, 0.8], lw=0.5, zorder=2)
    
    # Decoration
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.set_ylabel('Variables')
    ylabels = ['$' + xvar[0] + '_{' + str(int(xvar.split('x')[1]) + 1) + '}$' for xvar in xvars]  # 1-indexing
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_ylim(len(ylabels) - 0.5, -0.5)
    return


def plot_descriptor_distribution(ax, x, y, xvars, show_yticks=False, clear=True):
    if clear:
        ax.clear()
        
    y = np.asarray(y)
    y_flat = y.reshape(-1)

    name_to_idx = {name: i for i, name in enumerate(xvars)}
    counts = np.zeros(len(xvars), dtype=int)
    for v in y_flat:
        if v in name_to_idx:
            counts[name_to_idx[v]] += 1

    y_pos = np.arange(len(xvars))
    ax.barh(y_pos, counts, align='center', color='k')

    ax.set_yticks(y_pos)
    if show_yticks:
        ax.set_yticklabels(xvars)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(len(xvars)-0.5, -0.5)
    ax.set_xlabel('Count')
    ax.set_xlim(0, max(1, counts.max()))
    return


def plot_descriptor_distributions(ax, samples, highlight=None, yticklabels=False):
    xvars = sorted(samples[0].keys(), key=lambda k: int(k[1:]))
    ylabels = ['$' + xvar[0] + '_{' + str(int(xvar.split('x')[1]) + 1) + '}$' for xvar in xvars]  # 1-indexing
    
    # Collect probabilities
    data = {k: [] for k in xvars}
    values = []
    for d in samples:
        for k in xvars:
            data[k].append(d[k])
            values.append(d[k])

    # Draw initial boxplot (uniform style first)
    bp = ax.boxplot(
        [data[k] for k in xvars], labels=ylabels, vert=False, patch_artist=True,
        boxprops=dict(facecolor='w', edgecolor='k', linewidth=1.0),
        medianprops=dict(color='k', linewidth=1.0),
        whiskerprops=dict(color='k', linewidth=1.0),
        capprops=dict(color='k', linewidth=1.0)
    )

    # Override colors for highlighted indices
    hl = set(highlight) if highlight is not None else set()
    for idx, box in enumerate(bp['boxes']):
        if idx in hl:
            box.set_facecolor(CMAP_IMPROVEMENT['discrimination'])
        else:
            box.set_facecolor('w')

    ax.set_xlabel('Probability')
    ax.set_xlim(-0.05, np.max(values)*1.1)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    if not yticklabels:
        ax.set_yticklabels([])

    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    return


def plot_trajectory(ax, x, y, colors=None, xlabel=None, ylabel=None, labels=None, clear=True, legend=True):
    if clear:
        ax.clear()
        
    ydim = y.shape[-1]
    if ydim == 1:
        ax.plot(x, y, '.-k')
    else:
        if colors is None:
            colors = get_carray(ydim, mapping='discrete', palette='winter')
            
        for i in range(ydim):
            if labels:
                ax.plot(x, y[:,i], '.-', color=colors[i,:], label=labels[i])
            else:
                ax.plot(x, y[:,i], '.-', color=colors[i,:])
    
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.xaxis.set_ticklabels([])
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    
    if legend:
        ax.legend(frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.97), fontsize=8)
    return


def render_animation(folder, extension='.png', duration=1000, max_frame=30):
    import imageio  # optional dependency (requirements-optional.txt); only GIF export needs it
    #mpl.use('TkAgg')
    
    cprint('Generating GIF using image files in', folder, '...')
    
    # Find files
    filenames = [f for f in os.listdir(folder) if extension in f]
    
    # Sort
    indices = list(map(lambda f: int(f.split('.')[0]), filenames))
    filenames = np.array(filenames)[np.argsort(indices)]
    
    # Set save directory
    parentdir = os.path.abspath(os.path.join(folder, os.path.pardir))
    index = os.path.basename(folder)
    gifdir = os.path.join(parentdir, 'animation' + '_' + index + '.gif')
    
    # Write GIF
    with imageio.get_writer(gifdir, mode='I', duration=duration, loop=0, subrectangles=False) as writer:  # duration in ms
        for i, filename in enumerate(filenames):
            if i > max_frame:
                continue  # only up to 30 frames
            
            # Read image
            filedir = os.path.join(folder, filename)
            img = imageio.imread(filedir)
            
            # Add white background
            if img.ndim == 3 and img.shape[-1] == 4:
                rgb, a = img[..., :3].astype(np.float32), img[..., 3:4].astype(np.float32) / 255.0
                bg = np.full_like(rgb, 255.0)
                img = (rgb * a + bg * (1.0 - a)).astype(np.uint8)
            
            # Write
            writer.append_data(img)
    return
    
    
def plot_data_patterns(dataset_name, show=True, save=True):
    from models import clip_functions, eval_func, KANMean
    
    for data_name in DATASET[dataset_name]:
        df = load_dataset(dataset_name=dataset_name, data_name=data_name)
        
        X = get_X(df)
        y = get_Y(df)
        d = X.shape[1]
        xvars = [f'$x_{{{i+1}}}$' for i in range(d)]
        
        indices_descriptors = (np.array(DESCRIPTOR_INDICES[data_name]) - 1).tolist()
        
        # Plot data
        fig, axes = plot_data(X, y, xlabelcolor=[0.15, 0.39, 0.30], xvars=xvars, 
                              yvar='y', descriptors=indices_descriptors, show=False)

        # Regress data with KAN
        regressor = KANMean()
        dataset = {}
        dataset['train_input'] = dataset['test_input'] = X
        dataset['train_label'] = dataset['test_label'] = y.reshape(-1, 1)
        
        x_range = (0.1, 0.9)
        regressor.fit(dataset, x_range=x_range, verbose=False)
        
        f_KAN = regressor.formula
        xvars = sorted(f_KAN.free_symbols, key=lambda s: s.name)
        f_numpy = sp.lambdify(xvars, f_KAN, modules=[clip_functions, 'numpy'])
        y_KAN = eval_func(f_numpy, X)

        # Draw univariate trends using spline regression
        for i in range(d):
            if i in indices_descriptors:
                color = [0.15, 0.39, 0.30]
            else:
                color = '0.5'
                
            ax = axes[i]
            x = X[:, i]
            spline = spline_regression(x, y_KAN, smooth=0.1, degree=3, show=False)
            
            x_spline = np.linspace(x.min(), x.max(), 100)
            y_spline = spline(x_spline)
            
            ax.plot(x_spline, y_spline, color=color, lw=2, zorder=-10)
        
        # Decoration
        fig.suptitle(data_name + ' data', fontweight='bold', fontsize=12)
        
        if save:
            filedir = os.path.join(imgdir, dataset_name, f'xy_{data_name}.svg')
            fig.savefig(filedir)
    
    plt.show()
    return
