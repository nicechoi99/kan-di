"""
plot_style.py — KAN-DI project figure styles (consolidated).

This file captures ALL colour palettes, rcParams presets, and helper
functions actually used in this paper's codebase. Import what you need.

Usage:
    from common.plot_style import apply_paper_style, COLORS, save_fig
    apply_paper_style()           # default FS=14 / DPI=200
    apply_paper_style(preset='manuscript')  # FS=9 / DPI=600 for journal
"""

import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Core method colours (BO comparison figures) ──────────────────────────────
COLORS = {
    'ZERO-EI':      '#B3B3B3',
    'ZERO-TS':      '#4D4D4D',
    'ZERO-UCB':     '#7F7F7F',
    'KAN-EI':       '#FFA64D',
    'KAN-DI':       '#FF7F0E',
    'KAN-DI-EI':    '#A7C5FF',
    'KAN-DI-TS':    '#4A90E2',
    'KAN-DI-UCB-H': '#1A4FCC',
    'KAN-DI-UCB-L': '#1A4FCC',
}
HATCHES = {'KAN-DI-UCB-H': '////'}

# ── Semantic / role-based colours ────────────────────────────────────────────
C_KAN       = '#1B6B35'   # KAN highlight — dark green
C_ZERO      = '#555555'   # baseline GP — dark grey
C_HIGHLIGHT = '#1B6B35'   # positive emphasis (= KAN green)
C_WARNING   = '#C0392B'   # failure / warning — red
C_TARGET    = '#F9A825'   # y_target annotation — amber
C_NEUTRAL   = '#888888'   # neutral / secondary text
C_SAASBO    = '#6A1B9A'   # SAASBO — purple

# ── Acquisition-function component colours ───────────────────────────────────
C_IE = '#F4511E'   # I_e exploitation — orange-red
C_ID = '#8E24AA'   # I_d discriminative — purple
C_IU = '#1E88E5'   # I_u uniformity — blue

# ── KAN prior-mean comparison (bar charts) ───────────────────────────────────
C_BAR_ZERO = '#ADB5BD'   # zero-mean GP baseline
C_BAR_SAME = '#E76F51'   # same-data KAN-mean
C_BAR_SEP  = '#2A9D8F'   # separate-data KAN-mean

# ── Campaign descriptor palette ─────────────────────────────────────────────
DESC_PALETTE = {
    'density':   '#9b59b6',
    'bp':        '#3498db',
    'mp':        '#16a085',
    'pKa':       '#C0392B',
    'bde_amide': '#1B5E20',
    'viscosity': '#7f8c8d',
    'logP':      '#e67e22',
}

# ── Campaign figure green palette ────────────────────────────────────────────
C_DARK_GREEN  = '#2E7D32'
C_GREEN       = '#43A047'
C_LIGHT_GREEN = '#81C784'
C_GREY        = '#546E7A'

# ── Grid / axis helper colours ───────────────────────────────────────────────
C_GRID   = '#EEEEEE'
C_AXLINE = '#BBBBBB'
C_BANDLO = '#FFE0B2'

# ── Colorblind-safe universal (Okabe-Ito) ────────────────────────────────────
OKABE_ITO = [
    '#0072B2', '#E69F00', '#009E73', '#D55E00',
    '#CC79A7', '#56B4E9', '#F0E442', '#000000',
]


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE SIZES (inches)
# ═══════════════════════════════════════════════════════════════════════════════
WIDTH_SINGLE = 3.5    # 89 mm  — single-column (Nature/ACS)
WIDTH_DOUBLE = 7.0    # 180 mm — double-column
WIDTH_SI     = 10.0   # SI supplementary (wide)


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLE PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_paper_style(preset='si'):
    """Apply rcParams matching this project's figures.

    Presets
    -------
    'si'         : FS=14, DPI=200  — script_main / defense SI figures (default)
    'manuscript' : FS=9,  DPI=600  — camera-ready main-text figures
    'poster'     : FS=22, DPI=200  — common/utils.py legacy (large font)
    'campaign'   : FS=26, DPI=300  — campaign/fig_combined.py (26x25 canvas)
    """
    presets = {
        'si': dict(
            fs=14, dpi=200,
            font_sans=['Inter', 'Arial', 'Calibri', 'DejaVu Sans'],
            extra={
                'legend.columnspacing': 0.6,
                'legend.handletextpad': 0.3,
                'axes.xmargin': 0.0,
                'mathtext.default': 'regular',
            },
        ),
        'manuscript': dict(
            fs=9, dpi=600,
            font_sans=['Arial', 'Helvetica', 'DejaVu Sans'],
            extra={
                'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                'axes.spines.top': False, 'axes.spines.right': False,
                'axes.linewidth': 0.8, 'axes.labelpad': 4,
                'xtick.direction': 'out', 'ytick.direction': 'out',
                'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
                'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
                'xtick.minor.visible': True, 'ytick.minor.visible': True,
                'savefig.pad_inches': 0.05,
                'axes.prop_cycle': mpl.cycler(color=OKABE_ITO),
            },
        ),
        'poster': dict(
            fs=22, dpi=200,
            font_sans=['Inter', 'Arial', 'DejaVu Sans'],
            extra={
                'font.weight': '300',
                'axes.titleweight': '400', 'axes.labelweight': '300',
                'axes.labelpad': 2,
                'legend.labelspacing': 0.5,
                'legend.columnspacing': 0.5,
                'legend.handletextpad': 0.2,
                'hatch.linewidth': 0.5, 'hatch.color': 'w',
                'mathtext.default': 'regular',
            },
        ),
        'campaign': dict(
            fs=26, dpi=300,
            font_sans=['Inter', 'Arial', 'DejaVu Sans'],
            extra={
                'axes.titlesize': 30, 'xtick.labelsize': 25,
                'ytick.labelsize': 25, 'legend.fontsize': 26,
                'axes.linewidth': 1.8,
                'xtick.major.width': 1.8, 'ytick.major.width': 1.8,
                'xtick.major.size': 7, 'ytick.major.size': 7,
            },
        ),
    }

    cfg = presets.get(preset, presets['si'])
    fs = cfg['fs']

    rc = {
        'figure.dpi':       cfg['dpi'],
        'savefig.dpi':      cfg['dpi'] * 4,
        'font.size':        fs,
        'axes.titlesize':   fs,
        'axes.labelsize':   fs,
        'xtick.labelsize':  fs - 2,
        'ytick.labelsize':  fs - 2,
        'legend.fontsize':  fs - 2,
        'legend.frameon':   False,
        'lines.linewidth':  1.5,
        'font.family':      'sans-serif',
        'font.sans-serif':  cfg['font_sans'],
        'savefig.transparent': False,
        'text.usetex':      False,
        'savefig.bbox':     'tight',
    }
    rc.update(cfg.get('extra', {}))
    mpl.rcParams.update(rc)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_spines(ax, keep_all=False):
    """Remove top/right spines and style remaining ones."""
    color = '#212529'
    if keep_all:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(color)
            s.set_linewidth(0.8)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(colors=color)


def panel_label(ax, label, x=-0.07, y=1.02, fs=None):
    """Add bold panel label (a), (b), ... to axes."""
    if fs is None:
        fs = mpl.rcParams['font.size'] + 2
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=fs, fontweight='bold', va='bottom', ha='right')


def convergence_band(ax, trajectories, color, label=None, alpha=0.18):
    """Plot median + IQR band for convergence trajectories."""
    arr = np.array(trajectories)
    med = np.median(arr, axis=0)
    q25 = np.percentile(arr, 25, axis=0)
    q75 = np.percentile(arr, 75, axis=0)
    ks  = np.arange(len(med))
    ax.plot(ks, med, color=color, lw=2, label=label, zorder=3)
    ax.fill_between(ks, q25, q75, color=color, alpha=alpha, zorder=2)
    return med, q25, q75


def save_fig(fig, name, outdir='figures', formats=('svg', 'png')):
    """Save figure in multiple formats. Default: SVG + PNG."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out / f'{name}.{fmt}'
        fig.savefig(path, bbox_inches='tight')
    print(f"[saved] {', '.join(str(out / f'{name}.{f}') for f in formats)}")
