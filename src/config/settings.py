# -*- coding: utf-8 -*-
"""
Configuration settings for Quantum DNA Analysis
"""

import os
import multiprocessing as mp
import matplotlib.pyplot as plt

# CPU optimization settings
CPU_COUNT = mp.cpu_count()
MAX_WORKERS = CPU_COUNT  # ✅ USE ALL 16 CORES!
print(f"💻 CPU Cores Available: {CPU_COUNT}")
print(f"🔥 Max Workers: {MAX_WORKERS} (using ALL cores)")

# ENHANCED CONFIGURATION
CONFIG = {
    'n_sequences': 30,
    'n_trials': 30,
    'shots': 8192,
    'gc_levels': [0.1, 0.3, 0.5, 0.7, 0.9],
    'sequence_length': 14,
    'max_scaling_length': 24,
    'scaling_step': 2,
    'diversity_trials': 50,
    'statistical_sequences': 50,
    'noise_sequences': 25,
    'scaling_sequences': 15,
    'figure_dpi': 600,
    'random_seed': 42,
}

# MASTER SEED MANAGEMENT
ANALYSIS_SEEDS = {
    'diversity': 42,
    'statistical': 84,
    'scaling': 126,
    'numpy': 168,
    'quantum_circuits': 210
}

# DISTINCT IBM COLORS FOR GC LEVELS
IBM_COLORS = {
    # Primary colors for GC levels
    'GC_0.1': '#0f62fe',  # IBM Blue 60
    'GC_0.3': '#ff832b',  # IBM Orange 50
    'GC_0.5': '#24a148',  # IBM Green 50
    'GC_0.7': '#8a3ffc',  # IBM Purple 60
    'GC_0.9': '#fa4d56',  # IBM Red 50

    # Method colors
    'NEQR': '#1192e8',    # IBM Blue 60
    'FRQI': '#ff832b',    # IBM Orange 50
    'Classical': '#24a148', # IBM Green 50

    # Supporting colors
    'Error_Bars': '#525252',  # IBM Gray 70
    'Perfect_Line': '#da1e28', # IBM Red 60
    'Title': '#161616',       # IBM Gray 100
    'Axis_Labels': '#525252', # IBM Gray 70
    'Grid': '#e0e0e0',       # IBM Gray 20
    'Background': '#ffffff',  # White
}

# Minimalistic matplotlib settings
def setup_matplotlib():
    """Setup matplotlib with minimalistic styling"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': CONFIG['figure_dpi'],
        'savefig.format': 'pdf',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': IBM_COLORS['Grid'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1,
        'lines.linewidth': 2,
        'lines.markersize': 8,
    })

def setup_directories():
    """Setup output directories"""
    directories = [
        'results',
        'results/figures',
        'results/tables',
        'results/supplementary',
        'results/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def apply_minimalistic_style(ax, title="", xlabel="", ylabel=""):
    """Apply minimalistic IBM styling to matplotlib axes"""
    ax.grid(True, alpha=0.3, color=IBM_COLORS['Grid'], linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(IBM_COLORS['Axis_Labels'])
    ax.spines['bottom'].set_color(IBM_COLORS['Axis_Labels'])

    # Minimalistic labels
    if title:
        ax.set_title(title, color=IBM_COLORS['Title'], fontweight='bold', pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, color=IBM_COLORS['Axis_Labels'])
    if ylabel:
        ax.set_ylabel(ylabel, color=IBM_COLORS['Axis_Labels'])

    # Clean ticks
    ax.tick_params(colors=IBM_COLORS['Axis_Labels'], direction='out')

    return ax