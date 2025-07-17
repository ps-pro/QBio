# Config package initialization
"""
Enhanced Quantum DNA Analysis Package
"""

__version__ = "1.0.0"
__author__ = "Quantum DNA Analysis Team"
__email__ = "contact@quantumdna.com"

# src/config/__init__.py
"""
Configuration module for quantum DNA analysis
"""

from .settings import CONFIG, ANALYSIS_SEEDS, IBM_COLORS, setup_matplotlib, setup_directories, apply_minimalistic_style

__all__ = [
    'CONFIG',
    'ANALYSIS_SEEDS', 
    'IBM_COLORS',
    'setup_matplotlib',
    'setup_directories',
    'apply_minimalistic_style'
]
