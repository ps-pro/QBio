# -*- coding: utf-8 -*-
"""
Enhanced statistical analysis utilities
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

class EnhancedStatisticalAnalysis:
    """Enhanced statistical analysis with FDR correction and ANOVA"""

    @staticmethod
    def apply_fdr_correction(p_values, alpha=0.05):
        """Apply FDR correction using Benjamini-Hochberg procedure"""
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        return rejected, p_corrected

    @staticmethod
    def perform_anova_analysis(data_dict, dependent_var):
        """Perform ANOVA analysis across groups"""
        groups = [data_dict[group][dependent_var] for group in data_dict.keys()]

        # Remove any empty groups
        groups = [group for group in groups if len(group) > 0]

        if len(groups) < 2:
            return None

        f_stat, p_value = f_oneway(*groups)

        # Calculate effect size (eta-squared)
        total_mean = np.mean(np.concatenate(groups))
        between_group_var = sum(len(group) * (np.mean(group) - total_mean)**2 for group in groups)
        within_group_var = sum(np.sum((group - np.mean(group))**2) for group in groups)

        eta_squared = between_group_var / (between_group_var + within_group_var)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'degrees_of_freedom': (len(groups) - 1, len(np.concatenate(groups)) - len(groups))
        }

    @staticmethod
    def calculate_comprehensive_stats(data):
        """Calculate comprehensive statistical summary"""
        if len(data) == 0:
            return {}

        return {
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'median': np.median(data),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'ci_lower': np.percentile(data, 2.5),
            'ci_upper': np.percentile(data, 97.5),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'n_samples': len(data)
        }

    @staticmethod
    def bootstrap_confidence_interval(data, statistic=np.mean, n_bootstrap=1000, confidence=0.95):
        """Calculate bootstrap confidence interval"""
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))

        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        return {
            'lower': np.percentile(bootstrap_stats, lower_percentile),
            'upper': np.percentile(bootstrap_stats, upper_percentile),
            'bootstrap_std': np.std(bootstrap_stats)
        }