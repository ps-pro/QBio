# -*- coding: utf-8 -*-
"""
Enhanced visualization utilities for quantum DNA analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from ..config.settings import IBM_COLORS, CONFIG, apply_minimalistic_style

class QuantumDNAVisualizer:
    """Enhanced visualization utilities for quantum DNA analysis"""

    @staticmethod
    def plot_diversity_comparison(results_df):
        """Create minimalistic diversity comparison plot"""
        fig, ax = plt.subplots(figsize=(16, 10))

        x = np.arange(len(results_df))
        width = 0.25

        # Plot bars with distinct IBM colors
        neqr_bars = ax.bar(x - width, results_df['NEQR_Mean'], width,
                          yerr=results_df['NEQR_Std'],
                          label='NEQR', color=IBM_COLORS['NEQR'],
                          capsize=4, alpha=0.9,
                          error_kw={'color': IBM_COLORS['Error_Bars'], 'linewidth': 1})

        frqi_bars = ax.bar(x, results_df['FRQI_Mean'], width,
                          yerr=results_df['FRQI_Std'],
                          label='FRQI', color=IBM_COLORS['FRQI'],
                          capsize=4, alpha=0.9,
                          error_kw={'color': IBM_COLORS['Error_Bars'], 'linewidth': 1})

        classical_bars = ax.bar(x + width, results_df['Hamming_Similarity'], width,
                               label='Classical', color=IBM_COLORS['Classical'],
                               alpha=0.9)

        # Apply minimalistic styling
        apply_minimalistic_style(ax,
                                title='Quantum vs Classical DNA Similarity Analysis',
                                xlabel='Test Case',
                                ylabel='Similarity Score')

        # Customize x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Description'], rotation=45, ha='right')

        # Position legend to avoid overlap
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.02, 0.98))

        # Set y-axis limits
        ax.set_ylim(-0.1, 1.1)

        plt.tight_layout()
        plt.savefig('results/figures/diversity_comparison_enhanced.pdf',
                   dpi=CONFIG['figure_dpi'], bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_clean_correlation_analysis(results_df):
        """Create clean 2x2 panel correlation analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel A: NEQR Correlation (all GC levels combined)
        ax1.scatter(results_df['hamming'], results_df['neqr_mean'], 
                   color=IBM_COLORS['NEQR'], alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        # Add regression line and R² value for NEQR
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            results_df['hamming'], results_df['neqr_mean'])
        line = slope * results_df['hamming'] + intercept
        ax1.plot(results_df['hamming'], line, color=IBM_COLORS['Perfect_Line'], linewidth=2, linestyle='--')
        ax1.text(0.05, 0.95, f'R² = {r_value**2:.3f}\np < 0.001', 
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        apply_minimalistic_style(ax1, title='NEQR Performance', 
                               xlabel='Classical Hamming Similarity', 
                               ylabel='NEQR Quantum Similarity')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Panel B: FRQI Correlation (all GC levels combined)
        ax2.scatter(results_df['hamming'], results_df['frqi_mean'],
                   color=IBM_COLORS['FRQI'], alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        # Add regression line and R² value for FRQI
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            results_df['hamming'], results_df['frqi_mean'])
        line = slope * results_df['hamming'] + intercept
        ax2.plot(results_df['hamming'], line, color=IBM_COLORS['Perfect_Line'], linewidth=2, linestyle='--')
        ax2.text(0.05, 0.95, f'R² = {r_value**2:.3f}\np < 0.001', 
                transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        apply_minimalistic_style(ax2, title='FRQI Performance', 
                               xlabel='Classical Hamming Similarity', 
                               ylabel='FRQI Quantum Similarity')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Panel C: NEQR vs FRQI Direct Comparison
        ax3.scatter(results_df['neqr_mean'], results_df['frqi_mean'],
                   color=IBM_COLORS['Classical'], alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        # Add y=x reference line
        ax3.plot([0, 1], [0, 1], color=IBM_COLORS['Perfect_Line'], 
                linewidth=2, linestyle='-', label='Perfect Agreement')
        
        # Calculate correlation between methods
        corr_coef, p_val = stats.pearsonr(results_df['neqr_mean'], results_df['frqi_mean'])
        ax3.text(0.05, 0.95, f'r = {corr_coef:.3f}\nn = {len(results_df)}', 
                transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        apply_minimalistic_style(ax3, title='Method Comparison', 
                               xlabel='NEQR Similarity', 
                               ylabel='FRQI Similarity')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_aspect('equal')
        
        # Panel D: Performance Difference vs Classical Similarity
        performance_diff = results_df['neqr_mean'] - results_df['frqi_mean']
        ax4.scatter(results_df['hamming'], performance_diff,
                   color=IBM_COLORS['Error_Bars'], alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        # Add horizontal line at y=0
        ax4.axhline(y=0, color=IBM_COLORS['Perfect_Line'], linewidth=2, linestyle='-')
        
        # Calculate mean improvement
        mean_diff = performance_diff.mean()
        ax4.text(0.05, 0.95, f'Mean Δ = {mean_diff:.3f}\nNEQR Superior: {(performance_diff > 0).sum()}/{len(performance_diff)}', 
                transform=ax4.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        apply_minimalistic_style(ax4, title='Performance Difference (NEQR - FRQI)', 
                               xlabel='Classical Hamming Similarity', 
                               ylabel='Performance Difference')
        ax4.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig('results/figures/correlation_panels_enhanced.pdf',
                   dpi=CONFIG['figure_dpi'], bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_performance_by_gc_content(results_df):
        """Create grouped error bar plots by GC content"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Calculate means and 95% CI for each GC level
        gc_levels = sorted(results_df['gc_content'].unique())
        x_pos = np.arange(len(gc_levels))
        width = 0.35
        
        neqr_means = []
        neqr_ci_lower = []
        neqr_ci_upper = []
        frqi_means = []
        frqi_ci_lower = []
        frqi_ci_upper = []
        
        neqr_errors = []
        frqi_errors = []
        
        for gc in gc_levels:
            gc_data = results_df[results_df['gc_content'] == gc]
            
            # Performance metrics
            neqr_mean = gc_data['neqr_mean'].mean()
            frqi_mean = gc_data['frqi_mean'].mean()
            neqr_means.append(neqr_mean)
            frqi_means.append(frqi_mean)
            
            # 95% confidence intervals
            neqr_ci = stats.t.interval(0.95, len(gc_data)-1, 
                                     loc=neqr_mean, 
                                     scale=stats.sem(gc_data['neqr_mean']))
            frqi_ci = stats.t.interval(0.95, len(gc_data)-1, 
                                     loc=frqi_mean, 
                                     scale=stats.sem(gc_data['frqi_mean']))
            
            neqr_ci_lower.append(neqr_mean - neqr_ci[0])
            neqr_ci_upper.append(neqr_ci[1] - neqr_mean)
            frqi_ci_lower.append(frqi_mean - frqi_ci[0])
            frqi_ci_upper.append(frqi_ci[1] - frqi_mean)
            
            # Error metrics
            neqr_errors.append(gc_data['neqr_error'].mean())
            frqi_errors.append(gc_data['frqi_error'].mean())
        
        # Left plot: Mean Performance by GC Content
        ax1.bar(x_pos - width/2, neqr_means, width, 
               yerr=[neqr_ci_lower, neqr_ci_upper],
               label='NEQR', color=IBM_COLORS['NEQR'], capsize=5, alpha=0.9)
        ax1.bar(x_pos + width/2, frqi_means, width,
               yerr=[frqi_ci_lower, frqi_ci_upper], 
               label='FRQI', color=IBM_COLORS['FRQI'], capsize=5, alpha=0.9)
        
        apply_minimalistic_style(ax1, title='Performance by GC Content', 
                               xlabel='GC Content Level', 
                               ylabel='Mean Similarity Score')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{gc:.1f}' for gc in gc_levels])
        ax1.legend(frameon=False, loc='upper right')
        ax1.set_ylim(0, 1)
        
        # Right plot: Error Rates by GC Content
        ax2.bar(x_pos - width/2, neqr_errors, width,
               label='NEQR', color=IBM_COLORS['NEQR'], alpha=0.9)
        ax2.bar(x_pos + width/2, frqi_errors, width,
               label='FRQI', color=IBM_COLORS['FRQI'], alpha=0.9)
        
        apply_minimalistic_style(ax2, title='Error Rates by GC Content', 
                               xlabel='GC Content Level', 
                               ylabel='Mean Absolute Error')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{gc:.1f}' for gc in gc_levels])
        ax2.legend(frameon=False, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('results/figures/performance_by_gc_enhanced.pdf',
                   dpi=CONFIG['figure_dpi'], bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_performance_distributions(results_df):
        """Create box plots and violin plots for distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Box plots by GC content
        gc_levels = sorted(results_df['gc_content'].unique())
        neqr_data_by_gc = [results_df[results_df['gc_content'] == gc]['neqr_mean'].values 
                           for gc in gc_levels]
        frqi_data_by_gc = [results_df[results_df['gc_content'] == gc]['frqi_mean'].values 
                           for gc in gc_levels]
        
        # Create side-by-side box plots
        positions_neqr = np.arange(len(gc_levels)) * 2 - 0.4
        positions_frqi = np.arange(len(gc_levels)) * 2 + 0.4
        
        bp1 = ax1.boxplot(neqr_data_by_gc, positions=positions_neqr, 
                          patch_artist=True, widths=0.6, 
                          boxprops=dict(facecolor=IBM_COLORS['NEQR'], alpha=0.7),
                          medianprops=dict(color='white', linewidth=2))
        bp2 = ax1.boxplot(frqi_data_by_gc, positions=positions_frqi,
                          patch_artist=True, widths=0.6,
                          boxprops=dict(facecolor=IBM_COLORS['FRQI'], alpha=0.7),
                          medianprops=dict(color='white', linewidth=2))
        
        # Custom legend for box plots
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=IBM_COLORS['NEQR'], alpha=0.7, label='NEQR'),
                          Patch(facecolor=IBM_COLORS['FRQI'], alpha=0.7, label='FRQI')]
        ax1.legend(handles=legend_elements, frameon=False, loc='upper right')
        
        apply_minimalistic_style(ax1, title='Performance Distributions by GC Content', 
                               xlabel='GC Content Level', 
                               ylabel='Similarity Score')
        ax1.set_xticks(np.arange(len(gc_levels)) * 2)
        ax1.set_xticklabels([f'{gc:.1f}' for gc in gc_levels])
        
        # Right: Method comparison violin plot
        combined_data = [results_df['neqr_mean'].values, results_df['frqi_mean'].values]
        parts = ax2.violinplot(combined_data, positions=[1, 2], widths=0.8, showmeans=True)
        
        # Color the violin plots
        parts['bodies'][0].set_facecolor(IBM_COLORS['NEQR'])
        parts['bodies'][0].set_alpha(0.7)
        parts['bodies'][1].set_facecolor(IBM_COLORS['FRQI'])
        parts['bodies'][1].set_alpha(0.7)
        
        # Add statistical comparison
        from scipy.stats import mannwhitneyu
        stat, p_val = mannwhitneyu(results_df['neqr_mean'], results_df['frqi_mean'], 
                                  alternative='two-sided')
        ax2.text(0.05, 0.95, f'Mann-Whitney U test\np = {p_val:.3e}', 
                transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        apply_minimalistic_style(ax2, title='Overall Method Comparison', 
                               xlabel='Method', 
                               ylabel='Similarity Score')
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(['NEQR', 'FRQI'])
        
        plt.tight_layout()
        plt.savefig('results/figures/distribution_comparison_enhanced.pdf',
                   dpi=CONFIG['figure_dpi'], bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_gc_content_trends(results_df):
        """Create trend analysis plots with confidence bands"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Calculate trends with confidence bands
        gc_levels = sorted(results_df['gc_content'].unique())
        
        neqr_means = []
        neqr_stds = []
        frqi_means = []
        frqi_stds = []
        improvement_by_gc = []
        
        for gc in gc_levels:
            gc_data = results_df[results_df['gc_content'] == gc]
            
            neqr_mean = gc_data['neqr_mean'].mean()
            neqr_std = gc_data['neqr_mean'].std()
            frqi_mean = gc_data['frqi_mean'].mean()
            frqi_std = gc_data['frqi_mean'].std()
            
            neqr_means.append(neqr_mean)
            neqr_stds.append(neqr_std)
            frqi_means.append(frqi_mean)
            frqi_stds.append(frqi_std)
            
            improvement = neqr_mean - frqi_mean
            improvement_by_gc.append(improvement)
        
        # Left: Performance trends with confidence bands
        ax1.plot(gc_levels, neqr_means, color=IBM_COLORS['NEQR'], 
                linewidth=3, marker='o', markersize=8, label='NEQR')
        ax1.fill_between(gc_levels, 
                        np.array(neqr_means) - np.array(neqr_stds), 
                        np.array(neqr_means) + np.array(neqr_stds), 
                        color=IBM_COLORS['NEQR'], alpha=0.2)
        
        ax1.plot(gc_levels, frqi_means, color=IBM_COLORS['FRQI'], 
                linewidth=3, marker='s', markersize=8, label='FRQI')
        ax1.fill_between(gc_levels, 
                        np.array(frqi_means) - np.array(frqi_stds), 
                        np.array(frqi_means) + np.array(frqi_stds), 
                        color=IBM_COLORS['FRQI'], alpha=0.2)
        
        apply_minimalistic_style(ax1, title='Performance Trends by GC Content', 
                               xlabel='GC Content', 
                               ylabel='Mean Similarity Score')
        ax1.legend(frameon=False, loc='best')
        ax1.set_ylim(0, 1)
        
        # Right: Performance improvement (NEQR - FRQI) by GC content
        colors = [IBM_COLORS['Classical'] if imp >= 0 else IBM_COLORS['Error_Bars'] 
                 for imp in improvement_by_gc]
        bars = ax2.bar(range(len(gc_levels)), improvement_by_gc, color=colors, alpha=0.8)
        
        # Add zero reference line
        ax2.axhline(y=0, color='black', linewidth=1, linestyle='-')
        
        # Annotate bars with values
        for i, (bar, val) in enumerate(zip(bars, improvement_by_gc)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                    f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        apply_minimalistic_style(ax2, title='NEQR Advantage by GC Content', 
                               xlabel='GC Content Level', 
                               ylabel='Performance Difference (NEQR - FRQI)')
        ax2.set_xticks(range(len(gc_levels)))
        ax2.set_xticklabels([f'{gc:.1f}' for gc in gc_levels])
        
        # Add legend for improvement
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=IBM_COLORS['Classical'], alpha=0.8, label='NEQR Superior'),
                          Patch(facecolor=IBM_COLORS['Error_Bars'], alpha=0.8, label='FRQI Superior')]
        ax2.legend(handles=legend_elements, frameon=False, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('results/figures/gc_trends_enhanced.pdf',
                   dpi=CONFIG['figure_dpi'], bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_scaling_results(results_df):
        """Create comprehensive scaling analysis plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

        # Group by length for plotting
        length_groups = results_df.groupby('length').agg({
            'neqr_error': ['mean', 'std'],
            'frqi_error': ['mean', 'std'],
            'neqr_circuit_depth': ['mean', 'std'],
            'frqi_circuit_depth': ['mean', 'std'],
            'neqr_gate_count': ['mean', 'std'],
            'frqi_gate_count': ['mean', 'std'],
            'actual_similarity': ['mean', 'std'],
            'neqr_mean': ['mean', 'std'],
            'frqi_mean': ['mean', 'std']
        })

        lengths = length_groups.index

        # Plot 1: Error scaling
        ax1.errorbar(lengths, length_groups[('neqr_error', 'mean')],
                    yerr=length_groups[('neqr_error', 'std')],
                    marker='o', color=IBM_COLORS['NEQR'], label='NEQR',
                    linewidth=2, markersize=8, capsize=5)
        ax1.errorbar(lengths, length_groups[('frqi_error', 'mean')],
                    yerr=length_groups[('frqi_error', 'std')],
                    marker='s', color=IBM_COLORS['FRQI'], label='FRQI',
                    linewidth=2, markersize=8, capsize=5)

        apply_minimalistic_style(ax1,
                                title='Error vs Sequence Length',
                                xlabel='Sequence Length',
                                ylabel='Mean Absolute Error')
        ax1.legend(frameon=False)

        # Plot 2: Circuit depth scaling (log scale)
        ax2.semilogy(lengths, length_groups[('neqr_circuit_depth', 'mean')],
                    marker='o', color=IBM_COLORS['NEQR'], label='NEQR',
                    linewidth=2, markersize=8)
        ax2.semilogy(lengths, length_groups[('frqi_circuit_depth', 'mean')],
                    marker='s', color=IBM_COLORS['FRQI'], label='FRQI',
                    linewidth=2, markersize=8)

        apply_minimalistic_style(ax2,
                                title='Circuit Depth vs Sequence Length',
                                xlabel='Sequence Length',
                                ylabel='Circuit Depth (log scale)')
        ax2.legend(frameon=False)

        # Plot 3: Gate count scaling (log scale)
        ax3.semilogy(lengths, length_groups[('neqr_gate_count', 'mean')],
                    marker='o', color=IBM_COLORS['NEQR'], label='NEQR',
                    linewidth=2, markersize=8)
        ax3.semilogy(lengths, length_groups[('frqi_gate_count', 'mean')],
                    marker='s', color=IBM_COLORS['FRQI'], label='FRQI',
                    linewidth=2, markersize=8)

        apply_minimalistic_style(ax3,
                                title='Gate Count vs Sequence Length',
                                xlabel='Sequence Length',
                                ylabel='Total Gates (log scale)')
        ax3.legend(frameon=False)

        # Plot 4: Correlation stability
        correlations_neqr = []
        correlations_frqi = []

        for length in lengths:
            length_data = results_df[results_df['length'] == length]
            if len(length_data) > 3:
                corr_neqr = stats.pearsonr(length_data['actual_similarity'], length_data['neqr_mean'])[0]
                corr_frqi = stats.pearsonr(length_data['actual_similarity'], length_data['frqi_mean'])[0]
                correlations_neqr.append(corr_neqr)
                correlations_frqi.append(corr_frqi)
            else:
                correlations_neqr.append(0)
                correlations_frqi.append(0)

        ax4.plot(lengths, correlations_neqr, marker='o', color=IBM_COLORS['NEQR'],
                label='NEQR', linewidth=2, markersize=8)
        ax4.plot(lengths, correlations_frqi, marker='s', color=IBM_COLORS['FRQI'],
                label='FRQI', linewidth=2, markersize=8)

        apply_minimalistic_style(ax4,
                                title='Correlation Stability vs Length',
                                xlabel='Sequence Length',
                                ylabel='Correlation with Classical')
        ax4.legend(frameon=False)
        ax4.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig('results/figures/scaling_analysis_enhanced.pdf',
                   dpi=CONFIG['figure_dpi'], bbox_inches='tight')
        plt.show()


# # -*- coding: utf-8 -*-
# """
# Visualization utilities for quantum DNA analysis
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from ..config.settings import IBM_COLORS, CONFIG, apply_minimalistic_style

# class QuantumDNAVisualizer:
#     """Visualization utilities for quantum DNA analysis"""

#     @staticmethod
#     def plot_diversity_comparison(results_df):
#         """Create minimalistic diversity comparison plot"""
#         fig, ax = plt.subplots(figsize=(16, 10))

#         x = np.arange(len(results_df))
#         width = 0.25

#         # Plot bars with distinct IBM colors
#         neqr_bars = ax.bar(x - width, results_df['NEQR_Mean'], width,
#                           yerr=results_df['NEQR_Std'],
#                           label='NEQR', color=IBM_COLORS['NEQR'],
#                           capsize=4, alpha=0.9,
#                           error_kw={'color': IBM_COLORS['Error_Bars'], 'linewidth': 1})

#         frqi_bars = ax.bar(x, results_df['FRQI_Mean'], width,
#                           yerr=results_df['FRQI_Std'],
#                           label='FRQI', color=IBM_COLORS['FRQI'],
#                           capsize=4, alpha=0.9,
#                           error_kw={'color': IBM_COLORS['Error_Bars'], 'linewidth': 1})

#         classical_bars = ax.bar(x + width, results_df['Hamming_Similarity'], width,
#                                label='Classical', color=IBM_COLORS['Classical'],
#                                alpha=0.9)

#         # Apply minimalistic styling
#         apply_minimalistic_style(ax,
#                                 title='Quantum vs Classical DNA Similarity Analysis',
#                                 xlabel='Test Case',
#                                 ylabel='Similarity Score')

#         # Customize x-axis
#         ax.set_xticks(x)
#         ax.set_xticklabels(results_df['Description'], rotation=45, ha='right')

#         # Position legend to avoid overlap
#         ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.02, 0.98))

#         # Set y-axis limits
#         ax.set_ylim(-0.1, 1.1)

#         plt.tight_layout()
#         plt.savefig('results/figures/diversity_comparison_enhanced.pdf',
#                    dpi=CONFIG['figure_dpi'], bbox_inches='tight')
#         plt.show()

#     @staticmethod
#     def plot_correlation_analysis(results_df):
#         """Create enhanced correlation plot with distinct colors"""
#         fig, ax = plt.subplots(figsize=(12, 12))

#         # Plot NEQR data with circles
#         for gc_content in sorted(results_df['gc_content'].unique()):
#             gc_data = results_df[results_df['gc_content'] == gc_content]
#             ax.scatter(gc_data['hamming'], gc_data['neqr_mean'],
#                       c=IBM_COLORS[f'GC_{gc_content}'], marker='o', s=100, alpha=0.8,
#                       label=f'NEQR (GC={gc_content:.1f})', edgecolors='white', linewidth=1)

#         # Plot FRQI data with crosses
#         for gc_content in sorted(results_df['gc_content'].unique()):
#             gc_data = results_df[results_df['gc_content'] == gc_content]
#             ax.scatter(gc_data['hamming'], gc_data['frqi_mean'],
#                       c=IBM_COLORS[f'GC_{gc_content}'], marker='x', s=120, alpha=0.8,
#                       label=f'FRQI (GC={gc_content:.1f})', linewidth=3)

#         # Add perfect agreement line
#         ax.plot([0, 1], [0, 1], '--', color=IBM_COLORS['Perfect_Line'],
#                 alpha=0.8, linewidth=2, label='Perfect Agreement')

#         # Apply minimalistic styling
#         apply_minimalistic_style(ax,
#                                 title='Quantum vs Classical Similarity Correlation',
#                                 xlabel='Classical Hamming Similarity',
#                                 ylabel='Quantum Similarity')

#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.set_aspect('equal')

#         # Position legend to avoid overlap
#         ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

#         plt.tight_layout()
#         plt.savefig('results/figures/correlation_analysis_enhanced.pdf',
#                    dpi=CONFIG['figure_dpi'], bbox_inches='tight')
#         plt.show()

#     @staticmethod
#     def plot_gc_content_heatmap(results_df):
#         """Create GC content vs performance heatmap"""
#         # Prepare data for heatmap
#         gc_levels = sorted(results_df['gc_content'].unique())

#         # Create summary data
#         heatmap_data = {}
#         for gc in gc_levels:
#             gc_data = results_df[results_df['gc_content'] == gc]
#             heatmap_data[f'GC {gc:.1f}'] = {
#                 'NEQR Mean': gc_data['neqr_mean'].mean(),
#                 'FRQI Mean': gc_data['frqi_mean'].mean(),
#                 'NEQR Error': gc_data['neqr_error'].mean(),
#                 'FRQI Error': gc_data['frqi_error'].mean(),
#                 'Correlation': gc_data['hamming'].mean()
#             }

#         # Convert to DataFrame for heatmap
#         heatmap_df = pd.DataFrame(heatmap_data).T

#         # Create heatmap
#         fig, ax = plt.subplots(figsize=(10, 8))

#         # Use a sequential colormap
#         im = ax.imshow(heatmap_df.values, cmap='RdYlBu_r', aspect='auto')

#         # Set ticks and labels
#         ax.set_xticks(range(len(heatmap_df.columns)))
#         ax.set_yticks(range(len(heatmap_df.index)))
#         ax.set_xticklabels(heatmap_df.columns)
#         ax.set_yticklabels(heatmap_df.index)

#         # Add text annotations
#         for i in range(len(heatmap_df.index)):
#             for j in range(len(heatmap_df.columns)):
#                 text = ax.text(j, i, f'{heatmap_df.iloc[i, j]:.3f}',
#                               ha="center", va="center", color="black", fontweight='bold')

#         # Add colorbar
#         cbar = plt.colorbar(im, ax=ax)
#         cbar.set_label('Performance Value', rotation=270, labelpad=20)

#         apply_minimalistic_style(ax,
#                                 title='GC Content vs Quantum Performance',
#                                 xlabel='Metric',
#                                 ylabel='GC Content Level')

#         plt.tight_layout()
#         plt.savefig('results/figures/gc_content_heatmap.pdf',
#                    dpi=CONFIG['figure_dpi'], bbox_inches='tight')
#         plt.show()

#     @staticmethod
#     def plot_scaling_results(results_df):
#         """Create comprehensive scaling analysis plots"""
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

#         # Group by length for plotting
#         length_groups = results_df.groupby('length').agg({
#             'neqr_error': ['mean', 'std'],
#             'frqi_error': ['mean', 'std'],
#             'neqr_circuit_depth': ['mean', 'std'],
#             'frqi_circuit_depth': ['mean', 'std'],
#             'neqr_gate_count': ['mean', 'std'],
#             'frqi_gate_count': ['mean', 'std'],
#             'actual_similarity': ['mean', 'std'],
#             'neqr_mean': ['mean', 'std'],
#             'frqi_mean': ['mean', 'std']
#         })

#         lengths = length_groups.index

#         # Plot 1: Error scaling
#         ax1.errorbar(lengths, length_groups[('neqr_error', 'mean')],
#                     yerr=length_groups[('neqr_error', 'std')],
#                     marker='o', color=IBM_COLORS['NEQR'], label='NEQR',
#                     linewidth=2, markersize=8, capsize=5)
#         ax1.errorbar(lengths, length_groups[('frqi_error', 'mean')],
#                     yerr=length_groups[('frqi_error', 'std')],
#                     marker='s', color=IBM_COLORS['FRQI'], label='FRQI',
#                     linewidth=2, markersize=8, capsize=5)

#         apply_minimalistic_style(ax1,
#                                 title='Error vs Sequence Length',
#                                 xlabel='Sequence Length',
#                                 ylabel='Mean Absolute Error')
#         ax1.legend(frameon=False)

#         # Plot 2: Circuit depth scaling (log scale)
#         ax2.semilogy(lengths, length_groups[('neqr_circuit_depth', 'mean')],
#                     marker='o', color=IBM_COLORS['NEQR'], label='NEQR',
#                     linewidth=2, markersize=8)
#         ax2.semilogy(lengths, length_groups[('frqi_circuit_depth', 'mean')],
#                     marker='s', color=IBM_COLORS['FRQI'], label='FRQI',
#                     linewidth=2, markersize=8)

#         apply_minimalistic_style(ax2,
#                                 title='Circuit Depth vs Sequence Length',
#                                 xlabel='Sequence Length',
#                                 ylabel='Circuit Depth (log scale)')
#         ax2.legend(frameon=False)

#         # Plot 3: Gate count scaling (log scale)
#         ax3.semilogy(lengths, length_groups[('neqr_gate_count', 'mean')],
#                     marker='o', color=IBM_COLORS['NEQR'], label='NEQR',
#                     linewidth=2, markersize=8)
#         ax3.semilogy(lengths, length_groups[('frqi_gate_count', 'mean')],
#                     marker='s', color=IBM_COLORS['FRQI'], label='FRQI',
#                     linewidth=2, markersize=8)

#         apply_minimalistic_style(ax3,
#                                 title='Gate Count vs Sequence Length',
#                                 xlabel='Sequence Length',
#                                 ylabel='Total Gates (log scale)')
#         ax3.legend(frameon=False)

#         # Plot 4: Correlation stability
#         import scipy.stats as stats
#         correlations_neqr = []
#         correlations_frqi = []

#         for length in lengths:
#             length_data = results_df[results_df['length'] == length]
#             if len(length_data) > 3:
#                 corr_neqr = stats.pearsonr(length_data['actual_similarity'], length_data['neqr_mean'])[0]
#                 corr_frqi = stats.pearsonr(length_data['actual_similarity'], length_data['frqi_mean'])[0]
#                 correlations_neqr.append(corr_neqr)
#                 correlations_frqi.append(corr_frqi)
#             else:
#                 correlations_neqr.append(0)
#                 correlations_frqi.append(0)

#         ax4.plot(lengths, correlations_neqr, marker='o', color=IBM_COLORS['NEQR'],
#                 label='NEQR', linewidth=2, markersize=8)
#         ax4.plot(lengths, correlations_frqi, marker='s', color=IBM_COLORS['FRQI'],
#                 label='FRQI', linewidth=2, markersize=8)

#         apply_minimalistic_style(ax4,
#                                 title='Correlation Stability vs Length',
#                                 xlabel='Sequence Length',
#                                 ylabel='Correlation with Classical')
#         ax4.legend(frameon=False)
#         ax4.set_ylim(0, 1)

#         plt.tight_layout()
#         plt.savefig('results/figures/scaling_analysis_enhanced.pdf',
#                    dpi=CONFIG['figure_dpi'], bbox_inches='tight')
#         plt.show()