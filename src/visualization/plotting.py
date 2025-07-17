# -*- coding: utf-8 -*-
"""
Visualization utilities for quantum DNA analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..config.settings import IBM_COLORS, CONFIG, apply_minimalistic_style

class QuantumDNAVisualizer:
    """Visualization utilities for quantum DNA analysis"""

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
    def plot_correlation_analysis(results_df):
        """Create enhanced correlation plot with distinct colors"""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot NEQR data with circles
        for gc_content in sorted(results_df['gc_content'].unique()):
            gc_data = results_df[results_df['gc_content'] == gc_content]
            ax.scatter(gc_data['hamming'], gc_data['neqr_mean'],
                      c=IBM_COLORS[f'GC_{gc_content}'], marker='o', s=100, alpha=0.8,
                      label=f'NEQR (GC={gc_content:.1f})', edgecolors='white', linewidth=1)

        # Plot FRQI data with crosses
        for gc_content in sorted(results_df['gc_content'].unique()):
            gc_data = results_df[results_df['gc_content'] == gc_content]
            ax.scatter(gc_data['hamming'], gc_data['frqi_mean'],
                      c=IBM_COLORS[f'GC_{gc_content}'], marker='x', s=120, alpha=0.8,
                      label=f'FRQI (GC={gc_content:.1f})', linewidth=3)

        # Add perfect agreement line
        ax.plot([0, 1], [0, 1], '--', color=IBM_COLORS['Perfect_Line'],
                alpha=0.8, linewidth=2, label='Perfect Agreement')

        # Apply minimalistic styling
        apply_minimalistic_style(ax,
                                title='Quantum vs Classical Similarity Correlation',
                                xlabel='Classical Hamming Similarity',
                                ylabel='Quantum Similarity')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Position legend to avoid overlap
        ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

        plt.tight_layout()
        plt.savefig('results/figures/correlation_analysis_enhanced.pdf',
                   dpi=CONFIG['figure_dpi'], bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_gc_content_heatmap(results_df):
        """Create GC content vs performance heatmap"""
        # Prepare data for heatmap
        gc_levels = sorted(results_df['gc_content'].unique())

        # Create summary data
        heatmap_data = {}
        for gc in gc_levels:
            gc_data = results_df[results_df['gc_content'] == gc]
            heatmap_data[f'GC {gc:.1f}'] = {
                'NEQR Mean': gc_data['neqr_mean'].mean(),
                'FRQI Mean': gc_data['frqi_mean'].mean(),
                'NEQR Error': gc_data['neqr_error'].mean(),
                'FRQI Error': gc_data['frqi_error'].mean(),
                'Correlation': gc_data['hamming'].mean()
            }

        # Convert to DataFrame for heatmap
        heatmap_df = pd.DataFrame(heatmap_data).T

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use a sequential colormap
        im = ax.imshow(heatmap_df.values, cmap='RdYlBu_r', aspect='auto')

        # Set ticks and labels
        ax.set_xticks(range(len(heatmap_df.columns)))
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_xticklabels(heatmap_df.columns)
        ax.set_yticklabels(heatmap_df.index)

        # Add text annotations
        for i in range(len(heatmap_df.index)):
            for j in range(len(heatmap_df.columns)):
                text = ax.text(j, i, f'{heatmap_df.iloc[i, j]:.3f}',
                              ha="center", va="center", color="black", fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Value', rotation=270, labelpad=20)

        apply_minimalistic_style(ax,
                                title='GC Content vs Quantum Performance',
                                xlabel='Metric',
                                ylabel='GC Content Level')

        plt.tight_layout()
        plt.savefig('results/figures/gc_content_heatmap.pdf',
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
        import scipy.stats as stats
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