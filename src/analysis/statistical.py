# -*- coding: utf-8 -*-
"""
Enhanced statistical analysis with ANOVA and advanced visualizations
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
import gc
from ..config.settings import CONFIG, ANALYSIS_SEEDS
from ..utils.sequence import EnhancedSequenceAnalyzer
from ..utils.stats import EnhancedStatisticalAnalysis
from ..utils.logging import logger
from ..visualization.plotting import QuantumDNAVisualizer
import pickle

class EnhancedStatisticalAnalyzer:
    """Enhanced statistical analyzer with ANOVA and advanced visualizations"""

    def __init__(self, runner):
        self.runner = runner
        self.results = None
        self.anova_results = None
        self.statistical_summary = None

    def analyze(self, n_sequences=CONFIG['statistical_sequences'],
                sequence_length=CONFIG['sequence_length'],
                gc_levels=CONFIG['gc_levels'],
                n_trials=CONFIG['n_trials']):
        """Run enhanced statistical analysis"""
        print(f"\n📈 Enhanced Statistical Analysis...")
        print(f"   - Sequences per GC level: {n_sequences}")
        print(f"   - Trials per sequence: {n_trials}")
        print(f"   - GC levels: {gc_levels}")
        print(f"   - Total circuits: {len(gc_levels) * n_sequences * n_trials * 2}")

        # Set seed for reproducibility
        np.random.seed(ANALYSIS_SEEDS['statistical'])

        results_data = []
        gc_grouped_data = {gc: {'neqr': [], 'frqi': [], 'hamming': []} for gc in gc_levels}

        total_work = len(gc_levels) * n_sequences
        with tqdm(total=total_work, 
                  desc="📈 Statistical Analysis", 
                  ncols=120, 
                  ascii=False,
                  position=0,
                  bar_format='{desc}: {percentage:3.0f}%|{bar:80}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for gc_content in gc_levels:
                for seq_idx in range(n_sequences):
                    # Generate sequences with controlled GC content
                    seq1 = EnhancedSequenceAnalyzer.generate_random_sequence(
                        sequence_length, gc_content=gc_content, seed=seq_idx
                    )
                    seq2 = EnhancedSequenceAnalyzer.generate_random_sequence(
                        sequence_length, gc_content=gc_content, seed=seq_idx + 1000
                    )

                    # Calculate classical similarity
                    hamming = EnhancedSequenceAnalyzer.calculate_hamming_similarity(seq1, seq2)

                    # Run quantum simulations
                    neqr_trials, frqi_trials = self.runner.run_both_methods_parallel(seq1, seq2, n_trials)

                    # Calculate comprehensive statistics
                    neqr_stats = EnhancedStatisticalAnalysis.calculate_comprehensive_stats(neqr_trials)
                    frqi_stats = EnhancedStatisticalAnalysis.calculate_comprehensive_stats(frqi_trials)

                    # Statistical tests
                    _, p_neqr_classical = stats.ttest_1samp(neqr_trials, hamming)
                    _, p_frqi_classical = stats.ttest_1samp(frqi_trials, hamming)
                    _, p_neqr_frqi = stats.ttest_ind(neqr_trials, frqi_trials)

                    # Store for ANOVA
                    gc_grouped_data[gc_content]['neqr'].extend(neqr_trials)
                    gc_grouped_data[gc_content]['frqi'].extend(frqi_trials)
                    gc_grouped_data[gc_content]['hamming'].append(hamming)

                    # Store results
                    results_data.append({
                        'gc_content': gc_content,
                        'sequence_pair': seq_idx,
                        'sequence_length': sequence_length,
                        'hamming': hamming,

                        # NEQR comprehensive stats
                        'neqr_mean': neqr_stats['mean'],
                        'neqr_std': neqr_stats['std'],
                        'neqr_median': neqr_stats['median'],
                        'neqr_ci_lower': neqr_stats['ci_lower'],
                        'neqr_ci_upper': neqr_stats['ci_upper'],
                        'neqr_skewness': neqr_stats['skewness'],
                        'neqr_kurtosis': neqr_stats['kurtosis'],

                        # FRQI comprehensive stats
                        'frqi_mean': frqi_stats['mean'],
                        'frqi_std': frqi_stats['std'],
                        'frqi_median': frqi_stats['median'],
                        'frqi_ci_lower': frqi_stats['ci_lower'],
                        'frqi_ci_upper': frqi_stats['ci_upper'],
                        'frqi_skewness': frqi_stats['skewness'],
                        'frqi_kurtosis': frqi_stats['kurtosis'],

                        # Error metrics
                        'neqr_error': abs(neqr_stats['mean'] - hamming),
                        'frqi_error': abs(frqi_stats['mean'] - hamming),

                        # Statistical tests
                        'p_neqr_classical': p_neqr_classical,
                        'p_frqi_classical': p_frqi_classical,
                        'p_neqr_frqi': p_neqr_frqi,

                        'n_trials': n_trials
                    })

                    pbar.update(1)

                    # Periodic memory cleanup
                    if seq_idx % 25 == 0:
                        gc.collect()

        self.results = pd.DataFrame(results_data)

        # Perform ANOVA analysis
        self._perform_anova_analysis(gc_grouped_data)

        # Apply FDR correction
        self._apply_fdr_corrections()

        # Generate statistical summary
        self._generate_statistical_summary()

        # Clear cache
        self.runner.clear_cache()

        return self.results

    def _perform_anova_analysis(self, gc_grouped_data):
        """Perform ANOVA analysis across GC content levels"""
        self.anova_results = {}

        # ANOVA for NEQR performance
        neqr_anova = EnhancedStatisticalAnalysis.perform_anova_analysis(
            {f'GC_{gc}': {'neqr': data['neqr']} for gc, data in gc_grouped_data.items()},
            'neqr'
        )
        self.anova_results['neqr'] = neqr_anova

        # ANOVA for FRQI performance
        frqi_anova = EnhancedStatisticalAnalysis.perform_anova_analysis(
            {f'GC_{gc}': {'frqi': data['frqi']} for gc, data in gc_grouped_data.items()},
            'frqi'
        )
        self.anova_results['frqi'] = frqi_anova

        # Log ANOVA results
        logger.log_statistical_result('anova_neqr', neqr_anova)
        logger.log_statistical_result('anova_frqi', frqi_anova)

    def _apply_fdr_corrections(self):
        """Apply FDR correction to all p-values"""
        p_columns = ['p_neqr_classical', 'p_frqi_classical', 'p_neqr_frqi']

        for col in p_columns:
            if col in self.results.columns:
                rejected, p_corrected = EnhancedStatisticalAnalysis.apply_fdr_correction(
                    self.results[col].values
                )
                self.results[f'{col}_fdr'] = p_corrected
                self.results[f'{col}_significant_fdr'] = rejected

    def _generate_statistical_summary(self):
        """Generate comprehensive statistical summary"""
        self.statistical_summary = {
            'total_sequence_pairs': len(self.results),
            'total_trials': self.results['n_trials'].sum(),
            'correlations': {
                'neqr_correlation': stats.pearsonr(self.results['hamming'], self.results['neqr_mean']),
                'frqi_correlation': stats.pearsonr(self.results['hamming'], self.results['frqi_mean']),
            },
            'overall_performance': {
                'neqr_mean_error': self.results['neqr_error'].mean(),
                'frqi_mean_error': self.results['frqi_error'].mean(),
                'improvement_percentage': ((self.results['frqi_error'].mean() - self.results['neqr_error'].mean()) / self.results['frqi_error'].mean()) * 100
            },
            'significance_counts': {
                'neqr_vs_classical_significant': sum(self.results['p_neqr_classical_fdr'] < 0.05),
                'frqi_vs_classical_significant': sum(self.results['p_frqi_classical_fdr'] < 0.05),
                'neqr_vs_frqi_significant': sum(self.results['p_neqr_frqi_fdr'] < 0.05),
            },
            'anova_results': self.anova_results
        }

    def plot_clean_correlation_analysis(self):
        """Create clean 2x2 panel correlation analysis"""
        if self.results is None:
            raise ValueError("Run analysis first")

        QuantumDNAVisualizer.plot_clean_correlation_analysis(self.results)

    def plot_performance_by_gc_content(self):
        """Create grouped error bar plots by GC content"""
        if self.results is None:
            raise ValueError("Run analysis first")

        QuantumDNAVisualizer.plot_performance_by_gc_content(self.results)

    def plot_performance_distributions(self):
        """Create box plots and violin plots for distribution comparison"""
        if self.results is None:
            raise ValueError("Run analysis first")

        QuantumDNAVisualizer.plot_performance_distributions(self.results)

    def plot_gc_content_trends(self):
        """Create trend analysis plots with confidence bands"""
        if self.results is None:
            raise ValueError("Run analysis first")

        QuantumDNAVisualizer.plot_gc_content_trends(self.results)

    def save_comprehensive_results(self):
        """Save comprehensive results and summaries"""
        # Save main results
        self.results.to_csv('results/tables/statistical_results_enhanced.csv',
                           index=False, encoding='utf-16')

        # Save statistical summary
        summary_df = pd.DataFrame([self.statistical_summary['overall_performance']])
        summary_df.to_csv('results/tables/statistical_summary_enhanced.csv',
                         index=False, encoding='utf-16')

        # Save ANOVA results
        if self.anova_results:
            anova_df = pd.DataFrame(self.anova_results).T
            anova_df.to_csv('results/tables/anova_results_enhanced.csv',
                           encoding='utf-16')

        # Save GC content breakdown
        gc_breakdown = self.results.groupby('gc_content').agg({
            'neqr_mean': ['mean', 'std', 'count'],
            'frqi_mean': ['mean', 'std', 'count'],
            'neqr_error': ['mean', 'std'],
            'frqi_error': ['mean', 'std']
        }).round(6)

        gc_breakdown.to_csv('results/tables/gc_content_breakdown_enhanced.csv',
                           encoding='utf-16')

        # Save checkpoint
        with open('results/supplementary/checkpoint_statistical_enhanced.pkl', 'wb') as f:
            pickle.dump(self.results, f)

        # Create all new visualization plots
        print("\n📊 Generating Enhanced Visualizations...")
        self.plot_clean_correlation_analysis()
        self.plot_performance_by_gc_content()
        self.plot_performance_distributions()
        self.plot_gc_content_trends()

        # Log performance
        logger.log_performance('statistical_analysis', self.statistical_summary)

        print("\n📊 Enhanced Statistical Analysis Summary:")
        print(f"   - Total sequence pairs: {self.statistical_summary['total_sequence_pairs']}")
        print(f"   - NEQR correlation: {self.statistical_summary['correlations']['neqr_correlation'][0]:.6f}")
        print(f"   - FRQI correlation: {self.statistical_summary['correlations']['frqi_correlation'][0]:.6f}")
        print(f"   - NEQR improvement: {self.statistical_summary['overall_performance']['improvement_percentage']:.2f}%")
        print(f"   - ANOVA NEQR F-stat: {self.anova_results['neqr']['f_statistic']:.3f}")
        print(f"   - ANOVA FRQI F-stat: {self.anova_results['frqi']['f_statistic']:.3f}")

        print("✅ Enhanced statistical analysis complete!")



# # -*- coding: utf-8 -*-
# """
# Enhanced statistical analysis with ANOVA and advanced visualizations
# """

# import numpy as np
# import pandas as pd
# import scipy.stats as stats
# from tqdm import tqdm
# import gc
# from ..config.settings import CONFIG, ANALYSIS_SEEDS
# from ..utils.sequence import EnhancedSequenceAnalyzer
# from ..utils.stats import EnhancedStatisticalAnalysis
# from ..utils.logging import logger
# from ..visualization.plotting import QuantumDNAVisualizer
# import pickle

# class EnhancedStatisticalAnalyzer:
#     """Enhanced statistical analyzer with ANOVA and advanced visualizations"""

#     def __init__(self, runner):
#         self.runner = runner
#         self.results = None
#         self.anova_results = None
#         self.statistical_summary = None

#     def analyze(self, n_sequences=CONFIG['statistical_sequences'],
#                 sequence_length=CONFIG['sequence_length'],
#                 gc_levels=CONFIG['gc_levels'],
#                 n_trials=CONFIG['n_trials']):
#         """Run enhanced statistical analysis"""
#         print(f"\n📈 Enhanced Statistical Analysis...")
#         print(f"   - Sequences per GC level: {n_sequences}")
#         print(f"   - Trials per sequence: {n_trials}")
#         print(f"   - GC levels: {gc_levels}")
#         print(f"   - Total circuits: {len(gc_levels) * n_sequences * n_trials * 2}")

#         # Set seed for reproducibility
#         np.random.seed(ANALYSIS_SEEDS['statistical'])

#         results_data = []
#         gc_grouped_data = {gc: {'neqr': [], 'frqi': [], 'hamming': []} for gc in gc_levels}

#         total_work = len(gc_levels) * n_sequences

#         with tqdm(total=total_work, 
#                 desc="📈 Statistical Analysis", 
#                 ncols=120, 
#                 ascii=False,
#                 position=0,
#                 bar_format='{desc}: {percentage:3.0f}%|{bar:80}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
#             for gc_content in gc_levels:
#                 for seq_idx in range(n_sequences):
#                     # Generate sequences with controlled GC content
#                     seq1 = EnhancedSequenceAnalyzer.generate_random_sequence(
#                         sequence_length, gc_content=gc_content, seed=seq_idx
#                     )
#                     seq2 = EnhancedSequenceAnalyzer.generate_random_sequence(
#                         sequence_length, gc_content=gc_content, seed=seq_idx + 1000
#                     )

#                     # Calculate classical similarity
#                     hamming = EnhancedSequenceAnalyzer.calculate_hamming_similarity(seq1, seq2)

#                     # Run quantum simulations
#                     neqr_trials = self.runner.run_multiple_trials_enhanced(
#                         seq1, seq2, 'neqr', n_trials
#                     )
#                     frqi_trials = self.runner.run_multiple_trials_enhanced(
#                         seq1, seq2, 'frqi', n_trials
#                     )

#                     # Calculate comprehensive statistics
#                     neqr_stats = EnhancedStatisticalAnalysis.calculate_comprehensive_stats(neqr_trials)
#                     frqi_stats = EnhancedStatisticalAnalysis.calculate_comprehensive_stats(frqi_trials)

#                     # Statistical tests
#                     _, p_neqr_classical = stats.ttest_1samp(neqr_trials, hamming)
#                     _, p_frqi_classical = stats.ttest_1samp(frqi_trials, hamming)
#                     _, p_neqr_frqi = stats.ttest_ind(neqr_trials, frqi_trials)

#                     # Store for ANOVA
#                     gc_grouped_data[gc_content]['neqr'].extend(neqr_trials)
#                     gc_grouped_data[gc_content]['frqi'].extend(frqi_trials)
#                     gc_grouped_data[gc_content]['hamming'].append(hamming)

#                     # Store results
#                     results_data.append({
#                         'gc_content': gc_content,
#                         'sequence_pair': seq_idx,
#                         'sequence_length': sequence_length,
#                         'hamming': hamming,

#                         # NEQR comprehensive stats
#                         'neqr_mean': neqr_stats['mean'],
#                         'neqr_std': neqr_stats['std'],
#                         'neqr_median': neqr_stats['median'],
#                         'neqr_ci_lower': neqr_stats['ci_lower'],
#                         'neqr_ci_upper': neqr_stats['ci_upper'],
#                         'neqr_skewness': neqr_stats['skewness'],
#                         'neqr_kurtosis': neqr_stats['kurtosis'],

#                         # FRQI comprehensive stats
#                         'frqi_mean': frqi_stats['mean'],
#                         'frqi_std': frqi_stats['std'],
#                         'frqi_median': frqi_stats['median'],
#                         'frqi_ci_lower': frqi_stats['ci_lower'],
#                         'frqi_ci_upper': frqi_stats['ci_upper'],
#                         'frqi_skewness': frqi_stats['skewness'],
#                         'frqi_kurtosis': frqi_stats['kurtosis'],

#                         # Error metrics
#                         'neqr_error': abs(neqr_stats['mean'] - hamming),
#                         'frqi_error': abs(frqi_stats['mean'] - hamming),

#                         # Statistical tests
#                         'p_neqr_classical': p_neqr_classical,
#                         'p_frqi_classical': p_frqi_classical,
#                         'p_neqr_frqi': p_neqr_frqi,

#                         'n_trials': n_trials
#                     })

#                     pbar.update(1)

#                     # Periodic memory cleanup
#                     if seq_idx % 25 == 0:
#                         gc.collect()

#         self.results = pd.DataFrame(results_data)

#         # Perform ANOVA analysis
#         self._perform_anova_analysis(gc_grouped_data)

#         # Apply FDR correction
#         self._apply_fdr_corrections()

#         # Generate statistical summary
#         self._generate_statistical_summary()

#         # Clear cache
#         self.runner.clear_cache()

#         return self.results

#     def _perform_anova_analysis(self, gc_grouped_data):
#         """Perform ANOVA analysis across GC content levels"""
#         self.anova_results = {}

#         # ANOVA for NEQR performance
#         neqr_anova = EnhancedStatisticalAnalysis.perform_anova_analysis(
#             {f'GC_{gc}': {'neqr': data['neqr']} for gc, data in gc_grouped_data.items()},
#             'neqr'
#         )
#         self.anova_results['neqr'] = neqr_anova

#         # ANOVA for FRQI performance
#         frqi_anova = EnhancedStatisticalAnalysis.perform_anova_analysis(
#             {f'GC_{gc}': {'frqi': data['frqi']} for gc, data in gc_grouped_data.items()},
#             'frqi'
#         )
#         self.anova_results['frqi'] = frqi_anova

#         # Log ANOVA results
#         logger.log_statistical_result('anova_neqr', neqr_anova)
#         logger.log_statistical_result('anova_frqi', frqi_anova)

#     def _apply_fdr_corrections(self):
#         """Apply FDR correction to all p-values"""
#         p_columns = ['p_neqr_classical', 'p_frqi_classical', 'p_neqr_frqi']

#         for col in p_columns:
#             if col in self.results.columns:
#                 rejected, p_corrected = EnhancedStatisticalAnalysis.apply_fdr_correction(
#                     self.results[col].values
#                 )
#                 self.results[f'{col}_fdr'] = p_corrected
#                 self.results[f'{col}_significant_fdr'] = rejected

#     def _generate_statistical_summary(self):
#         """Generate comprehensive statistical summary"""
#         self.statistical_summary = {
#             'total_sequence_pairs': len(self.results),
#             'total_trials': self.results['n_trials'].sum(),
#             'correlations': {
#                 'neqr_correlation': stats.pearsonr(self.results['hamming'], self.results['neqr_mean']),
#                 'frqi_correlation': stats.pearsonr(self.results['hamming'], self.results['frqi_mean']),
#             },
#             'overall_performance': {
#                 'neqr_mean_error': self.results['neqr_error'].mean(),
#                 'frqi_mean_error': self.results['frqi_error'].mean(),
#                 'improvement_percentage': ((self.results['frqi_error'].mean() - self.results['neqr_error'].mean()) / self.results['frqi_error'].mean()) * 100
#             },
#             'significance_counts': {
#                 'neqr_vs_classical_significant': sum(self.results['p_neqr_classical_fdr'] < 0.05),
#                 'frqi_vs_classical_significant': sum(self.results['p_frqi_classical_fdr'] < 0.05),
#                 'neqr_vs_frqi_significant': sum(self.results['p_neqr_frqi_fdr'] < 0.05),
#             },
#             'anova_results': self.anova_results
#         }

#     def plot_correlation_analysis(self):
#         """Create enhanced correlation plot with distinct colors"""
#         if self.results is None:
#             raise ValueError("Run analysis first")

#         QuantumDNAVisualizer.plot_correlation_analysis(self.results)

#     def plot_gc_content_heatmap(self):
#         """Create GC content vs performance heatmap"""
#         if self.results is None:
#             raise ValueError("Run analysis first")

#         QuantumDNAVisualizer.plot_gc_content_heatmap(self.results)

#     def plot_3d_performance_surface(self):
#         """Create 3D surface plot of performance"""
#         if self.results is None:
#             raise ValueError("Run analysis first")

#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D
#         from ..config.settings import CONFIG

#         fig = plt.figure(figsize=(15, 12))

#         # Create subplots for NEQR and FRQI
#         ax1 = fig.add_subplot(121, projection='3d')
#         ax2 = fig.add_subplot(122, projection='3d')

#         # Prepare data
#         gc_levels = sorted(self.results['gc_content'].unique())

#         # Create meshgrid for surface plot
#         X, Y = np.meshgrid(gc_levels, gc_levels)  # GC content vs GC content
#         Z_neqr = np.zeros_like(X)
#         Z_frqi = np.zeros_like(X)

#         # Fill Z values (performance)
#         for i, gc1 in enumerate(gc_levels):
#             for j, gc2 in enumerate(gc_levels):
#                 # For demonstration, use diagonal values (same GC content)
#                 if i == j:
#                     gc_data = self.results[self.results['gc_content'] == gc1]
#                     Z_neqr[i, j] = gc_data['neqr_mean'].mean()
#                     Z_frqi[i, j] = gc_data['frqi_mean'].mean()
#                 else:
#                     # Interpolate for off-diagonal
#                     Z_neqr[i, j] = (Z_neqr[i, i] + Z_neqr[j, j]) / 2
#                     Z_frqi[i, j] = (Z_frqi[i, i] + Z_frqi[j, j]) / 2

#         # Create surface plots
#         surf1 = ax1.plot_surface(X, Y, Z_neqr, cmap='viridis', alpha=0.8)
#         surf2 = ax2.plot_surface(X, Y, Z_frqi, cmap='plasma', alpha=0.8)

#         # Customize axes
#         ax1.set_xlabel('GC Content 1')
#         ax1.set_ylabel('GC Content 2')
#         ax1.set_zlabel('NEQR Performance')
#         ax1.set_title('NEQR Performance Surface')

#         ax2.set_xlabel('GC Content 1')
#         ax2.set_ylabel('GC Content 2')
#         ax2.set_zlabel('FRQI Performance')
#         ax2.set_title('FRQI Performance Surface')

#         # Add colorbars
#         fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
#         fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

#         plt.tight_layout()
#         plt.savefig('results/figures/3d_performance_surface.pdf',
#                    dpi=CONFIG['figure_dpi'], bbox_inches='tight')
#         plt.show()

#     def save_comprehensive_results(self):
#         """Save comprehensive results and summaries"""
#         # Save main results
#         self.results.to_csv('results/tables/statistical_results_enhanced.csv',
#                            index=False, encoding='utf-16')

#         # Save statistical summary
#         summary_df = pd.DataFrame([self.statistical_summary['overall_performance']])
#         summary_df.to_csv('results/tables/statistical_summary_enhanced.csv',
#                          index=False, encoding='utf-16')

#         # Save ANOVA results
#         if self.anova_results:
#             anova_df = pd.DataFrame(self.anova_results).T
#             anova_df.to_csv('results/tables/anova_results_enhanced.csv',
#                            encoding='utf-16')

#         # Save GC content breakdown
#         gc_breakdown = self.results.groupby('gc_content').agg({
#             'neqr_mean': ['mean', 'std', 'count'],
#             'frqi_mean': ['mean', 'std', 'count'],
#             'neqr_error': ['mean', 'std'],
#             'frqi_error': ['mean', 'std']
#         }).round(6)

#         gc_breakdown.to_csv('results/tables/gc_content_breakdown_enhanced.csv',
#                            encoding='utf-16')

#         # Save checkpoint
#         with open('results/supplementary/checkpoint_statistical_enhanced.pkl', 'wb') as f:
#             pickle.dump(self.results, f)

#         # Log performance
#         logger.log_performance('statistical_analysis', self.statistical_summary)

#         print("\n📊 Enhanced Statistical Analysis Summary:")
#         print(f"   - Total sequence pairs: {self.statistical_summary['total_sequence_pairs']}")
#         print(f"   - NEQR correlation: {self.statistical_summary['correlations']['neqr_correlation'][0]:.6f}")
#         print(f"   - FRQI correlation: {self.statistical_summary['correlations']['frqi_correlation'][0]:.6f}")
#         print(f"   - NEQR improvement: {self.statistical_summary['overall_performance']['improvement_percentage']:.2f}%")
#         print(f"   - ANOVA NEQR F-stat: {self.anova_results['neqr']['f_statistic']:.3f}")
#         print(f"   - ANOVA FRQI F-stat: {self.anova_results['frqi']['f_statistic']:.3f}")

#         print("✅ Enhanced statistical analysis complete!")