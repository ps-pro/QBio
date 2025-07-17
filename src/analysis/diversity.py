# -*- coding: utf-8 -*-
"""
Enhanced diversity analysis for quantum DNA similarity
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
from ..config.settings import CONFIG, ANALYSIS_SEEDS
from ..utils.sequence import EnhancedSequenceAnalyzer
from ..utils.stats import EnhancedStatisticalAnalysis
from ..utils.logging import logger
from ..visualization.plotting import QuantumDNAVisualizer
import pickle

class EnhancedDiversityAnalysis:
    """Enhanced diversity analysis with comprehensive test cases"""

    def __init__(self, runner):
        self.runner = runner
        self.results = None
        self.statistical_summary = None

    def generate_comprehensive_test_cases(self, length=CONFIG['sequence_length']):
        """Generate comprehensive test cases for diversity analysis"""
        test_cases = {}

        # Set seed for reproducibility
        np.random.seed(ANALYSIS_SEEDS['diversity'])

        # Basic similarity cases
        base_seq = EnhancedSequenceAnalyzer.generate_random_sequence(length, 0.5)
        test_cases['Identical'] = [(base_seq, base_seq, 'Identical')]
        test_cases['Completely Different'] = [('A' * length, 'G' * length, 'Completely Different')]
        test_cases['Single Mismatch'] = [(base_seq, base_seq[:-1] + 'T', 'Single Mismatch')]
        test_cases['Half Mismatch'] = [(base_seq, base_seq[:length//2] + 'T'*(length-length//2), 'Half Mismatch')]

        # GC content extremes for each level
        for gc in CONFIG['gc_levels']:
            test_cases[f'GC {gc:.1f}'] = [(
                EnhancedSequenceAnalyzer.generate_random_sequence(length, gc),
                EnhancedSequenceAnalyzer.generate_random_sequence(length, gc),
                f'GC {gc:.1f}'
            )]

        # Controlled similarity cases
        ref_seq = EnhancedSequenceAnalyzer.generate_random_sequence(length, 0.5)
        for sim in [0.2, 0.4, 0.6, 0.8]:
            test_cases[f'{int(sim*100)}% Similar'] = [(
                ref_seq,
                EnhancedSequenceAnalyzer.create_controlled_similarity_sequence(ref_seq, sim),
                f'{int(sim*100)}% Similar'
            )]

        # GC content contrast cases
        test_cases['GC Contrast Low-High'] = [(
            EnhancedSequenceAnalyzer.generate_random_sequence(length, 0.1),
            EnhancedSequenceAnalyzer.generate_random_sequence(length, 0.9),
            'GC Contrast Low-High'
        )]

        return test_cases

    def analyze(self, n_trials=CONFIG['diversity_trials']):
        """Run enhanced diversity analysis"""
        print(f"\n🧬 Enhanced Diversity Analysis (n_trials={n_trials})...")

        # Set seed for reproducibility
        np.random.seed(ANALYSIS_SEEDS['diversity'])

        test_cases = self.generate_comprehensive_test_cases()
        results_data = []

        print(f"📊 Total test cases: {len(test_cases)}")
        print(f"🔥 Total circuits: {len(test_cases) * n_trials * 2}")

        with tqdm(total=len(test_cases), 
                desc="Diversity Analysis", 
                ncols=120, 
                ascii=False,
                position=0,
                bar_format='{desc}: {percentage:3.0f}%|{bar:80}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            for category, cases in test_cases.items():
                for seq1, seq2, desc in cases:
                    # Calculate classical similarity
                    hamming = EnhancedSequenceAnalyzer.calculate_hamming_similarity(seq1, seq2)

                    # Calculate sequence complexity
                    complexity1 = EnhancedSequenceAnalyzer.calculate_sequence_complexity(seq1)
                    complexity2 = EnhancedSequenceAnalyzer.calculate_sequence_complexity(seq2)

                    # Run quantum simulations
                    neqr_scores = self.runner.run_multiple_trials_enhanced(
                        seq1, seq2, 'neqr', n_trials
                    )
                    frqi_scores = self.runner.run_multiple_trials_enhanced(
                        seq1, seq2, 'frqi', n_trials
                    )

                    # Calculate comprehensive statistics
                    neqr_stats = EnhancedStatisticalAnalysis.calculate_comprehensive_stats(neqr_scores)
                    frqi_stats = EnhancedStatisticalAnalysis.calculate_comprehensive_stats(frqi_scores)

                    # Statistical tests
                    _, p_neqr_classical = stats.ttest_1samp(neqr_scores, hamming)
                    _, p_frqi_classical = stats.ttest_1samp(frqi_scores, hamming)
                    _, p_neqr_frqi = stats.ttest_ind(neqr_scores, frqi_scores)

                    # Bootstrap confidence intervals
                    neqr_bootstrap = EnhancedStatisticalAnalysis.bootstrap_confidence_interval(neqr_scores)
                    frqi_bootstrap = EnhancedStatisticalAnalysis.bootstrap_confidence_interval(frqi_scores)

                    # Effect sizes
                    neqr_effect = (neqr_stats['mean'] - hamming) / neqr_stats['std'] if neqr_stats['std'] > 0 else 0
                    frqi_effect = (frqi_stats['mean'] - hamming) / frqi_stats['std'] if frqi_stats['std'] > 0 else 0

                    results_data.append({
                        'Description': desc,
                        'Category': category,
                        'Sequence_Length': len(seq1),
                        'GC_Content_Seq1': complexity1['gc_content'],
                        'GC_Content_Seq2': complexity2['gc_content'],
                        'Entropy_Seq1': complexity1['entropy'],
                        'Entropy_Seq2': complexity2['entropy'],
                        'Hamming_Similarity': hamming,

                        # NEQR results
                        'NEQR_Mean': neqr_stats['mean'],
                        'NEQR_Std': neqr_stats['std'],
                        'NEQR_Median': neqr_stats['median'],
                        'NEQR_CI_Lower': neqr_stats['ci_lower'],
                        'NEQR_CI_Upper': neqr_stats['ci_upper'],
                        'NEQR_Bootstrap_Lower': neqr_bootstrap['lower'],
                        'NEQR_Bootstrap_Upper': neqr_bootstrap['upper'],
                        'NEQR_Skewness': neqr_stats['skewness'],
                        'NEQR_Kurtosis': neqr_stats['kurtosis'],

                        # FRQI results
                        'FRQI_Mean': frqi_stats['mean'],
                        'FRQI_Std': frqi_stats['std'],
                        'FRQI_Median': frqi_stats['median'],
                        'FRQI_CI_Lower': frqi_stats['ci_lower'],
                        'FRQI_CI_Upper': frqi_stats['ci_upper'],
                        'FRQI_Bootstrap_Lower': frqi_bootstrap['lower'],
                        'FRQI_Bootstrap_Upper': frqi_bootstrap['upper'],
                        'FRQI_Skewness': frqi_stats['skewness'],
                        'FRQI_Kurtosis': frqi_stats['kurtosis'],

                        # Statistical tests
                        'P_Value_NEQR_Classical': p_neqr_classical,
                        'P_Value_FRQI_Classical': p_frqi_classical,
                        'P_Value_NEQR_FRQI': p_neqr_frqi,
                        'Effect_Size_NEQR': neqr_effect,
                        'Effect_Size_FRQI': frqi_effect,

                        # Error metrics
                        'NEQR_MAE': np.mean(np.abs(neqr_scores - hamming)),
                        'FRQI_MAE': np.mean(np.abs(frqi_scores - hamming)),
                        'NEQR_RMSE': np.sqrt(np.mean((neqr_scores - hamming)**2)),
                        'FRQI_RMSE': np.sqrt(np.mean((frqi_scores - hamming)**2)),

                        'N_Trials': n_trials
                    })

                    pbar.update(1)

        self.results = pd.DataFrame(results_data)

        # Apply FDR correction to p-values
        self._apply_fdr_corrections()

        # Generate statistical summary
        self._generate_statistical_summary()

        # Clear cache
        self.runner.clear_cache()

        return self.results

    def _apply_fdr_corrections(self):
        """Apply FDR correction to all p-values"""
        p_columns = ['P_Value_NEQR_Classical', 'P_Value_FRQI_Classical', 'P_Value_NEQR_FRQI']

        for col in p_columns:
            if col in self.results.columns:
                rejected, p_corrected = EnhancedStatisticalAnalysis.apply_fdr_correction(
                    self.results[col].values
                )
                self.results[f'{col}_FDR'] = p_corrected
                self.results[f'{col}_Significant_FDR'] = rejected

    def _generate_statistical_summary(self):
        """Generate comprehensive statistical summary"""
        self.statistical_summary = {
            'total_test_cases': len(self.results),
            'neqr_overall_performance': {
                'mean_correlation': self.results['NEQR_Mean'].mean(),
                'std_correlation': self.results['NEQR_Mean'].std(),
                'mean_mae': self.results['NEQR_MAE'].mean(),
                'mean_rmse': self.results['NEQR_RMSE'].mean(),
            },
            'frqi_overall_performance': {
                'mean_correlation': self.results['FRQI_Mean'].mean(),
                'std_correlation': self.results['FRQI_Mean'].std(),
                'mean_mae': self.results['FRQI_MAE'].mean(),
                'mean_rmse': self.results['FRQI_RMSE'].mean(),
            },
            'significance_tests': {
                'neqr_vs_classical_significant': sum(self.results['P_Value_NEQR_Classical_FDR'] < 0.05),
                'frqi_vs_classical_significant': sum(self.results['P_Value_FRQI_Classical_FDR'] < 0.05),
                'neqr_vs_frqi_significant': sum(self.results['P_Value_NEQR_FRQI_FDR'] < 0.05),
            },
            'improvement_metrics': {
                'neqr_improvement_over_frqi': ((self.results['FRQI_MAE'].mean() - self.results['NEQR_MAE'].mean()) / self.results['FRQI_MAE'].mean()) * 100,
                'mean_effect_size_neqr': self.results['Effect_Size_NEQR'].mean(),
                'mean_effect_size_frqi': self.results['Effect_Size_FRQI'].mean(),
            }
        }

    def plot_minimalistic_results(self):
        """Create minimalistic diversity comparison plot"""
        if self.results is None:
            raise ValueError("Run analysis first")

        # Use the visualizer
        QuantumDNAVisualizer.plot_diversity_comparison(self.results)

        # Save results
        self.results.to_csv('results/tables/diversity_results_enhanced.csv',
                           index=False, encoding='utf-16')

        # Save checkpoint
        with open('results/supplementary/checkpoint_diversity_enhanced.pkl', 'wb') as f:
            pickle.dump(self.results, f)

        # Log performance
        logger.log_performance('diversity_analysis', self.statistical_summary)

        print("\n📊 Enhanced Diversity Analysis Summary:")
        print(f"   - Total test cases: {self.statistical_summary['total_test_cases']}")
        print(f"   - NEQR mean performance: {self.statistical_summary['neqr_overall_performance']['mean_correlation']:.6f}")
        print(f"   - FRQI mean performance: {self.statistical_summary['frqi_overall_performance']['mean_correlation']:.6f}")
        print(f"   - NEQR improvement: {self.statistical_summary['improvement_metrics']['neqr_improvement_over_frqi']:.2f}%")
        print(f"   - Significant differences (FDR): {self.statistical_summary['significance_tests']['neqr_vs_frqi_significant']}/{self.statistical_summary['total_test_cases']}")

        print("✅ Enhanced diversity analysis complete!")