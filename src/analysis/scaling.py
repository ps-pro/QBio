# -*- coding: utf-8 -*-
"""
Enhanced scaling analysis with comprehensive metrics
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
from ..core.encoders import NEQREncoder, FRQIEncoder
import pickle

class EnhancedScalingAnalysis:
    """Enhanced scaling analysis with comprehensive metrics"""

    def __init__(self, runner):
        self.runner = runner
        self.results = None
        self.scaling_summary = None

    def analyze(self, min_length=12, max_length=CONFIG['max_scaling_length'],
                step=CONFIG['scaling_step'], n_sequences=CONFIG['scaling_sequences'],
                n_trials=20):
        """Run enhanced scaling analysis"""
        print(f"\n📏 Enhanced Scaling Analysis...")
        print(f"   - Length range: {min_length}-{max_length} (step {step})")
        print(f"   - Sequences per length: {n_sequences}")
        print(f"   - Trials per sequence: {n_trials}")

        # Set seed for reproducibility
        np.random.seed(ANALYSIS_SEEDS['scaling'])

        lengths = list(range(min_length, max_length + 1, step))
        similarity_targets = [0.2, 0.5, 0.8, 1.0]
        gc_levels = [0.3, 0.5, 0.7]

        total_combinations = len(lengths) * len(similarity_targets) * len(gc_levels) * n_sequences
        print(f"   - Total combinations: {total_combinations}")
        print(f"   - Total circuits: {total_combinations * n_trials * 2}")

        results_data = []

        with tqdm(total=total_combinations, 
                desc="Scaling Analysis", 
                ncols=120, 
                ascii=False,
                position=0,
                bar_format='{desc}: {percentage:3.0f}%|{bar:80}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for length in lengths:
                for similarity_target in similarity_targets:
                    for gc_content in gc_levels:
                        for seq_idx in range(n_sequences):
                            # Generate sequences with controlled parameters
                            seq1 = EnhancedSequenceAnalyzer.generate_random_sequence(
                                length, gc_content, seed=seq_idx
                            )

                            if similarity_target == 1.0:
                                seq2 = seq1  # Perfect similarity
                            else:
                                seq2 = EnhancedSequenceAnalyzer.create_controlled_similarity_sequence(
                                    seq1, similarity_target, seed=seq_idx + 1000
                                )

                            # Calculate actual similarity
                            hamming = EnhancedSequenceAnalyzer.calculate_hamming_similarity(seq1, seq2)

                            # Run quantum simulations
                            neqr_trials = self.runner.run_multiple_trials_enhanced(
                                seq1, seq2, 'neqr', n_trials
                            )
                            frqi_trials = self.runner.run_multiple_trials_enhanced(
                                seq1, seq2, 'frqi', n_trials
                            )

                            # Create circuits to measure complexity
                            neqr_circuit = NEQREncoder().create_swap_test_circuit(seq1, seq2)
                            frqi_circuit = FRQIEncoder().create_comparison_circuit(seq1, seq2)

                            # Calculate comprehensive statistics
                            neqr_stats = EnhancedStatisticalAnalysis.calculate_comprehensive_stats(neqr_trials)
                            frqi_stats = EnhancedStatisticalAnalysis.calculate_comprehensive_stats(frqi_trials)

                            # Circuit complexity metrics
                            neqr_ops = neqr_circuit.count_ops()
                            frqi_ops = frqi_circuit.count_ops()

                            results_data.append({
                                'length': length,
                                'target_similarity': similarity_target,
                                'actual_similarity': hamming,
                                'gc_content': gc_content,
                                'sequence_pair': seq_idx,

                                # NEQR performance
                                'neqr_mean': neqr_stats['mean'],
                                'neqr_std': neqr_stats['std'],
                                'neqr_ci_lower': neqr_stats['ci_lower'],
                                'neqr_ci_upper': neqr_stats['ci_upper'],
                                'neqr_error': abs(neqr_stats['mean'] - hamming),

                                # FRQI performance
                                'frqi_mean': frqi_stats['mean'],
                                'frqi_std': frqi_stats['std'],
                                'frqi_ci_lower': frqi_stats['ci_lower'],
                                'frqi_ci_upper': frqi_stats['ci_upper'],
                                'frqi_error': abs(frqi_stats['mean'] - hamming),

                                # Circuit complexity
                                'neqr_circuit_depth': neqr_circuit.depth(),
                                'frqi_circuit_depth': frqi_circuit.depth(),
                                'neqr_gate_count': sum(neqr_ops.values()),
                                'frqi_gate_count': sum(frqi_ops.values()),
                                'neqr_qubit_count': neqr_circuit.num_qubits,
                                'frqi_qubit_count': frqi_circuit.num_qubits,

                                # Gate type breakdown
                                'neqr_h_gates': neqr_ops.get('h', 0),
                                'neqr_cx_gates': neqr_ops.get('cx', 0),
                                'neqr_mcx_gates': neqr_ops.get('mcx', 0),
                                'frqi_h_gates': frqi_ops.get('h', 0),
                                'frqi_cx_gates': frqi_ops.get('cx', 0),
                                'frqi_mcry_gates': frqi_ops.get('mcry', 0),

                                'n_trials': n_trials
                            })

                            pbar.update(1)

                            # Memory management
                            if len(results_data) % 100 == 0:
                                gc.collect()

        self.results = pd.DataFrame(results_data)

        # Generate scaling summary
        self._generate_scaling_summary()

        # Clear cache
        self.runner.clear_cache()

        return self.results

    def _generate_scaling_summary(self):
        """Generate comprehensive scaling summary"""
        length_groups = self.results.groupby('length')

        scaling_trends = {}
        for length in sorted(self.results['length'].unique()):
            length_data = self.results[self.results['length'] == length]
            scaling_trends[length] = {
                'neqr_mean_performance': length_data['neqr_mean'].mean(),
                'frqi_mean_performance': length_data['frqi_mean'].mean(),
                'neqr_mean_error': length_data['neqr_error'].mean(),
                'frqi_mean_error': length_data['frqi_error'].mean(),
                'neqr_avg_depth': length_data['neqr_circuit_depth'].mean(),
                'frqi_avg_depth': length_data['frqi_circuit_depth'].mean(),
                'neqr_avg_gates': length_data['neqr_gate_count'].mean(),
                'frqi_avg_gates': length_data['frqi_gate_count'].mean(),
                'neqr_avg_qubits': length_data['neqr_qubit_count'].mean(),
                'frqi_avg_qubits': length_data['frqi_qubit_count'].mean(),
            }

        # Calculate scaling coefficients
        lengths = sorted(self.results['length'].unique())
        neqr_errors = [scaling_trends[l]['neqr_mean_error'] for l in lengths]
        frqi_errors = [scaling_trends[l]['frqi_mean_error'] for l in lengths]

        self.scaling_summary = {
            'length_range': (min(lengths), max(lengths)),
            'total_data_points': len(self.results),
            'scaling_trends': scaling_trends,
            'overall_performance': {
                'neqr_mean_error': np.mean(neqr_errors),
                'frqi_mean_error': np.mean(frqi_errors),
                'neqr_error_trend': np.polyfit(lengths, neqr_errors, 1)[0],  # Linear coefficient
                'frqi_error_trend': np.polyfit(lengths, frqi_errors, 1)[0],
            },
            'complexity_analysis': {
                'neqr_depth_scaling': self._calculate_complexity_scaling('neqr_circuit_depth'),
                'frqi_depth_scaling': self._calculate_complexity_scaling('frqi_circuit_depth'),
                'neqr_gate_scaling': self._calculate_complexity_scaling('neqr_gate_count'),
                'frqi_gate_scaling': self._calculate_complexity_scaling('frqi_gate_count'),
            }
        }

    def _calculate_complexity_scaling(self, metric):
        """Calculate how complexity scales with length"""
        length_groups = self.results.groupby('length')[metric].mean()
        lengths = length_groups.index.values
        values = length_groups.values

        # Fit exponential model: y = a * exp(b * x)
        log_values = np.log(values + 1)  # Add 1 to avoid log(0)
        coeffs = np.polyfit(lengths, log_values, 1)

        return {
            'exponential_coefficient': coeffs[0],
            'base_complexity': np.exp(coeffs[1]),
            'r_squared': stats.pearsonr(lengths, log_values)[0]**2
        }

    def plot_scaling_results(self):
        """Create comprehensive scaling analysis plots"""
        if self.results is None:
            raise ValueError("Run analysis first")

        QuantumDNAVisualizer.plot_scaling_results(self.results)

    def save_comprehensive_results(self):
        """Save comprehensive scaling results"""
        # Save main results
        self.results.to_csv('results/tables/scaling_results_enhanced.csv',
                           index=False, encoding='utf-16')

        # Save scaling summary
        summary_df = pd.DataFrame([self.scaling_summary['overall_performance']])
        summary_df.to_csv('results/tables/scaling_summary_enhanced.csv',
                         index=False, encoding='utf-16')

        # Save complexity analysis
        complexity_df = pd.DataFrame(self.scaling_summary['complexity_analysis']).T
        complexity_df.to_csv('results/tables/complexity_analysis_enhanced.csv',
                           encoding='utf-16')

        # Save length breakdown
        length_breakdown = self.results.groupby('length').agg({
            'neqr_error': ['mean', 'std'],
            'frqi_error': ['mean', 'std'],
            'neqr_circuit_depth': ['mean', 'std'],
            'frqi_circuit_depth': ['mean', 'std'],
            'neqr_gate_count': ['mean', 'std'],
            'frqi_gate_count': ['mean', 'std']
        }).round(6)

        length_breakdown.to_csv('results/tables/length_breakdown_enhanced.csv',
                               encoding='utf-16')

        # Save checkpoint
        with open('results/supplementary/checkpoint_scaling_enhanced.pkl', 'wb') as f:
            pickle.dump(self.results, f)

        # Log performance
        logger.log_performance('scaling_analysis', self.scaling_summary)

        max_length = max(self.results['length'])
        max_neqr_gates = self.scaling_summary['scaling_trends'][max_length]['neqr_avg_gates']
        max_frqi_gates = self.scaling_summary['scaling_trends'][max_length]['frqi_avg_gates']
        gate_ratio = max_frqi_gates / max_neqr_gates if max_neqr_gates > 0 else 0

        print("\n📊 Enhanced Scaling Analysis Summary:")
        print(f"   - Length range: {self.scaling_summary['length_range'][0]}-{self.scaling_summary['length_range'][1]}")
        print(f"   - Total data points: {self.scaling_summary['total_data_points']}")
        print(f"   - NEQR error trend: {self.scaling_summary['overall_performance']['neqr_error_trend']:.6f}")
        print(f"   - FRQI error trend: {self.scaling_summary['overall_performance']['frqi_error_trend']:.6f}")
        print(f"   - Gate ratio (FRQI/NEQR) at max length: {gate_ratio:.1f}×")
        print(f"   - NEQR depth scaling: {self.scaling_summary['complexity_analysis']['neqr_depth_scaling']['exponential_coefficient']:.3f}")
        print(f"   - FRQI depth scaling: {self.scaling_summary['complexity_analysis']['frqi_depth_scaling']['exponential_coefficient']:.3f}")

        print("✅ Enhanced scaling analysis complete!")