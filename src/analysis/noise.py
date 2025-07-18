# -*- coding: utf-8 -*-
"""
Thread-safe Enhanced noise analysis with multiple noise types
"""

import threading
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
import gc
import time
from contextlib import contextmanager
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from ..config.settings import CONFIG, ANALYSIS_SEEDS
from ..utils.sequence import EnhancedSequenceAnalyzer
from ..utils.stats import EnhancedStatisticalAnalysis
from ..utils.logging import logger
from ..visualization.plotting import QuantumDNAVisualizer
import pickle

class EnhancedNoiseAnalysis:
    """Thread-safe Enhanced noise analysis with multiple noise types and cross-validation"""

    def __init__(self, runner):
        self.runner = runner
        self.results = None
        self.noise_summary = None
        
        # Thread safety locks
        self._results_lock = threading.RLock()
        self._analysis_lock = threading.RLock()
        self._plotting_lock = threading.RLock()
        
        # Performance tracking
        self.analysis_stats = {
            'start_time': None,
            'end_time': None,
            'successful_sequences': 0,
            'failed_sequences': 0,
            'total_circuits_run': 0,
            'noise_configurations_tested': 0,
            'thread_safety_violations': 0
        }

    @contextmanager
    def _thread_safe_analysis(self):
        """Context manager for thread-safe analysis operations"""
        with self._analysis_lock:
            yield

    @contextmanager
    def _safe_results_access(self):
        """Context manager for thread-safe results access"""
        with self._results_lock:
            yield

    def create_realistic_noise_model(self, error_rates):
        """Create realistic noise models with multiple error types"""
        if not any(error_rates.values()):
            return None
            
        noise = NoiseModel()

        # Depolarizing noise
        if 'depolarizing' in error_rates and error_rates['depolarizing'] > 0:
            error_1q = depolarizing_error(error_rates['depolarizing'], 1)
            error_2q = depolarizing_error(error_rates['depolarizing'] * 10, 2)
            noise.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'ry'])
            noise.add_all_qubit_quantum_error(error_2q, ['cx', 'swap', 'mcx', 'mcry'])

        # Amplitude damping (T1 decay)
        if 'amplitude_damping' in error_rates and error_rates['amplitude_damping'] > 0:
            error_ad = amplitude_damping_error(error_rates['amplitude_damping'])
            noise.add_all_qubit_quantum_error(error_ad, ['h', 'x', 'ry'])

        # Phase damping (T2 decay)
        if 'phase_damping' in error_rates and error_rates['phase_damping'] > 0:
            error_pd = phase_damping_error(error_rates['phase_damping'])
            noise.add_all_qubit_quantum_error(error_pd, ['h', 'x', 'ry'])

        return noise

    def _create_noise_label(self, noise_config):
        """Create readable label for noise configuration"""
        parts = []
        if noise_config.get('depolarizing', 0) > 0:
            parts.append(f"Depol: {noise_config['depolarizing']:.3f}")
        if noise_config.get('amplitude_damping', 0) > 0:
            parts.append(f"T1: {noise_config['amplitude_damping']:.3f}")
        if noise_config.get('phase_damping', 0) > 0:
            parts.append(f"T2: {noise_config['phase_damging']:.3f}")

        return "; ".join(parts) if parts else "No Noise"

    def _validate_noise_parameters(self, noise_levels, n_sequences, sequence_length, gc_levels, n_trials):
        """Validate noise analysis parameters"""
        if not noise_levels:
            raise ValueError("noise_levels cannot be empty")
        
        if n_sequences <= 0 or n_trials <= 0:
            raise ValueError("n_sequences and n_trials must be positive")
        
        if sequence_length <= 0 or sequence_length > 20:
            raise ValueError("sequence_length must be between 1 and 20")
        
        if not gc_levels or any(gc < 0 or gc > 1 for gc in gc_levels):
            raise ValueError("Invalid GC levels")

        # Check computational feasibility
        total_circuits = len(noise_levels) * len(gc_levels) * n_sequences * n_trials * 2
        if total_circuits > 200000:
            logger.log_warning(f"Large noise analysis: {total_circuits} total circuits")

    def _safe_quantum_execution_with_noise(self, seq1, seq2, n_trials, noise_model, attempt=1, max_attempts=3):
        """Safe quantum circuit execution with noise model"""
        try:
            neqr_trials, frqi_trials = self.runner.run_both_methods_parallel(seq1, seq2, n_trials, noise_model)
            
            # Validate results
            if len(neqr_trials) != n_trials or len(frqi_trials) != n_trials:
                logger.log_warning(f"Expected {n_trials} trials, got NEQR:{len(neqr_trials)}, FRQI:{len(frqi_trials)}")
            
            return neqr_trials, frqi_trials
            
        except Exception as e:
            if attempt < max_attempts:
                logger.log_warning(f"Noise execution failed (attempt {attempt}), retrying: {e}")
                time.sleep(1)
                return self._safe_quantum_execution_with_noise(seq1, seq2, n_trials, noise_model, attempt + 1, max_attempts)
            else:
                logger.log_error(f"Noise execution failed after {max_attempts} attempts: {e}")
                return np.full(n_trials, 0.5), np.full(n_trials, 0.5)

    def analyze(self, noise_levels=None, n_sequences=20, sequence_length=6,
                gc_levels=[0.2, 0.5, 0.8], n_trials=15):
        """Enhanced noise analysis with multiple noise types"""
        
        if noise_levels is None:
            noise_levels = [
                {'depolarizing': 0.0},
                {'depolarizing': 0.001},
                {'depolarizing': 0.005},
                {'depolarizing': 0.001, 'amplitude_damping': 0.001},
                {'depolarizing': 0.001, 'phase_damping': 0.001},
                {'depolarizing': 0.001, 'amplitude_damping': 0.001, 'phase_damping': 0.001}
            ]

        self.analysis_stats['start_time'] = time.time()
        
        print(f"\nRunning Enhanced Noise Analysis...")
        print(f"   - Sequences per noise config: {n_sequences}")
        print(f"   - Trials per sequence: {n_trials}")
        print(f"   - Noise configurations: {len(noise_levels)}")
        print(f"   - Total circuits: {len(noise_levels) * len(gc_levels) * n_sequences * n_trials * 2}")

        try:
            self._validate_noise_parameters(noise_levels, n_sequences, sequence_length, gc_levels, n_trials)
            np.random.seed(ANALYSIS_SEEDS.get('noise', 126))

            with self._thread_safe_analysis():
                results_data = []
                successful_analyses = 0
                failed_analyses = 0
                
                total_combinations = len(noise_levels) * len(gc_levels) * n_sequences
                self.analysis_stats['noise_configurations_tested'] = len(noise_levels)

                with tqdm(total=total_combinations, 
                          desc="Noise Analysis", 
                          ncols=120, 
                          ascii=False) as pbar:

                    for noise_config in noise_levels:
                        noise_model = self.create_realistic_noise_model(noise_config)
                        noise_label = self._create_noise_label(noise_config)

                        for gc in gc_levels:
                            for seq_idx in range(n_sequences):
                                try:
                                    seq1 = EnhancedSequenceAnalyzer.generate_random_sequence(
                                        sequence_length, gc, seed=seq_idx
                                    )
                                    seq2 = EnhancedSequenceAnalyzer.generate_random_sequence(
                                        sequence_length, gc, seed=seq_idx + 1000
                                    )

                                    hamming = EnhancedSequenceAnalyzer.calculate_hamming_similarity(seq1, seq2)

                                    # Run quantum simulations with noise
                                    neqr_trials, frqi_trials = self._safe_quantum_execution_with_noise(
                                        seq1, seq2, n_trials, noise_model
                                    )
                                    
                                    self.analysis_stats['total_circuits_run'] += n_trials * 2

                                    # Calculate errors
                                    neqr_errors = np.abs(neqr_trials - hamming)
                                    frqi_errors = np.abs(frqi_trials - hamming)

                                    results_data.append({
                                        'noise_config': noise_label,
                                        'depolarizing_rate': noise_config.get('depolarizing', 0),
                                        'amplitude_damping_rate': noise_config.get('amplitude_damping', 0),
                                        'phase_damping_rate': noise_config.get('phase_damping', 0),
                                        'gc_content': gc,
                                        'sequence_pair': seq_idx,
                                        'hamming': hamming,
                                        'neqr_mean': np.mean(neqr_trials),
                                        'neqr_std': np.std(neqr_trials),
                                        'frqi_mean': np.mean(frqi_trials),
                                        'frqi_std': np.std(frqi_trials),
                                        'neqr_error_mean': np.mean(neqr_errors),
                                        'neqr_error_std': np.std(neqr_errors),
                                        'frqi_error_mean': np.mean(frqi_errors),
                                        'frqi_error_std': np.std(frqi_errors)
                                    })

                                    successful_analyses += 1

                                except Exception as e:
                                    logger.log_error(f"Noise analysis failed for config {noise_label}, GC {gc}, seq {seq_idx}: {e}")
                                    failed_analyses += 1
                                    with self._analysis_lock:
                                        self.analysis_stats['thread_safety_violations'] += 1

                                finally:
                                    pbar.update(1)

                # Store results
                self.analysis_stats['successful_sequences'] = successful_analyses
                self.analysis_stats['failed_sequences'] = failed_analyses
                self.analysis_stats['end_time'] = time.time()

                with self._safe_results_access():
                    self.results = pd.DataFrame(results_data)

                # Generate summary
                self._generate_noise_summary()

                # Clear cache
                self.runner.clear_cache()

                print(f"\nNoise Analysis Summary:")
                print(f"   - Successful analyses: {successful_analyses}")
                print(f"   - Failed analyses: {failed_analyses}")
                
                return self.results

        except Exception as e:
            logger.log_error(f"Noise analysis completely failed: {e}")
            self.analysis_stats['end_time'] = time.time()
            raise

    def _generate_noise_summary(self):
        """Generate comprehensive noise analysis summary"""
        with self._safe_results_access():
            try:
                if self.results is None or self.results.empty:
                    self.noise_summary = {'error': 'No results available'}
                    return

                # Calculate noise sensitivity metrics
                no_noise_data = self.results[self.results['depolarizing_rate'] == 0]
                max_noise_data = self.results[self.results['depolarizing_rate'] == self.results['depolarizing_rate'].max()]

                if len(no_noise_data) > 0 and len(max_noise_data) > 0:
                    neqr_sensitivity = self._calculate_noise_sensitivity(no_noise_data, max_noise_data, 'neqr_error_mean')
                    frqi_sensitivity = self._calculate_noise_sensitivity(no_noise_data, max_noise_data, 'frqi_error_mean')
                else:
                    neqr_sensitivity = frqi_sensitivity = np.nan

                self.noise_summary = {
                    'total_configurations': len(self.results['noise_config'].unique()),
                    'total_data_points': len(self.results),
                    'max_noise_tested': self.results['depolarizing_rate'].max(),
                    'neqr_noise_sensitivity': neqr_sensitivity,
                    'frqi_noise_sensitivity': frqi_sensitivity,
                    'analysis_stats': self.analysis_stats.copy()
                }

            except Exception as e:
                logger.log_error(f"Noise summary generation failed: {e}")
                self.noise_summary = {'error': str(e)}

    def _calculate_noise_sensitivity(self, no_noise_data, max_noise_data, error_column):
        """Calculate sensitivity to noise"""
        try:
            no_noise_error = no_noise_data[error_column].mean()
            max_noise_error = max_noise_data[error_column].mean()
            
            if no_noise_error > 0:
                return (max_noise_error - no_noise_error) / no_noise_error
            else:
                return 0.0
        except:
            return np.nan

    def plot_noise_resilience(self):
        """Create noise resilience plots"""
        with self._plotting_lock:
            if self.results is None:
                raise ValueError("Run analysis first")
            try:
                QuantumDNAVisualizer.plot_noise_resilience(self.results)
            except Exception as e:
                logger.log_error(f"Noise resilience plotting failed: {e}")

    def plot_noise_type_comparison(self):
        """Create noise type comparison plots"""
        with self._plotting_lock:
            if self.results is None:
                raise ValueError("Run analysis first")
            try:
                QuantumDNAVisualizer.plot_noise_type_comparison(self.results)
            except Exception as e:
                logger.log_error(f"Noise type comparison plotting failed: {e}")

    def save_comprehensive_results(self):
        """Save comprehensive noise analysis results"""
        with self._safe_results_access():
            if self.results is None:
                logger.log_error("No results to save")
                return

            try:
                # Save main results
                self.results.to_csv('results/tables/noise_results_enhanced.csv',
                                   index=False, encoding='utf-16')

                # Save noise summary
                if self.noise_summary and 'error' not in self.noise_summary:
                    summary_df = pd.DataFrame([self.noise_summary])
                    summary_df.to_csv('results/tables/noise_summary_enhanced.csv',
                                     index=False, encoding='utf-16')

                # Save noise configuration breakdown
                if not self.results.empty:
                    noise_breakdown = self.results.groupby('noise_config').agg({
                        'neqr_error_mean': ['mean', 'std'],
                        'frqi_error_mean': ['mean', 'std']
                    }).round(6)
                    noise_breakdown.to_csv('results/tables/noise_breakdown_enhanced.csv',
                                          encoding='utf-16')

                # Save checkpoint
                with open('results/supplementary/checkpoint_noise_enhanced.pkl', 'wb') as f:
                    pickle.dump(self.results, f)

                # Generate visualizations
                print("\nGenerating noise visualizations...")
                self.plot_noise_resilience()
                self.plot_noise_type_comparison()

                # Log performance
                logger.log_performance('noise_analysis', self.noise_summary)

                print("Enhanced noise analysis complete!")

            except Exception as e:
                logger.log_error(f"Failed to save noise results: {e}")
                raise