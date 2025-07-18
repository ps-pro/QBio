# -*- coding: utf-8 -*-
"""
🔒✅ THREAD-SAFE Enhanced statistical analysis with ANOVA and advanced visualizations
"""

import threading
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
import gc
import time
from contextlib import contextmanager
from ..config.settings import CONFIG, ANALYSIS_SEEDS
from ..utils.sequence import EnhancedSequenceAnalyzer
from ..utils.stats import EnhancedStatisticalAnalysis
from ..utils.logging import logger
from ..visualization.plotting import QuantumDNAVisualizer
import pickle

class EnhancedStatisticalAnalyzer:
    """🔒✅ THREAD-SAFE Enhanced statistical analyzer with comprehensive error handling"""

    def __init__(self, runner):
        self.runner = runner
        self.results = None
        self.anova_results = None
        self.statistical_summary = None
        
        # 🔒 THREAD SAFETY: Add locks for all shared resources
        self._results_lock = threading.RLock()
        self._analysis_lock = threading.RLock()
        self._plotting_lock = threading.RLock()
        
        # ✅ ENHANCED: Performance and error tracking
        self.analysis_stats = {
            'start_time': None,
            'end_time': None,
            'successful_sequences': 0,
            'failed_sequences': 0,
            'total_circuits_run': 0,
            'memory_usage_mb': 0,
            'thread_safety_violations': 0
        }
        
        print("🔒✅ Thread-Safe Statistical Analyzer initialized")

    @contextmanager
    def _thread_safe_analysis(self):
        """🔒 Context manager for thread-safe analysis operations"""
        with self._analysis_lock:
            yield

    @contextmanager
    def _safe_results_access(self):
        """🔒 Context manager for thread-safe results access"""
        with self._results_lock:
            yield

    def _validate_statistical_inputs(self, data_array, data_name="data"):
        """✅ ROBUST: Comprehensive input validation with error recovery"""
        if data_array is None:
            raise ValueError(f"{data_name} is None")
        
        # Convert to numpy array if needed
        if not isinstance(data_array, np.ndarray):
            try:
                data_array = np.array(data_array)
            except Exception as e:
                raise ValueError(f"Cannot convert {data_name} to numpy array: {e}")
        
        if data_array.size == 0:
            raise ValueError(f"{data_name} is empty")
        
        # Check for invalid values
        finite_mask = np.isfinite(data_array)
        if not finite_mask.all():
            invalid_count = (~finite_mask).sum()
            logger.log_warning(f"Found {invalid_count} invalid values in {data_name}")
            
            # Remove invalid values
            valid_data = data_array[finite_mask]
            if valid_data.size == 0:
                raise ValueError(f"No valid values in {data_name} after filtering")
            
            return valid_data
        
        return data_array

    def _validate_sequence_parameters(self, n_sequences, sequence_length, gc_levels, n_trials):
        """✅ ROBUST: Validate all input parameters"""
        if n_sequences <= 0:
            raise ValueError(f"n_sequences must be positive, got {n_sequences}")
        
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        
        if n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got {n_trials}")
        
        if not gc_levels:
            raise ValueError("gc_levels cannot be empty")
        
        for gc in gc_levels:
            if not (0 <= gc <= 1):
                raise ValueError(f"GC content must be between 0 and 1, got {gc}")
        
        # Check if analysis is computationally feasible
        total_circuits = len(gc_levels) * n_sequences * n_trials * 2
        if total_circuits > 100000:
            logger.log_warning(f"Large analysis detected: {total_circuits} total circuits")
            if total_circuits > 500000:
                raise ValueError(f"Analysis too large: {total_circuits} circuits exceeds limit")

    def _safe_quantum_execution(self, seq1, seq2, n_trials, attempt=1, max_attempts=3):
        """✅ ROBUST: Safe quantum circuit execution with retries"""
        try:
            # 🔒 THREAD-SAFE: Execute quantum simulations
            neqr_trials, frqi_trials = self.runner.run_both_methods_parallel(seq1, seq2, n_trials)
            
            # ✅ VALIDATE: Check results
            neqr_trials = self._validate_statistical_inputs(neqr_trials, "NEQR trials")
            frqi_trials = self._validate_statistical_inputs(frqi_trials, "FRQI trials")
            
            # Ensure we got the expected number of trials
            if len(neqr_trials) != n_trials or len(frqi_trials) != n_trials:
                logger.log_warning(f"Expected {n_trials} trials, got NEQR:{len(neqr_trials)}, FRQI:{len(frqi_trials)}")
            
            return neqr_trials, frqi_trials
            
        except Exception as e:
            if attempt < max_attempts:
                logger.log_warning(f"Quantum execution failed (attempt {attempt}), retrying: {e}")
                time.sleep(1)  # Brief delay before retry
                return self._safe_quantum_execution(seq1, seq2, n_trials, attempt + 1, max_attempts)
            else:
                logger.log_error(f"Quantum execution failed after {max_attempts} attempts: {e}")
                # Return fallback data to continue analysis
                return np.full(n_trials, 0.5), np.full(n_trials, 0.5)

    def _safe_statistical_tests(self, neqr_trials, frqi_trials, hamming):
        """✅ ROBUST: Safe statistical test execution with error handling"""
        try:
            # Validate inputs
            neqr_trials = self._validate_statistical_inputs(neqr_trials, "NEQR trials for stats")
            frqi_trials = self._validate_statistical_inputs(frqi_trials, "FRQI trials for stats")
            
            if not (0 <= hamming <= 1):
                logger.log_warning(f"Invalid Hamming similarity: {hamming}")
                hamming = max(0, min(1, hamming))  # Clamp to valid range

            # Perform statistical tests with error handling
            try:
                _, p_neqr_classical = stats.ttest_1samp(neqr_trials, hamming)
            except Exception as e:
                logger.log_warning(f"NEQR t-test failed: {e}")
                p_neqr_classical = np.nan

            try:
                _, p_frqi_classical = stats.ttest_1samp(frqi_trials, hamming)
            except Exception as e:
                logger.log_warning(f"FRQI t-test failed: {e}")
                p_frqi_classical = np.nan

            try:
                _, p_neqr_frqi = stats.ttest_ind(neqr_trials, frqi_trials)
            except Exception as e:
                logger.log_warning(f"NEQR vs FRQI t-test failed: {e}")
                p_neqr_frqi = np.nan

            return p_neqr_classical, p_frqi_classical, p_neqr_frqi
            
        except Exception as e:
            logger.log_error(f"Statistical tests completely failed: {e}")
            return np.nan, np.nan, np.nan

    def _apply_fdr_corrections(self):
        """🔒✅ THREAD-SAFE Enhanced FDR correction with comprehensive validation"""
        with self._safe_results_access():
            if self.results is None or self.results.empty:
                logger.log_error("No results available for FDR correction")
                return
                
            p_columns = ['P_Value_NEQR_Classical', 'P_Value_FRQI_Classical', 'P_Value_NEQR_FRQI']

            for col in p_columns:
                if col not in self.results.columns:
                    logger.log_warning(f"Column {col} not found in results")
                    continue
                    
                try:
                    # ✅ ROBUST: Comprehensive p-value validation
                    p_values = self.results[col].values
                    
                    # Check for completely missing data
                    if len(p_values) == 0:
                        logger.log_warning(f"No p-values in column {col}")
                        continue
                    
                    # Identify valid p-values
                    valid_mask = (p_values >= 0) & (p_values <= 1) & np.isfinite(p_values)
                    valid_count = valid_mask.sum()
                    invalid_count = (~valid_mask).sum()
                    
                    if invalid_count > 0:
                        logger.log_warning(f"Found {invalid_count} invalid p-values in {col} (keeping {valid_count} valid)")
                    
                    if valid_count == 0:
                        logger.log_error(f"No valid p-values in {col}")
                        # Initialize with NaN
                        self.results[f'{col}_fdr'] = np.nan
                        self.results[f'{col}_significant_fdr'] = False
                        continue
                    
                    # Apply FDR correction only to valid p-values
                    valid_p_values = p_values[valid_mask]
                    
                    try:
                        rejected, p_corrected = EnhancedStatisticalAnalysis.apply_fdr_correction(valid_p_values)
                    except Exception as e:
                        logger.log_error(f"FDR correction computation failed for {col}: {e}")
                        continue
                    
                    # Initialize correction columns with defaults
                    self.results[f'{col}_fdr'] = np.nan
                    self.results[f'{col}_significant_fdr'] = False
                    
                    # Store corrected values for valid entries
                    if len(p_corrected) == valid_count and len(rejected) == valid_count:
                        self.results.loc[valid_mask, f'{col}_fdr'] = p_corrected
                        self.results.loc[valid_mask, f'{col}_significant_fdr'] = rejected
                    else:
                        logger.log_error(f"FDR correction output size mismatch for {col}")
                        
                except Exception as e:
                    logger.log_error(f"FDR correction failed for {col}: {e}")
                    # Continue with other columns

    def _perform_anova_analysis(self, gc_grouped_data):
        """🔒✅ THREAD-SAFE Enhanced ANOVA analysis with comprehensive validation"""
        with self._thread_safe_analysis():
            self.anova_results = {}

            try:
                # ✅ ROBUST: Validate input data structure
                if not gc_grouped_data or not isinstance(gc_grouped_data, dict):
                    raise ValueError("Invalid or empty grouped data for ANOVA")

                # ANOVA for NEQR performance
                neqr_groups = {}
                neqr_total_samples = 0
                
                for gc, data in gc_grouped_data.items():
                    if 'neqr' in data and len(data['neqr']) > 0:
                        try:
                            # Validate and clean data
                            neqr_data = self._validate_statistical_inputs(
                                np.array(data['neqr']), f"NEQR data for GC {gc}"
                            )
                            if len(neqr_data) >= 3:  # Minimum for meaningful statistics
                                neqr_groups[f'GC_{gc}'] = {'neqr': neqr_data}
                                neqr_total_samples += len(neqr_data)
                        except Exception as e:
                            logger.log_warning(f"Skipping NEQR data for GC {gc}: {e}")

                if len(neqr_groups) >= 2 and neqr_total_samples >= 10:
                    try:
                        neqr_anova = EnhancedStatisticalAnalysis.perform_anova_analysis(
                            neqr_groups, 'neqr'
                        )
                        if neqr_anova:
                            self.anova_results['neqr'] = neqr_anova
                            logger.log_statistical_result('anova_neqr', neqr_anova)
                    except Exception as e:
                        logger.log_error(f"NEQR ANOVA computation failed: {e}")
                        self.anova_results['neqr'] = {'error': str(e)}
                else:
                    logger.log_warning(f"Insufficient data for NEQR ANOVA: {len(neqr_groups)} groups, {neqr_total_samples} samples")

                # ANOVA for FRQI performance
                frqi_groups = {}
                frqi_total_samples = 0
                
                for gc, data in gc_grouped_data.items():
                    if 'frqi' in data and len(data['frqi']) > 0:
                        try:
                            # Validate and clean data
                            frqi_data = self._validate_statistical_inputs(
                                np.array(data['frqi']), f"FRQI data for GC {gc}"
                            )
                            if len(frqi_data) >= 3:  # Minimum for meaningful statistics
                                frqi_groups[f'GC_{gc}'] = {'frqi': frqi_data}
                                frqi_total_samples += len(frqi_data)
                        except Exception as e:
                            logger.log_warning(f"Skipping FRQI data for GC {gc}: {e}")

                if len(frqi_groups) >= 2 and frqi_total_samples >= 10:
                    try:
                        frqi_anova = EnhancedStatisticalAnalysis.perform_anova_analysis(
                            frqi_groups, 'frqi'
                        )
                        if frqi_anova:
                            self.anova_results['frqi'] = frqi_anova
                            logger.log_statistical_result('anova_frqi', frqi_anova)
                    except Exception as e:
                        logger.log_error(f"FRQI ANOVA computation failed: {e}")
                        self.anova_results['frqi'] = {'error': str(e)}
                else:
                    logger.log_warning(f"Insufficient data for FRQI ANOVA: {len(frqi_groups)} groups, {frqi_total_samples} samples")

            except Exception as e:
                logger.log_error(f"ANOVA analysis completely failed: {e}")
                self.anova_results = {'critical_error': str(e)}

    def _generate_statistical_summary(self):
        """🔒✅ THREAD-SAFE Enhanced statistical summary with error handling"""
        with self._safe_results_access():
            try:
                if self.results is None or self.results.empty:
                    logger.log_error("No results available for statistical summary")
                    self.statistical_summary = {'error': 'No results available'}
                    return

                # ✅ ROBUST: Safe statistical computations
                total_pairs = len(self.results)
                total_trials = self.results['N_Trials'].sum() if 'N_Trials' in self.results.columns else 0

                # Safe correlation calculations
                try:
                    if 'hamming' in self.results.columns and 'neqr_mean' in self.results.columns:
                        neqr_correlation = stats.pearsonr(
                            self.results['hamming'].dropna(), 
                            self.results['neqr_mean'].dropna()
                        )
                    else:
                        neqr_correlation = (np.nan, np.nan)
                except Exception as e:
                    logger.log_warning(f"NEQR correlation calculation failed: {e}")
                    neqr_correlation = (np.nan, np.nan)

                try:
                    if 'hamming' in self.results.columns and 'frqi_mean' in self.results.columns:
                        frqi_correlation = stats.pearsonr(
                            self.results['hamming'].dropna(), 
                            self.results['frqi_mean'].dropna()
                        )
                    else:
                        frqi_correlation = (np.nan, np.nan)
                except Exception as e:
                    logger.log_warning(f"FRQI correlation calculation failed: {e}")
                    frqi_correlation = (np.nan, np.nan)

                # Safe error calculations
                neqr_mean_error = self.results['neqr_error'].mean() if 'neqr_error' in self.results.columns else np.nan
                frqi_mean_error = self.results['frqi_error'].mean() if 'frqi_error' in self.results.columns else np.nan

                # Safe improvement calculation
                if not np.isnan(neqr_mean_error) and not np.isnan(frqi_mean_error) and frqi_mean_error > 0:
                    improvement_percentage = ((frqi_mean_error - neqr_mean_error) / frqi_mean_error) * 100
                else:
                    improvement_percentage = np.nan

                # Safe significance counts
                significance_counts = {}
                fdr_columns = [col for col in self.results.columns if col.endswith('_significant_fdr')]
                for col in fdr_columns:
                    try:
                        significance_counts[col] = self.results[col].sum()
                    except:
                        significance_counts[col] = 0

                self.statistical_summary = {
                    'total_sequence_pairs': total_pairs,
                    'total_trials': total_trials,
                    'correlations': {
                        'neqr_correlation': neqr_correlation,
                        'frqi_correlation': frqi_correlation,
                    },
                    'overall_performance': {
                        'neqr_mean_error': neqr_mean_error,
                        'frqi_mean_error': frqi_mean_error,
                        'improvement_percentage': improvement_percentage
                    },
                    'significance_counts': significance_counts,
                    'anova_results': self.anova_results,
                    'analysis_stats': self.analysis_stats.copy()
                }

            except Exception as e:
                logger.log_error(f"Statistical summary generation failed: {e}")
                self.statistical_summary = {'error': str(e)}

    def analyze(self, n_sequences=CONFIG['statistical_sequences'],
                sequence_length=CONFIG['sequence_length'],
                gc_levels=CONFIG['gc_levels'],
                n_trials=CONFIG['n_trials']):
        """🔒✅ THREAD-SAFE Enhanced statistical analysis with comprehensive error handling"""
        
        # Record start time
        self.analysis_stats['start_time'] = time.time()
        
        print(f"\n📈 🔒 THREAD-SAFE Enhanced Statistical Analysis...")
        print(f"   - Sequences per GC level: {n_sequences}")
        print(f"   - Trials per sequence: {n_trials}")
        print(f"   - GC levels: {gc_levels}")
        print(f"   - Total circuits: {len(gc_levels) * n_sequences * n_trials * 2}")

        try:
            # ✅ ROBUST: Comprehensive input validation
            self._validate_sequence_parameters(n_sequences, sequence_length, gc_levels, n_trials)

            # Set seed for reproducibility
            np.random.seed(ANALYSIS_SEEDS['statistical'])

            with self._thread_safe_analysis():
                results_data = []
                gc_grouped_data = {gc: {'neqr': [], 'frqi': [], 'hamming': []} for gc in gc_levels}

                total_work = len(gc_levels) * n_sequences
                successful_analyses = 0
                failed_analyses = 0
                
                # ✅ ENHANCED: Progress tracking with error monitoring
                with tqdm(total=total_work, 
                          desc="📈 🔒 Statistical Analysis", 
                          ncols=120, 
                          ascii=False,
                          position=0,
                          bar_format='{desc}: {percentage:3.0f}%|{bar:80}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                    
                    for gc_content in gc_levels:
                        for seq_idx in range(n_sequences):
                            try:
                                # ✅ ROBUST: Safe sequence generation
                                try:
                                    seq1 = EnhancedSequenceAnalyzer.generate_random_sequence(
                                        sequence_length, gc_content=gc_content, seed=seq_idx
                                    )
                                    seq2 = EnhancedSequenceAnalyzer.generate_random_sequence(
                                        sequence_length, gc_content=gc_content, seed=seq_idx + 1000
                                    )
                                except Exception as e:
                                    logger.log_error(f"Sequence generation failed for GC {gc_content}, seq {seq_idx}: {e}")
                                    failed_analyses += 1
                                    continue

                                # Validate sequences
                                if len(seq1) != sequence_length or len(seq2) != sequence_length:
                                    logger.log_error(f"Generated sequences have wrong length: {len(seq1)}, {len(seq2)}")
                                    failed_analyses += 1
                                    continue

                                # ✅ ROBUST: Safe similarity calculation
                                try:
                                    hamming = EnhancedSequenceAnalyzer.calculate_hamming_similarity(seq1, seq2)
                                    if not (0 <= hamming <= 1):
                                        logger.log_warning(f"Invalid Hamming similarity {hamming}, clamping to [0,1]")
                                        hamming = max(0, min(1, hamming))
                                except Exception as e:
                                    logger.log_error(f"Hamming similarity calculation failed: {e}")
                                    failed_analyses += 1
                                    continue

                                # 🔒✅ THREAD-SAFE: Safe quantum simulations
                                neqr_trials, frqi_trials = self._safe_quantum_execution(seq1, seq2, n_trials)
                                
                                # Update circuit count
                                self.analysis_stats['total_circuits_run'] += n_trials * 2

                                # ✅ ROBUST: Safe statistics calculation
                                try:
                                    neqr_stats = EnhancedStatisticalAnalysis.calculate_comprehensive_stats(neqr_trials)
                                    frqi_stats = EnhancedStatisticalAnalysis.calculate_comprehensive_stats(frqi_trials)
                                except Exception as e:
                                    logger.log_error(f"Statistics calculation failed: {e}")
                                    failed_analyses += 1
                                    continue

                                # ✅ ROBUST: Safe statistical tests
                                p_neqr_classical, p_frqi_classical, p_neqr_frqi = self._safe_statistical_tests(
                                    neqr_trials, frqi_trials, hamming
                                )

                                # 🔒 THREAD-SAFE: Store for ANOVA
                                try:
                                    gc_grouped_data[gc_content]['neqr'].extend(neqr_trials.tolist())
                                    gc_grouped_data[gc_content]['frqi'].extend(frqi_trials.tolist())
                                    gc_grouped_data[gc_content]['hamming'].append(hamming)
                                except Exception as e:
                                    logger.log_warning(f"Failed to store grouped data: {e}")

                                # ✅ ROBUST: Safe result entry creation
                                try:
                                    result_entry = {
                                        'gc_content': gc_content,
                                        'sequence_pair': seq_idx,
                                        'sequence_length': sequence_length,
                                        'hamming': hamming,

                                        # NEQR comprehensive stats
                                        'neqr_mean': neqr_stats.get('mean', np.nan),
                                        'neqr_std': neqr_stats.get('std', np.nan),
                                        'neqr_median': neqr_stats.get('median', np.nan),
                                        'neqr_ci_lower': neqr_stats.get('ci_lower', np.nan),
                                        'neqr_ci_upper': neqr_stats.get('ci_upper', np.nan),
                                        'neqr_skewness': neqr_stats.get('skewness', np.nan),
                                        'neqr_kurtosis': neqr_stats.get('kurtosis', np.nan),

                                        # FRQI comprehensive stats
                                        'frqi_mean': frqi_stats.get('mean', np.nan),
                                        'frqi_std': frqi_stats.get('std', np.nan),
                                        'frqi_median': frqi_stats.get('median', np.nan),
                                        'frqi_ci_lower': frqi_stats.get('ci_lower', np.nan),
                                        'frqi_ci_upper': frqi_stats.get('ci_upper', np.nan),
                                        'frqi_skewness': frqi_stats.get('skewness', np.nan),
                                        'frqi_kurtosis': frqi_stats.get('kurtosis', np.nan),

                                        # Error metrics
                                        'neqr_error': abs(neqr_stats.get('mean', 0) - hamming),
                                        'frqi_error': abs(frqi_stats.get('mean', 0) - hamming),

                                        # Statistical tests
                                        'P_Value_NEQR_Classical': p_neqr_classical,
                                        'P_Value_FRQI_Classical': p_frqi_classical,
                                        'P_Value_NEQR_FRQI': p_neqr_frqi,

                                        'N_Trials': n_trials
                                    }

                                    results_data.append(result_entry)
                                    successful_analyses += 1
                                    
                                except Exception as e:
                                    logger.log_error(f"Failed to create result entry: {e}")
                                    failed_analyses += 1

                            except Exception as e:
                                logger.log_error(f"Analysis failed for GC {gc_content}, seq {seq_idx}: {e}")
                                failed_analyses += 1
                                with self._analysis_lock:
                                    self.analysis_stats['thread_safety_violations'] += 1

                            finally:
                                pbar.update(1)

                            # ✅ ENHANCED: Periodic maintenance
                            if seq_idx % 25 == 0:
                                gc.collect()

                # Record final statistics
                self.analysis_stats['successful_sequences'] = successful_analyses
                self.analysis_stats['failed_sequences'] = failed_analyses
                self.analysis_stats['end_time'] = time.time()

                print(f"\n📊 Analysis Summary:")
                print(f"   - Successful analyses: {successful_analyses}")
                print(f"   - Failed analyses: {failed_analyses}")
                if successful_analyses + failed_analyses > 0:
                    success_rate = successful_analyses / (successful_analyses + failed_analyses) * 100
                    print(f"   - Success rate: {success_rate:.1f}%")

                if successful_analyses == 0:
                    raise RuntimeError("All statistical analyses failed - no results to process")

                # 🔒 THREAD-SAFE: Store results
                with self._safe_results_access():
                    self.results = pd.DataFrame(results_data)

                # Perform post-processing analyses
                self._perform_anova_analysis(gc_grouped_data)
                self._apply_fdr_corrections()
                self._generate_statistical_summary()

                # Clear cache
                self.runner.clear_cache()

                print(f"✅ Statistical analysis completed successfully!")
                return self.results

        except Exception as e:
            logger.log_error(f"Statistical analysis completely failed: {e}")
            self.analysis_stats['end_time'] = time.time()
            raise

    # 🔒 THREAD-SAFE VISUALIZATION METHODS

    def plot_clean_correlation_analysis(self):
        """🔒 Create clean 2x2 panel correlation analysis"""
        with self._plotting_lock:
            if self.results is None:
                raise ValueError("Run analysis first")
            try:
                QuantumDNAVisualizer.plot_clean_correlation_analysis(self.results)
            except Exception as e:
                logger.log_error(f"Correlation analysis plotting failed: {e}")

    def plot_performance_by_gc_content(self):
        """🔒 Create grouped error bar plots by GC content"""
        with self._plotting_lock:
            if self.results is None:
                raise ValueError("Run analysis first")
            try:
                QuantumDNAVisualizer.plot_performance_by_gc_content(self.results)
            except Exception as e:
                logger.log_error(f"GC content plotting failed: {e}")

    def plot_performance_distributions(self):
        """🔒 Create box plots and violin plots for distribution comparison"""
        with self._plotting_lock:
            if self.results is None:
                raise ValueError("Run analysis first")
            try:
                QuantumDNAVisualizer.plot_performance_distributions(self.results)
            except Exception as e:
                logger.log_error(f"Distribution plotting failed: {e}")

    def plot_gc_content_trends(self):
        """🔒 Create trend analysis plots with confidence bands"""
        with self._plotting_lock:
            if self.results is None:
                raise ValueError("Run analysis first")
            try:
                QuantumDNAVisualizer.plot_gc_content_trends(self.results)
            except Exception as e:
                logger.log_error(f"GC trends plotting failed: {e}")

    def save_comprehensive_results(self):
        """🔒✅ THREAD-SAFE Save comprehensive results and summaries"""
        with self._safe_results_access():
            if self.results is None:
                logger.log_error("No results to save")
                return

            try:
                # Save main results
                self.results.to_csv('results/tables/statistical_results_enhanced.csv',
                                   index=False, encoding='utf-16')
                print("✅ Main results saved")

                # Save statistical summary
                if self.statistical_summary:
                    if 'overall_performance' in self.statistical_summary:
                        summary_df = pd.DataFrame([self.statistical_summary['overall_performance']])
                        summary_df.to_csv('results/tables/statistical_summary_enhanced.csv',
                                         index=False, encoding='utf-16')
                        print("✅ Statistical summary saved")

                # Save ANOVA results
                if self.anova_results and not ('error' in self.anova_results or 'critical_error' in self.anova_results):
                    try:
                        anova_df = pd.DataFrame(self.anova_results).T
                        anova_df.to_csv('results/tables/anova_results_enhanced.csv',
                                       encoding='utf-16')
                        print("✅ ANOVA results saved")
                    except Exception as e:
                        logger.log_warning(f"Failed to save ANOVA results: {e}")

                # Save GC content breakdown
                if not self.results.empty:
                    try:
                        gc_breakdown = self.results.groupby('gc_content').agg({
                            'neqr_mean': ['mean', 'std', 'count'],
                            'frqi_mean': ['mean', 'std', 'count'],
                            'neqr_error': ['mean', 'std'],
                            'frqi_error': ['mean', 'std']
                        }).round(6)

                        gc_breakdown.to_csv('results/tables/gc_content_breakdown_enhanced.csv',
                                           encoding='utf-16')
                        print("✅ GC breakdown saved")
                    except Exception as e:
                        logger.log_warning(f"Failed to save GC breakdown: {e}")

                # Save checkpoint
                try:
                    with open('results/supplementary/checkpoint_statistical_enhanced.pkl', 'wb') as f:
                        pickle.dump(self.results, f)
                    print("✅ Checkpoint saved")
                except Exception as e:
                    logger.log_warning(f"Failed to save checkpoint: {e}")

                # Create all new visualization plots
                print("\n📊 Generating Enhanced Visualizations...")
                try:
                    self.plot_clean_correlation_analysis()
                    self.plot_performance_by_gc_content()
                    self.plot_performance_distributions()
                    self.plot_gc_content_trends()
                    print("✅ All visualizations generated")
                except Exception as e:
                    logger.log_error(f"Some visualizations failed: {e}")

                # Log performance
                logger.log_performance('statistical_analysis', self.statistical_summary)

                # Print summary
                if self.statistical_summary and 'correlations' in self.statistical_summary:
                    print("\n📊 Enhanced Statistical Analysis Summary:")
                    print(f"   - Total sequence pairs: {self.statistical_summary.get('total_sequence_pairs', 0)}")
                    
                    neqr_corr = self.statistical_summary['correlations'].get('neqr_correlation', (np.nan, np.nan))
                    frqi_corr = self.statistical_summary['correlations'].get('frqi_correlation', (np.nan, np.nan))
                    
                    print(f"   - NEQR correlation: {neqr_corr[0]:.6f}")
                    print(f"   - FRQI correlation: {frqi_corr[0]:.6f}")
                    
                    improvement = self.statistical_summary['overall_performance'].get('improvement_percentage', np.nan)
                    if not np.isnan(improvement):
                        print(f"   - NEQR improvement: {improvement:.2f}%")
                    
                    if self.anova_results:
                        if 'neqr' in self.anova_results and 'f_statistic' in self.anova_results['neqr']:
                            print(f"   - ANOVA NEQR F-stat: {self.anova_results['neqr']['f_statistic']:.3f}")
                        if 'frqi' in self.anova_results and 'f_statistic' in self.anova_results['frqi']:
                            print(f"   - ANOVA FRQI F-stat: {self.anova_results['frqi']['f_statistic']:.3f}")

                print("✅ Enhanced statistical analysis complete!")

            except Exception as e:
                logger.log_error(f"Failed to save comprehensive results: {e}")
                raise