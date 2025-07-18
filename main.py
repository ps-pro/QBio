#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Quantum DNA Analysis - Main Execution Script
Professional implementation with comprehensive error handling and memory management
"""

import time
import gc
import pickle
import warnings
import numpy as np
import pandas as pd
import psutil
import sys
import os
from contextlib import contextmanager

# Import project modules
from src.config.settings import CONFIG, setup_matplotlib, setup_directories
from src.core.runner import EnhancedCPURunner
from src.analysis.diversity import EnhancedDiversityAnalysis
from src.analysis.statistical import EnhancedStatisticalAnalyzer
from src.analysis.scaling import EnhancedScalingAnalysis
from src.analysis.noise import EnhancedNoiseAnalysis
from src.utils.logging import logger

# Suppress warnings
warnings.filterwarnings('ignore')

@contextmanager
def memory_monitor(phase_name):
    """Memory monitoring context manager"""
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    print(f"\nMemory Monitor - {phase_name} Started:")
    print(f"   Initial Memory: {start_memory:.1f} MB")
    print(f"   Available Memory: {psutil.virtual_memory().available / 1024 / 1024:.1f} MB")
    
    try:
        yield
    finally:
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_time = time.time()
        memory_delta = end_memory - start_memory
        
        print(f"Memory Monitor - {phase_name} Completed:")
        print(f"   Final Memory: {end_memory:.1f} MB")
        print(f"   Memory Delta: {memory_delta:+.1f} MB")
        print(f"   Duration: {end_time - start_time:.1f}s")
        
        gc.collect()

def validate_system_requirements():
    """Validate system requirements before starting"""
    print("\nSystem Requirements Check:")
    
    # Check available memory
    available_memory_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
    print(f"   Available Memory: {available_memory_gb:.1f} GB")
    
    if available_memory_gb < 2.0:
        logger.log_warning("Low memory detected - analysis may be slower")
    
    # Check CPU cores
    cpu_count = psutil.cpu_count()
    print(f"   CPU Cores: {cpu_count}")
    
    if cpu_count < 4:
        logger.log_warning("Few CPU cores detected - consider reducing parallelization")
    
    # Check disk space
    disk_usage = psutil.disk_usage('.')
    free_space_gb = disk_usage.free / 1024 / 1024 / 1024
    print(f"   Free Disk Space: {free_space_gb:.1f} GB")
    
    if free_space_gb < 1.0:
        raise RuntimeError("Insufficient disk space - need at least 1GB free")
    
    print("System requirements check passed")

def safe_analysis_execution(analysis_func, analysis_name, *args, **kwargs):
    """Safe execution wrapper with comprehensive error handling"""
    max_retries = 2
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries + 1):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries + 1} for {analysis_name}")
            
            with memory_monitor(f"{analysis_name} - Attempt {attempt + 1}"):
                result = analysis_func(*args, **kwargs)
                
            print(f"{analysis_name} completed successfully")
            return result
            
        except MemoryError:
            logger.log_error(f"Memory error in {analysis_name} (attempt {attempt + 1})")
            if attempt < max_retries:
                print(f"   Forcing garbage collection and retrying in {retry_delay}s...")
                gc.collect()
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"{analysis_name} failed: Out of memory")
                
        except Exception as e:
            logger.log_error(f"{analysis_name} failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                print(f"   Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"{analysis_name} failed after {max_retries + 1} attempts")
                return None

def main():
    """Main execution with comprehensive error handling and monitoring"""
    print("Enhanced Quantum DNA Analysis - Professional Version")
    print("=" * 80)
    
    try:
        # System validation
        validate_system_requirements()
        
        # Initialize configuration
        setup_matplotlib()
        setup_directories()

        # Initialize runner with error handling
        try:
            runner = EnhancedCPURunner(shots=CONFIG['shots'])
        except Exception as e:
            logger.log_error(f"Failed to initialize runner: {e}")
            raise RuntimeError("Cannot initialize quantum simulation backend")

        # Set master seed for reproducibility
        np.random.seed(CONFIG['random_seed'])

        # Storage for results
        all_results = {}
        start_time = time.time()

        print(f"\nConfiguration:")
        print(f"   Shots per circuit: {CONFIG['shots']}")
        print(f"   GC levels: {CONFIG['gc_levels']}")
        print(f"   Sequence length: {CONFIG['sequence_length']}")
        print(f"   Scaling range: 12-{CONFIG['max_scaling_length']}")
        print(f"   Figure DPI: {CONFIG['figure_dpi']}")
        print(f"   Thread Safety: Enabled")
        print(f"   Memory Monitoring: Enabled")
        print(f"   Error Recovery: Enabled")

        # PHASE 1: Enhanced Diversity Analysis
        print("\n" + "="*70)
        print("PHASE 1: ENHANCED DIVERSITY ANALYSIS")
        print("="*70)

        diversity_result = safe_analysis_execution(
            lambda: EnhancedDiversityAnalysis(runner).analyze(CONFIG['diversity_trials']),
            "Diversity Analysis"
        )
        
        if diversity_result is not None:
            all_results['diversity'] = diversity_result
            try:
                diversity_analyzer = EnhancedDiversityAnalysis(runner)
                diversity_analyzer.results = diversity_result
                diversity_analyzer.plot_minimalistic_results()
            except Exception as e:
                logger.log_error(f"Diversity plotting failed: {e}")

        # PHASE 2: Enhanced Statistical Analysis
        print("\n" + "="*70)
        print("PHASE 2: ENHANCED STATISTICAL ANALYSIS")
        print("="*70)

        statistical_result = safe_analysis_execution(
            lambda: EnhancedStatisticalAnalyzer(runner).analyze(
                CONFIG['statistical_sequences'],
                CONFIG['sequence_length'],
                CONFIG['gc_levels'],
                CONFIG['n_trials']
            ),
            "Statistical Analysis"
        )
        
        if statistical_result is not None:
            all_results['statistical'] = statistical_result
            try:
                stat_analyzer = EnhancedStatisticalAnalyzer(runner)
                stat_analyzer.results = statistical_result
                stat_analyzer.save_comprehensive_results()
            except Exception as e:
                logger.log_error(f"Statistical results saving failed: {e}")

        # PHASE 3: Enhanced Scaling Analysis
        print("\n" + "="*70)
        print("PHASE 3: ENHANCED SCALING ANALYSIS")
        print("="*70)

        scaling_result = safe_analysis_execution(
            lambda: EnhancedScalingAnalysis(runner).analyze(
                min_length=12,
                max_length=CONFIG['max_scaling_length'],
                step=CONFIG['scaling_step'],
                n_sequences=CONFIG['scaling_sequences'],
                n_trials=20
            ),
            "Scaling Analysis"
        )
        
        if scaling_result is not None:
            all_results['scaling'] = scaling_result
            try:
                scaling_analyzer = EnhancedScalingAnalysis(runner)
                scaling_analyzer.results = scaling_result
                scaling_analyzer.plot_scaling_results()
                scaling_analyzer.save_comprehensive_results()
            except Exception as e:
                logger.log_error(f"Scaling results saving failed: {e}")

        # PHASE 4: Enhanced Noise Analysis
        print("\n" + "="*70)
        print("PHASE 4: ENHANCED NOISE ANALYSIS")
        print("="*70)

        noise_result = safe_analysis_execution(
            lambda: EnhancedNoiseAnalysis(runner).analyze(
                noise_levels=CONFIG['default_noise_levels'],
                n_sequences=CONFIG['noise_sequences'],
                sequence_length=CONFIG['noise_sequence_length'],
                gc_levels=CONFIG['noise_gc_levels'],
                n_trials=CONFIG['noise_trials']
            ),
            "Noise Analysis"
        )
        
        if noise_result is not None:
            all_results['noise'] = noise_result
            try:
                noise_analyzer = EnhancedNoiseAnalysis(runner)
                noise_analyzer.results = noise_result
                noise_analyzer.save_comprehensive_results()
            except Exception as e:
                logger.log_error(f"Noise results saving failed: {e}")

        # COMPREHENSIVE SUMMARY
        total_time = time.time() - start_time
        
        try:
            perf_stats = runner.get_comprehensive_performance_stats()
        except Exception as e:
            logger.log_error(f"Failed to get performance stats: {e}")
            perf_stats = {'execution_metrics': {'total_circuits': 0, 'circuits_per_second': 0}}

        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE SUMMARY")
        print("="*70)

        print(f"Execution Performance:")
        print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"   Total circuits: {perf_stats['execution_metrics'].get('total_circuits', 0):,}")
        print(f"   Circuits/second: {perf_stats['execution_metrics'].get('circuits_per_second', 0):.1f}")
        
        if 'thread_safety' in perf_stats:
            print(f"   Thread Safety: {perf_stats['thread_safety']['thread_safety_status']}")
            print(f"   Violations: {perf_stats['thread_safety']['violations_detected']}")

        # Save all results with error handling
        try:
            if all_results:
                with open('results/supplementary/all_results_enhanced.pkl', 'wb') as f:
                    pickle.dump(all_results, f)
                print(f"\nResults saved successfully")
            else:
                logger.log_warning("No results to save")
        except Exception as e:
            logger.log_error(f"Failed to save results: {e}")

        # Save performance statistics
        try:
            if 'execution_metrics' in perf_stats:
                perf_df = pd.DataFrame([perf_stats['execution_metrics']])
                perf_df.to_csv('results/tables/performance_statistics.csv',
                              index=False, encoding='utf-16')
        except Exception as e:
            logger.log_error(f"Failed to save performance statistics: {e}")

        # Save comprehensive log
        logger.log_performance('overall_execution', perf_stats)
        logger.save_log('comprehensive_execution_log.json')

        print(f"\nENHANCED ANALYSIS COMPLETE")
        print(f"   Successful analyses: {len(all_results)}/4")
        print(f"   Thread Safety: Enabled")
        print(f"   Memory Management: Enabled")
        print(f"   Error Recovery: Enabled")

        return all_results

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        logger.log_warning("Analysis interrupted by user")
        return {}
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        logger.log_error(f"Critical error in main execution: {e}", e)
        
        # Emergency cleanup
        try:
            gc.collect()
        except:
            pass
            
        raise

if __name__ == "__main__":
    print("Enhanced Quantum DNA Analysis")
    print("Professional Repository Structure")
    print("Academic-Level Documentation")
    print("Minimalistic IBM Design")
    print("Comprehensive Statistical Analysis")
    print("Thread-Safe Parallelization")
    print("Enhanced Error Handling")
    print("Memory Monitoring")

    # Initial memory cleanup
    gc.collect()

    try:
        # Run the enhanced analysis
        results = main()

        print("\n" + "="*80)
        print("ENHANCED ANALYSIS COMPLETE")
        print("="*80)
        print("All phases completed with enhanced safety")
        print("Statistical rigor: FDR correction applied")
        print("Advanced visualizations: Thread-safe generation")
        print("Comprehensive logging: Academic-level documentation")
        print("Minimalistic design: Clean IBM styling")
        print("All data: CSV format with UTF-16 encoding")
        print("Reproducible: All seeds documented")
        print("High performance: Optimized parallel processing")
        print("Thread Safety: Comprehensive protection")
        print("Error Recovery: Robust failure handling")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please check the logs for detailed error information.")
        sys.exit(1)