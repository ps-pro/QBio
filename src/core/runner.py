# -*- coding: utf-8 -*-
"""
Enhanced CPU simulation runner with 16-core optimization
"""

import time
import gc
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from qiskit import transpile
from qiskit_aer import AerSimulator
from .cache import EnhancedCircuitCache
from ..config.settings import CONFIG, CPU_COUNT
from ..utils.logging import logger

# USE ALL 16 CORES!
MAX_WORKERS = CPU_COUNT  # Changed from min(CPU_COUNT, 8)

class EnhancedCPURunner:
    """Enhanced CPU simulation runner with comprehensive tracking and 16-core optimization"""

    def __init__(self, shots=CONFIG['shots']):
        self.shots = shots
        self.circuit_cache = EnhancedCircuitCache(max_size=3000)

        # Initialize optimized CPU backend
        self.backend = AerSimulator(method='statevector', device='CPU')
        self.backend.set_options(max_parallel_threads=CPU_COUNT)

        print(f"🚀 Enhanced CPU Backend Initialized!")
        print(f"   - CPU Cores Available: {CPU_COUNT}")
        print(f"   - ALL {CPU_COUNT} cores will be utilized!")
        print(f"   - Shots per circuit: {shots}")
        print(f"   - Enhanced cache size: 3000 circuits")

        # Enhanced performance tracking
        self.performance_metrics = {
            'total_circuits': 0,
            'total_time': 0,
            'parallel_jobs': 0,
            'method_breakdown': {'neqr': 0, 'frqi': 0},
            'length_breakdown': {},
            'gc_breakdown': {}
        }

    def run_single_circuit_optimized(self, circuit, noise_model=None):
        """Enhanced single circuit execution with tracking"""
        try:
            start_time = time.time()

            # Pre-transpile for efficiency
            transpiled = transpile(circuit, self.backend, optimization_level=1)

            # Use noise model if provided
            if noise_model:
                backend_with_noise = AerSimulator(noise_model=noise_model, device='CPU')
                backend_with_noise.set_options(max_parallel_threads=CPU_COUNT)
                job = backend_with_noise.run(transpiled, shots=self.shots)
            else:
                job = self.backend.run(transpiled, shots=self.shots)

            result = job.result()

            # Track execution time
            execution_time = time.time() - start_time
            self.performance_metrics['total_time'] += execution_time

            return result.get_counts()

        except Exception as e:
            logger.log_error(f"Circuit execution failed: {e}")
            return {'0': 0, '1': 0}

    def calculate_similarity_from_counts(self, counts, method):
        """Convert measurement counts to similarity score (method-specific)"""
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0

        p0 = counts.get('0', 0) / total_shots

        if method == 'neqr':
            # NEQR (swap test): p0 = probability of measuring |0⟩ on ancilla
            # p0 = 1 means identical, p0 = 0.5 means orthogonal
            # Scale from [0.5, 1] to [0, 1]
            if p0 >= 0.5:
                return 2 * (p0 - 0.5)  # Maps [0.5,1] → [0,1]
            else:
                return 0.0  # Below 0.5 = anti-correlated, clamp to 0

        elif method == 'frqi':
            # FRQI: Direct probability mapping
            return p0  # Maps [0,1] → [0,1]

        else:
            raise ValueError(f"Unknown method: {method}")

    
    def run_circuits_parallel_batch_fixed(self, circuit_configs, max_workers=None):
        """FIXED parallel batch processing with preserved order - USES SPECIFIED CORES"""
        if max_workers is None:
            max_workers = MAX_WORKERS

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs and preserve order mapping
            future_to_index = {}
            futures = []

            for i, config in enumerate(circuit_configs):
                seq1, seq2, method, noise_model = config
                circuit = self.circuit_cache.get_circuit(seq1, seq2, method)
                future = executor.submit(self.run_single_circuit_optimized, circuit, noise_model)
                future_to_index[future] = i
                futures.append(future)

            # Initialize results array with correct size
            all_results = [None] * len(circuit_configs)

            # Collect results in completion order but store in submission order
            method_name = circuit_configs[0][2].upper() if circuit_configs else "Processing"
            # for future in tqdm(as_completed(futures),
            #          total=len(circuit_configs),
            #          desc=f"{method_name}",  # Simplified - no core count
            #          leave=False,           # Don't leave completed bars
            #          disable=False,         # Always show progress
            #          ascii=True,           # Use simple ASCII
            #          ncols=60,             # Shorter width
            #          position=None):
            for future in as_completed(futures):
                try:
                    counts = future.result(timeout=60)
                    original_index = future_to_index[future]
                    all_results[original_index] = counts  # ✅ CORRECT ORDER!

                    # Track method usage
                    method = circuit_configs[original_index][2]
                    self.performance_metrics['method_breakdown'][method] += 1

                except Exception as e:
                    logger.log_error(f"Batch job failed: {e}")
                    original_index = future_to_index[future]
                    all_results[original_index] = {'0': 0, '1': 0}

        self.performance_metrics['total_circuits'] += len(circuit_configs)
        self.performance_metrics['parallel_jobs'] += 1

        return all_results


    def run_multiple_trials_enhanced_parallel(self, seq1, seq2, method='neqr', n_trials=CONFIG['n_trials'], 
                                            noise_model=None, track_metadata=True, max_workers=None):
        """Enhanced multiple trials with configurable core usage"""

        if max_workers is None:
            max_workers = MAX_WORKERS

        # Track sequence metadata
        if track_metadata:
            length = len(seq1)
            gc_content = (seq1.count('G') + seq1.count('C')) / length

            self.performance_metrics['length_breakdown'][length] = \
                self.performance_metrics['length_breakdown'].get(length, 0) + 1

            gc_key = f"{gc_content:.1f}"
            self.performance_metrics['gc_breakdown'][gc_key] = \
                self.performance_metrics['gc_breakdown'].get(gc_key, 0) + 1

        # Create batch configurations (ALL SAME METHOD - NO MIXING!)
        circuit_configs = [(seq1, seq2, method, noise_model) for _ in range(n_trials)]

        # Run all trials in parallel with FIXED order preservation
        all_counts = self.run_circuits_parallel_batch_fixed(circuit_configs, max_workers=max_workers)

        # Convert to similarity scores with METHOD-SPECIFIC calculation
        similarities = []
        for counts in all_counts:
            sim = self.calculate_similarity_from_counts(counts, method)
            similarities.append(sim)

        return np.array(similarities)

    def run_both_methods_split_cores(self, seq1, seq2, n_trials=CONFIG['n_trials'], noise_model=None):
        """🚀 OPTION 2: Run NEQR and FRQI simultaneously on split cores (8+8)"""
        
        # Split cores: 8 for NEQR, 8 for FRQI
        cores_per_method = MAX_WORKERS // 2  # 8 cores each
        
        print(f"    🔥 Running NEQR + FRQI simultaneously:")
        print(f"    ├── NEQR: {cores_per_method} cores")
        print(f"    └── FRQI: {cores_per_method} cores")
        
        # Create thread-local storage to avoid conflicts
        neqr_result = [None]
        frqi_result = [None]
        
        def run_neqr():
            try:
                result = self.run_multiple_trials_enhanced_parallel(
                    seq1, seq2, 'neqr', n_trials, noise_model, 
                    track_metadata=True, max_workers=cores_per_method
                )
                neqr_result[0] = result
            except Exception as e:
                logger.log_error(f"NEQR execution failed: {e}")
                neqr_result[0] = np.zeros(n_trials)
        
        def run_frqi():
            try:
                result = self.run_multiple_trials_enhanced_parallel(
                    seq1, seq2, 'frqi', n_trials, noise_model, 
                    track_metadata=True, max_workers=cores_per_method
                )
                frqi_result[0] = result
            except Exception as e:
                logger.log_error(f"FRQI execution failed: {e}")
                frqi_result[0] = np.zeros(n_trials)
        
        # Run both methods simultaneously on different core sets
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_neqr = executor.submit(run_neqr)
            future_frqi = executor.submit(run_frqi)
            
            # Wait for both to complete
            future_neqr.result()
            future_frqi.result()
        
        return neqr_result[0], frqi_result[0]

    def run_both_methods_sequential_full_cores(self, seq1, seq2, n_trials=CONFIG['n_trials'], noise_model=None):
        """🔥 OPTION 1: Run NEQR and FRQI sequentially but each using ALL 16 CORES"""
        
        print(f"    🔥 Running NEQR trials on ALL {MAX_WORKERS} cores...")
        neqr_trials = self.run_multiple_trials_enhanced_parallel(
            seq1, seq2, 'neqr', n_trials, noise_model, max_workers=MAX_WORKERS
        )
        
        print(f"    🔥 Running FRQI trials on ALL {MAX_WORKERS} cores...")
        frqi_trials = self.run_multiple_trials_enhanced_parallel(
            seq1, seq2, 'frqi', n_trials, noise_model, max_workers=MAX_WORKERS
        )
        
        return neqr_trials, frqi_trials

    # MAIN METHOD - Currently using Option 2 (split cores)
    def run_both_methods_parallel(self, seq1, seq2, n_trials=CONFIG['n_trials'], noise_model=None):
        """Main method for running both quantum algorithms - OPTION 2 IMPLEMENTATION"""
        return self.run_both_methods_sequential_full_cores(seq1, seq2, n_trials, noise_model)


    # Legacy compatibility methods
    def run_multiple_trials_enhanced(self, seq1, seq2, method='neqr', n_trials=CONFIG['n_trials'], 
                                   noise_model=None, track_metadata=True):
        """Legacy method for backward compatibility"""
        return self.run_multiple_trials_enhanced_parallel(
            seq1, seq2, method, n_trials, noise_model, track_metadata, max_workers=MAX_WORKERS
        )

    def run_circuits_parallel_batch(self, circuit_configs, max_workers=None):
        """Legacy method for backward compatibility"""
        return self.run_circuits_parallel_batch_fixed(circuit_configs, max_workers)

    def get_comprehensive_performance_stats(self):
        """Get comprehensive performance statistics"""
        cache_stats = self.circuit_cache.get_comprehensive_stats()

        avg_time = self.performance_metrics['total_time'] / self.performance_metrics['total_circuits'] \
                   if self.performance_metrics['total_circuits'] > 0 else 0
        circuits_per_sec = self.performance_metrics['total_circuits'] / self.performance_metrics['total_time'] \
                          if self.performance_metrics['total_time'] > 0 else 0

        return {
            'execution_metrics': {
                'total_circuits': self.performance_metrics['total_circuits'],
                'total_time': self.performance_metrics['total_time'],
                'avg_time_per_circuit': avg_time,
                'circuits_per_second': circuits_per_sec,
                'parallel_jobs': self.performance_metrics['parallel_jobs'],
            },
            'cache_metrics': cache_stats,
            'usage_breakdown': {
                'methods': self.performance_metrics['method_breakdown'],
                'lengths': self.performance_metrics['length_breakdown'],
                'gc_content': self.performance_metrics['gc_breakdown']
            },
            'system_metrics': {
                'cpu_utilization': f"{MAX_WORKERS}/{CPU_COUNT} cores",
                'memory_usage': f"{len(self.circuit_cache.cache)} circuits cached",
                'core_allocation': f"Split mode: {MAX_WORKERS//2} cores per method"
            }
        }

    def clear_cache(self):
        """Clear cache and memory"""
        self.circuit_cache.cache.clear()
        self.circuit_cache.access_count.clear()
        self.circuit_cache.creation_time.clear()
        gc.collect()

    # Performance monitoring methods
    def get_performance_summary(self):
        """Get quick performance summary"""
        stats = self.get_comprehensive_performance_stats()
        return (f"Performance: {stats['execution_metrics']['circuits_per_second']:.1f} circuits/sec, "
                f"Cache hit rate: {stats['cache_metrics']['hit_rate']:.1%}, "
                f"Using {stats['system_metrics']['cpu_utilization']}")

    def reset_performance_metrics(self):
        """Reset performance tracking"""
        self.performance_metrics = {
            'total_circuits': 0,
            'total_time': 0,
            'parallel_jobs': 0,
            'method_breakdown': {'neqr': 0, 'frqi': 0},
            'length_breakdown': {},
            'gc_breakdown': {}
        }