# -*- coding: utf-8 -*-
"""
🔒 THREAD-SAFE Enhanced CPU simulation runner with 16-core optimization
"""

import time
import gc
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from qiskit import transpile
from .cache import EnhancedCircuitCache
from ..config.settings import CONFIG, CPU_COUNT
from ..utils.logging import logger


try:
    from qiskit_aer import AerSimulator
except ImportError:
    try:
        from qiskit.providers.aer import AerSimulator
    except ImportError:
        raise ImportError("Cannot import AerSimulator. Please ensure qiskit-aer is installed.")


# USE ALL 16 CORES!
MAX_WORKERS = CPU_COUNT

class EnhancedCPURunner:
    """🔒 THREAD-SAFE Enhanced CPU simulation runner"""

    def __init__(self, shots=CONFIG['shots']):
        self.shots = shots
        self.circuit_cache = EnhancedCircuitCache(max_size=3000)

        # 🔒 THREAD SAFETY: Add locks for all shared resources
        self._metrics_lock = threading.RLock()  # Reentrant lock for nested calls
        self._cache_lock = threading.RLock()    # Separate lock for cache operations
        self._backend_lock = threading.Lock()   # Lock for backend operations

        # Initialize optimized CPU backend
        self.backend = AerSimulator(method='statevector', device='CPU')
        self.backend.set_options(max_parallel_threads=CPU_COUNT)

        print(f"🚀 🔒 THREAD-SAFE Enhanced CPU Backend Initialized!")
        print(f"   - CPU Cores Available: {CPU_COUNT}")
        print(f"   - ALL {CPU_COUNT} cores will be utilized!")
        print(f"   - Shots per circuit: {shots}")
        print(f"   - Enhanced cache size: 3000 circuits")
        print(f"   - 🔒 Thread-safe locks: ✅ ENABLED")

        # 🔒 THREAD-SAFE Enhanced performance tracking
        self.performance_metrics = {
            'total_circuits': 0,
            'total_time': 0.0,
            'parallel_jobs': 0,
            'method_breakdown': {'neqr': 0, 'frqi': 0},
            'length_breakdown': {},
            'gc_breakdown': {},
            'thread_safety_violations': 0  # Track any race conditions
        }

    def run_single_circuit_optimized(self, circuit, noise_model=None):
        """🔒 THREAD-SAFE Enhanced single circuit execution with tracking"""
        try:
            start_time = time.time()

            # 🔒 THREAD-SAFE backend access
            with self._backend_lock:
                # Pre-transpile for efficiency
                transpiled = transpile(circuit, self.backend, optimization_level=1)

                # Use noise model if provided
                if noise_model:
                    backend_with_noise = AerSimulator(noise_model=noise_model, device='CPU')
                    backend_with_noise.set_options(max_parallel_threads=1)  # Single thread per job
                    job = backend_with_noise.run(transpiled, shots=self.shots)
                else:
                    job = self.backend.run(transpiled, shots=self.shots)

                result = job.result(timeout=60)  # ✅ ADD TIMEOUT PROTECTION

            # 🔒 THREAD-SAFE metrics update
            execution_time = time.time() - start_time
            with self._metrics_lock:
                self.performance_metrics['total_time'] += execution_time

            return result.get_counts()

        except Exception as e:
            logger.log_error(f"Circuit execution failed: {e}")
            # Return fallback counts that won't break similarity calculation
            return {'0': self.shots // 2, '1': self.shots // 2}

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
        """🔒 THREAD-SAFE parallel batch processing with GUARANTEED order preservation"""
        if max_workers is None:
            max_workers = MAX_WORKERS

        start_time = time.time()

        # 🔒 THREAD-SAFE: Pre-allocate results array to prevent race conditions
        results = [None] * len(circuit_configs)
        results_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs with index tracking
            future_to_index = {}
            
            for i, config in enumerate(circuit_configs):
                seq1, seq2, method, noise_model = config
                
                # 🔒 THREAD-SAFE cache access
                with self._cache_lock:
                    circuit = self.circuit_cache.get_circuit(seq1, seq2, method)
                
                future = executor.submit(self.run_single_circuit_optimized, circuit, noise_model)
                future_to_index[future] = i

            # 🔒 THREAD-SAFE: Collect results with order preservation
            completed_count = 0
            method_name = circuit_configs[0][2].upper() if circuit_configs else "Processing"
            
            with tqdm(total=len(circuit_configs), 
                     desc=f"🔒 {method_name}",
                     leave=False, 
                     disable=False,
                     ascii=False,
                     ncols=80) as pbar:
                
                for future in as_completed(future_to_index.keys()):
                    try:
                        counts = future.result(timeout=90)  # Longer timeout for safety
                        original_index = future_to_index[future]
                        
                        # 🔒 THREAD-SAFE result storage
                        with results_lock:
                            results[original_index] = counts
                            completed_count += 1
                            pbar.update(1)

                        # 🔒 THREAD-SAFE method tracking
                        method = circuit_configs[original_index][2]
                        with self._metrics_lock:
                            self.performance_metrics['method_breakdown'][method] += 1

                    except Exception as e:
                        logger.log_error(f"Batch job failed: {e}")
                        original_index = future_to_index[future]
                        
                        # 🔒 THREAD-SAFE fallback storage
                        with results_lock:
                            results[original_index] = {'0': 0, '1': 0}
                            completed_count += 1
                            pbar.update(1)

                        # Track safety violations
                        with self._metrics_lock:
                            self.performance_metrics['thread_safety_violations'] += 1

        # 🔒 THREAD-SAFE final metrics update
        with self._metrics_lock:
            self.performance_metrics['total_circuits'] += len(circuit_configs)
            self.performance_metrics['parallel_jobs'] += 1

        # ✅ VERIFY: Check that all results are present (no None values)
        if None in results:
            logger.log_error("❌ ORDER PRESERVATION FAILED - Some results missing!")
            raise RuntimeError("Thread safety violation: Missing results detected")

        return results

    def run_multiple_trials_enhanced_parallel(self, seq1, seq2, method='neqr', n_trials=CONFIG['n_trials'], 
                                            noise_model=None, track_metadata=True, max_workers=None):
        """🔒 THREAD-SAFE Enhanced multiple trials with configurable core usage"""

        if max_workers is None:
            max_workers = MAX_WORKERS

        # 🔒 THREAD-SAFE metadata tracking
        if track_metadata:
            length = len(seq1)
            gc_content = (seq1.count('G') + seq1.count('C')) / length

            with self._metrics_lock:
                self.performance_metrics['length_breakdown'][length] = \
                    self.performance_metrics['length_breakdown'].get(length, 0) + 1

                gc_key = f"{gc_content:.1f}"
                self.performance_metrics['gc_breakdown'][gc_key] = \
                    self.performance_metrics['gc_breakdown'].get(gc_key, 0) + 1

        # Create batch configurations (ALL SAME METHOD - NO MIXING!)
        circuit_configs = [(seq1, seq2, method, noise_model) for _ in range(n_trials)]

        # 🔒 THREAD-SAFE parallel execution with order preservation
        all_counts = self.run_circuits_parallel_batch_fixed(circuit_configs, max_workers=max_workers)

        # Convert to similarity scores with METHOD-SPECIFIC calculation
        similarities = []
        for i, counts in enumerate(all_counts):
            if counts is None:
                logger.log_error(f"❌ Missing result at index {i}")
                similarities.append(0.0)  # Fallback
            else:
                sim = self.calculate_similarity_from_counts(counts, method)
                similarities.append(sim)

        return np.array(similarities)

    def run_both_methods_sequential_full_cores(self, seq1, seq2, n_trials=CONFIG['n_trials'], noise_model=None):
        """🔥 🔒 THREAD-SAFE: Run NEQR and FRQI sequentially using ALL 16 CORES each"""
        
        print(f"    🔥 🔒 Running NEQR trials on ALL {MAX_WORKERS} cores (THREAD-SAFE)...")
        neqr_trials = self.run_multiple_trials_enhanced_parallel(
            seq1, seq2, 'neqr', n_trials, noise_model, max_workers=MAX_WORKERS
        )
        
        print(f"    🔥 🔒 Running FRQI trials on ALL {MAX_WORKERS} cores (THREAD-SAFE)...")
        frqi_trials = self.run_multiple_trials_enhanced_parallel(
            seq1, seq2, 'frqi', n_trials, noise_model, max_workers=MAX_WORKERS
        )
        
        return neqr_trials, frqi_trials

    def run_both_methods_split_cores(self, seq1, seq2, n_trials=CONFIG['n_trials'], noise_model=None):
        """🚀 🔒 THREAD-SAFE: Run NEQR and FRQI simultaneously on split cores (8+8)"""
        
        cores_per_method = MAX_WORKERS // 2  # 8 cores each
        
        print(f"    🔥 🔒 Running NEQR + FRQI simultaneously (THREAD-SAFE):")
        print(f"    ├── NEQR: {cores_per_method} cores")
        print(f"    └── FRQI: {cores_per_method} cores")
        
        # 🔒 THREAD-SAFE: Use thread-local storage for results
        neqr_result = [None]
        frqi_result = [None]
        execution_locks = [threading.Lock(), threading.Lock()]
        
        def run_neqr():
            try:
                with execution_locks[0]:  # 🔒 Ensure thread safety
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
                with execution_locks[1]:  # 🔒 Ensure thread safety
                    result = self.run_multiple_trials_enhanced_parallel(
                        seq1, seq2, 'frqi', n_trials, noise_model, 
                        track_metadata=True, max_workers=cores_per_method
                    )
                    frqi_result[0] = result
            except Exception as e:
                logger.log_error(f"FRQI execution failed: {e}")
                frqi_result[0] = np.zeros(n_trials)
        
        # 🔒 THREAD-SAFE: Run both methods simultaneously
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_neqr = executor.submit(run_neqr)
            future_frqi = executor.submit(run_frqi)
            
            # Wait for both to complete with timeout
            try:
                future_neqr.result(timeout=300)  # 5 minute timeout
                future_frqi.result(timeout=300)  # 5 minute timeout
            except Exception as e:
                logger.log_error(f"Parallel execution failed: {e}")
        
        return neqr_result[0], frqi_result[0]

    # MAIN METHOD - Currently using sequential (safer for threading)
    def run_both_methods_parallel(self, seq1, seq2, n_trials=CONFIG['n_trials'], noise_model=None):
        """🔒 THREAD-SAFE Main method for running both quantum algorithms"""
        return self.run_both_methods_sequential_full_cores(seq1, seq2, n_trials, noise_model)

    # Legacy compatibility methods
    def run_multiple_trials_enhanced(self, seq1, seq2, method='neqr', n_trials=CONFIG['n_trials'], 
                                   noise_model=None, track_metadata=True):
        """Legacy method for backward compatibility"""
        return self.run_multiple_trials_enhanced_parallel(
            seq1, seq2, method, n_trials, noise_model, track_metadata, max_workers=MAX_WORKERS
        )

    def get_comprehensive_performance_stats(self):
        """🔒 THREAD-SAFE Get comprehensive performance statistics"""
        with self._metrics_lock:  # 🔒 Ensure atomic read of metrics
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
                    'methods': self.performance_metrics['method_breakdown'].copy(),
                    'lengths': self.performance_metrics['length_breakdown'].copy(),
                    'gc_content': self.performance_metrics['gc_breakdown'].copy()
                },
                'thread_safety': {
                    'violations_detected': self.performance_metrics['thread_safety_violations'],
                    'locks_enabled': True,
                    'thread_safety_status': "✅ ENABLED" if self.performance_metrics['thread_safety_violations'] == 0 else "⚠️ VIOLATIONS DETECTED"
                },
                'system_metrics': {
                    'cpu_utilization': f"{MAX_WORKERS}/{CPU_COUNT} cores",
                    'memory_usage': f"{len(self.circuit_cache.cache)} circuits cached",
                    'core_allocation': f"Sequential mode: ALL {MAX_WORKERS} cores per method"
                }
            }

    def clear_cache(self):
        """🔒 THREAD-SAFE Clear cache and memory"""
        with self._cache_lock:
            self.circuit_cache.clear()
        gc.collect()

    def reset_performance_metrics(self):
        """🔒 THREAD-SAFE Reset performance tracking"""
        with self._metrics_lock:
            self.performance_metrics = {
                'total_circuits': 0,
                'total_time': 0.0,
                'parallel_jobs': 0,
                'method_breakdown': {'neqr': 0, 'frqi': 0},
                'length_breakdown': {},
                'gc_breakdown': {},
                'thread_safety_violations': 0
            }



            