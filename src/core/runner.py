# -*- coding: utf-8 -*-
"""
Enhanced CPU simulation runner
"""

import time
import gc
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from qiskit import transpile
from qiskit_aer import AerSimulator
from .cache import EnhancedCircuitCache
from ..config.settings import CONFIG, CPU_COUNT, MAX_WORKERS
from ..utils.logging import logger

class EnhancedCPURunner:
    """Enhanced CPU simulation runner with comprehensive tracking"""

    def __init__(self, shots=CONFIG['shots']):
        self.shots = shots
        self.circuit_cache = EnhancedCircuitCache(max_size=3000)

        # Initialize optimized CPU backend
        self.backend = AerSimulator(method='statevector', device='CPU')
        self.backend.set_options(max_parallel_threads=CPU_COUNT)

        print(f"🚀 Enhanced CPU Backend Initialized!")
        print(f"   - Max parallel threads: {CPU_COUNT}")
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

    def run_circuits_parallel_batch(self, circuit_configs, max_workers=MAX_WORKERS):
        """Enhanced parallel batch processing"""
        start_time = time.time()
        all_results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {}
            for config in circuit_configs:
                seq1, seq2, method, noise_model = config
                circuit = self.circuit_cache.get_circuit(seq1, seq2, method)
                future = executor.submit(self.run_single_circuit_optimized, circuit, noise_model)
                future_to_config[future] = config

            for future in tqdm(as_completed(future_to_config),
                                            total=len(circuit_configs),
                                            desc="Processing Circuits",
                                            leave=False,
                                            ncols=120,
                                            ascii=False,
                                            position=1,
                                            bar_format='{desc}: {percentage:3.0f}%|{bar:60}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
                try:
                    counts = future.result(timeout=60)
                    all_results.append(counts)

                    # Track method usage
                    config = future_to_config[future]
                    method = config[2]
                    self.performance_metrics['method_breakdown'][method] += 1

                except Exception as e:
                    logger.log_error(f"Batch job failed: {e}")
                    all_results.append({'0': 0, '1': 0})

        # Update performance tracking
        self.performance_metrics['total_circuits'] += len(circuit_configs)
        self.performance_metrics['parallel_jobs'] += 1

        return all_results

    def run_multiple_trials_enhanced(self, seq1, seq2, method='neqr', n_trials=CONFIG['n_trials'],
                                   noise_model=None, track_metadata=True):
        """Enhanced multiple trials with metadata tracking"""

        # Track sequence metadata
        if track_metadata:
            length = len(seq1)
            gc_content = (seq1.count('G') + seq1.count('C')) / length

            self.performance_metrics['length_breakdown'][length] = \
                self.performance_metrics['length_breakdown'].get(length, 0) + 1

            gc_key = f"{gc_content:.1f}"
            self.performance_metrics['gc_breakdown'][gc_key] = \
                self.performance_metrics['gc_breakdown'].get(gc_key, 0) + 1

        # Create batch configurations
        circuit_configs = [(seq1, seq2, method, noise_model) for _ in range(n_trials)]

        # Run all trials in parallel
        all_counts = self.run_circuits_parallel_batch(circuit_configs)

        # Convert to similarity scores
        similarities = []
        for counts in all_counts:
            sim = self.calculate_similarity_from_counts(counts)
            similarities.append(sim)

        return np.array(similarities)

    def calculate_similarity_from_counts(self, counts):
        """Convert measurement counts to similarity score"""
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0
        p0 = counts.get('0', 0) / total_shots
        return 2 * p0 - 1  # Convert to [-1, 1] range

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
                'memory_usage': f"{len(self.circuit_cache.cache)} circuits cached"
            }
        }

    def clear_cache(self):
        """Clear cache and memory"""
        self.circuit_cache.clear()
        gc.collect()