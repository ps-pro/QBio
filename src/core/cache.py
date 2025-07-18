# -*- coding: utf-8 -*-
"""
Enhanced circuit cache system with thread safety
"""

import time
import threading
import numpy as np
from .encoders import NEQREncoder, FRQIEncoder

class EnhancedCircuitCache:
    """Advanced circuit cache with comprehensive tracking and thread safety"""
    
    def __init__(self, max_size=2000):
        self.cache = {}
        self.access_count = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.creation_time = {}
        self._lock = threading.Lock()  # ✅ ADD THREAD LOCK

    def get_circuit(self, seq1, seq2, method):
        key = (seq1, seq2, method)
        
        # Thread-safe cache access
        with self._lock:
            if key in self.cache:
                self.hits += 1
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
            else:
                self.misses += 1
        
        # Create circuit outside lock (expensive operation)
        start_time = time.time()
        if method == 'neqr':
            circuit = NEQREncoder().create_swap_test_circuit(seq1, seq2)
        else:
            circuit = FRQIEncoder().create_comparison_circuit(seq1, seq2)
        
        creation_time = time.time() - start_time
        
        # Thread-safe cache storage
        with self._lock:
            self.creation_time[key] = creation_time
            
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = circuit
            self.access_count[key] = 1
        
        return circuit

    def _evict_lru(self):
        """Evict least recently used circuits"""
        if self.access_count:
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
            if lru_key in self.creation_time:
                del self.creation_time[lru_key]

    def get_comprehensive_stats(self):
        with self._lock:  # ✅ Thread-safe stats access
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            avg_creation_time = np.mean(list(self.creation_time.values())) if self.creation_time else 0

            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'avg_creation_time': avg_creation_time,
                'total_circuits_created': len(self.creation_time)
            }

    def clear(self):
        """Clear cache and memory"""
        with self._lock:
            self.cache.clear()
            self.access_count.clear()
            self.creation_time.clear()