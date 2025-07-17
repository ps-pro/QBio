# -*- coding: utf-8 -*-
"""
Enhanced sequence analysis utilities
"""

import numpy as np

class EnhancedSequenceAnalyzer:
    """Enhanced sequence analysis with comprehensive functionality"""

    @staticmethod
    def calculate_hamming_similarity(seq1, seq2):
        if len(seq1) != len(seq2):
            raise ValueError("Sequences must have equal length")
        return sum(c1 == c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)

    @staticmethod
    def generate_random_sequence(length, gc_content=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if gc_content is None:
            return ''.join(np.random.choice(['A', 'T', 'G', 'C'], length))

        if not (0 <= gc_content <= 1):
            raise ValueError("gc_content must be between 0 and 1.")

        gc_count = int(length * gc_content)
        at_count = length - gc_count

        sequence = ['G'] * (gc_count // 2) + ['C'] * (gc_count - gc_count // 2)
        sequence.extend(['A'] * (at_count // 2) + ['T'] * (at_count - at_count // 2))

        np.random.shuffle(sequence)
        return ''.join(sequence)

    @staticmethod
    def create_controlled_similarity_sequence(reference_seq, target_similarity, seed=None):
        """Create a sequence with controlled similarity to reference"""
        if seed is not None:
            np.random.seed(seed)

        length = len(reference_seq)
        target_matches = int(length * target_similarity)

        new_seq = list(reference_seq)

        if target_matches < length:
            positions_to_change = np.random.choice(
                length, length - target_matches, replace=False
            )

            for pos in positions_to_change:
                nucleotides = ['A', 'T', 'G', 'C']
                nucleotides.remove(new_seq[pos])
                new_seq[pos] = np.random.choice(nucleotides)

        return ''.join(new_seq)

    @staticmethod
    def calculate_gc_content(sequence):
        """Calculate GC content of a sequence"""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)

    @staticmethod
    def calculate_sequence_complexity(sequence):
        """Calculate sequence complexity metrics"""
        length = len(sequence)
        unique_bases = len(set(sequence))

        # Calculate entropy
        base_counts = {base: sequence.count(base) for base in 'ATGC'}
        entropy = -sum((count/length) * np.log2(count/length)
                      for count in base_counts.values() if count > 0)

        return {
            'length': length,
            'unique_bases': unique_bases,
            'gc_content': EnhancedSequenceAnalyzer.calculate_gc_content(sequence),
            'entropy': entropy,
            'base_counts': base_counts
        }