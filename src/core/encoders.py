# -*- coding: utf-8 -*-
"""
Quantum DNA encoding implementations - CORRECT for Qiskit 0.40
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

class QuantumDNAEncoder:
    """Base class for quantum DNA encoding methods"""

    @staticmethod
    def nucleotide_to_binary(nucleotide):
        mapping = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
        return mapping.get(nucleotide.upper(), '00')

    @staticmethod
    def nucleotide_to_angle(nucleotide):
        mapping = {'A': np.pi, 'C': np.pi / 2, 'T': np.pi / 6, 'G': 0}
        return mapping.get(nucleotide.upper(), 0)

class NEQREncoder(QuantumDNAEncoder):
    """Enhanced NEQR implementation - CORRECT for Qiskit 0.40"""

    def create_swap_test_circuit(self, seq1, seq2):
        L = len(seq1)
        n_pos = max(1, int(np.ceil(np.log2(L))))
        n_val = 2

        def create_encoding_gate(sequence):
            pos_reg_inner = QuantumRegister(n_pos)
            val_reg_inner = QuantumRegister(n_val)
            qc_enc = QuantumCircuit(pos_reg_inner, val_reg_inner, name=f'NEQR_Enc')
            qc_enc.h(pos_reg_inner)

            for i, nucleotide in enumerate(sequence):
                if i >= 2**n_pos:
                    break

                pos_bin = format(i, f'0{n_pos}b')
                nuc_bin = self.nucleotide_to_binary(nucleotide)

                for j, bit in enumerate(pos_bin):
                    if bit == '0': qc_enc.x(pos_reg_inner[j])

                # ✅ CORRECT: Use original mcx method for Qiskit 0.40
                for j, bit in enumerate(nuc_bin):
                    if bit == '1': qc_enc.mcx(pos_reg_inner[:], val_reg_inner[j])

                for j, bit in enumerate(pos_bin):
                    if bit == '0': qc_enc.x(pos_reg_inner[j])

            return qc_enc.to_gate()

        n_qubits_per_seq = n_pos + n_val
        ancilla = QuantumRegister(1, 'ancilla')
        reg1 = QuantumRegister(n_qubits_per_seq, 'seq1')
        reg2 = QuantumRegister(n_qubits_per_seq, 'seq2')
        creg = ClassicalRegister(1, 'c')

        qc = QuantumCircuit(ancilla, reg1, reg2, creg)
        qc.h(ancilla[0])
        qc.append(create_encoding_gate(seq1), reg1[:])
        qc.append(create_encoding_gate(seq2), reg2[:])

        for i in range(n_qubits_per_seq):
            qc.cswap(ancilla[0], reg1[i], reg2[i])

        qc.h(ancilla[0])
        qc.measure(ancilla[0], creg[0])

        return qc

class FRQIEncoder(QuantumDNAEncoder):
    """Enhanced FRQI implementation - CORRECT for Qiskit 0.40"""

    def create_comparison_circuit(self, seq1, seq2):
        L = len(seq1)
        n_idx = max(1, int(np.ceil(np.log2(L))))

        strip = QuantumRegister(1, 'strip')
        idx = QuantumRegister(n_idx, 'idx')
        dna = QuantumRegister(1, 'dna')
        creg = ClassicalRegister(1, 'c')

        qc = QuantumCircuit(strip, idx, dna, creg)
        qc.h(strip[0])
        qc.h(idx)

        for i in range(min(L, 2**n_idx)):
            pos_bin = format(i, f'0{n_idx}b')

            for j, bit in enumerate(pos_bin):
                if bit == '0': qc.x(idx[j])

            angle1 = self.nucleotide_to_angle(seq1[i])
            angle2 = self.nucleotide_to_angle(seq2[i])

            controls = [strip[0]] + list(idx[:])
            
            # ✅ CORRECT: Use original mcry method for Qiskit 0.40
            qc.x(strip[0])
            if angle1 != 0: qc.mcry(angle1, controls, dna[0])
            qc.x(strip[0])
            if angle2 != 0: qc.mcry(angle2, controls, dna[0])

            for j, bit in enumerate(pos_bin):
                if bit == '0': qc.x(idx[j])

        qc.h(strip[0])
        qc.measure(strip[0], creg[0])

        return qc