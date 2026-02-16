#!/usr/bin/env python3
"""
Mitigation Comparison Test

Compares V1 (original) vs V2 (improved) PredictiveLattice
to measure impact of saturation mitigations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from typing import List, Dict, Tuple
from predictive_lattice import PredictiveLattice, LatticeConfig
from predictive_lattice_v2 import PredictiveLatticeV2, LatticeConfigV2


def create_collision_free_encoding(num_tokens: int, K: int, L: int) -> Dict[str, List[int]]:
    """Create token encoding with NO overlap between tokens."""
    token_to_neurons = {}
    for i in range(num_tokens):
        token = f"T{i}"
        start = i * K
        neurons = list(range(start, start + K))
        token_to_neurons[token] = neurons
    return token_to_neurons


def train_sequence(lattice, sequence: List[str], encoding: Dict, repetitions: int = 20):
    """Train a single sequence on either V1 or V2 lattice."""
    for _ in range(repetitions):
        lattice.reset_states(clear_trace=True)
        for token in sequence:
            neurons = encoding[token]
            for _ in range(3):
                u = np.zeros(lattice.config.size, dtype=np.complex128)
                for n in neurons:
                    u[n] = 1.0
                lattice.step(u)


def predict_next_token(lattice, current_token: str, encoding: Dict, steps: int = 5) -> str:
    """Predict next token using readout rule."""
    saved_z = lattice.z.copy()
    saved_e = lattice.e.copy()

    lattice.reset_states()
    neurons = encoding[current_token]

    # Phase-aligned stimulation
    for n in neurons:
        lattice.z[n] = 1.5

    for _ in range(steps):
        lattice.step(learn=False)

    # Readout
    token_scores = {}
    for token, token_neurons in encoding.items():
        if token == current_token:
            continue
        score = sum(np.abs(lattice.z[n]) for n in token_neurons)
        token_scores[token] = score

    lattice.z = saved_z
    lattice.e = saved_e

    if token_scores:
        return max(token_scores, key=token_scores.get)
    return None


def evaluate_sequences(lattice, sequences: List[List[str]], encoding: Dict) -> Dict:
    """Evaluate prediction accuracy."""
    correct = 0
    total = 0

    for seq in sequences:
        for i in range(len(seq) - 1):
            current = seq[i]
            expected = seq[i + 1]
            predicted = predict_next_token(lattice, current, encoding)

            if predicted == expected:
                correct += 1
            total += 1

    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total
    }


def get_edge_strength(lattice, source: int, target: int) -> float:
    """Get edge strength, handling both V1 and V2 interfaces."""
    if hasattr(lattice, 'get_edge_strength'):
        return lattice.get_edge_strength(source, target)
    elif hasattr(lattice, 'get_connection_strength'):
        return lattice.get_connection_strength(source, target)
    else:
        # Direct access
        if lattice.config.use_sparse:
            return float(np.abs(lattice.W[target, source]))
        return float(np.abs(lattice.W[target, source]))


def measure_snr(lattice, first_seq: List[str], encoding: Dict) -> float:
    """Measure signal-to-noise ratio for first sequence edges."""
    edge_strengths = []
    for i in range(len(first_seq) - 1):
        src_neurons = encoding[first_seq[i]]
        tgt_neurons = encoding[first_seq[i+1]]
        # Average edge strength
        strength = 0
        count = 0
        for s in src_neurons:
            for t in tgt_neurons:
                strength += get_edge_strength(lattice, s, t)
                count += 1
        edge_strengths.append(strength / count if count > 0 else 0)

    # Noise: edges to non-adjacent tokens
    noise_strengths = []
    src_neurons = encoding[first_seq[0]]
    for token in encoding:
        if token not in first_seq:
            tgt_neurons = encoding[token]
            strength = 0
            count = 0
            for s in src_neurons:
                for t in tgt_neurons:
                    strength += get_edge_strength(lattice, s, t)
                    count += 1
            noise_strengths.append(strength / count if count > 0 else 0)

    signal = np.mean(edge_strengths) if edge_strengths else 0
    noise = np.mean(noise_strengths) if noise_strengths else 0

    return signal / (noise + 1e-8)


def compare_versions():
    """Main comparison test."""
    print("=" * 70)
    print("V1 vs V2 COMPARISON TEST")
    print("=" * 70)

    # Parameters (reduced for faster testing)
    L = 256
    K = 8
    V = 25
    T = 3
    repetitions = 15

    # Create encoding (collision-free)
    encoding = create_collision_free_encoding(V, K, L)

    # Generate sequences
    np.random.seed(42)
    all_sequences = []
    for _ in range(20):
        seq = [f"T{np.random.randint(V)}" for _ in range(T)]
        all_sequences.append(seq)

    print(f"\nParameters: L={L}, K={K}, V={V}, T={T}")
    print(f"Collision-free encoding (isolating saturation effects)\n")

    # Test at different sequence counts
    sequence_counts = [1, 3, 5, 10, 15]

    print(f"{'Sequences':<10} {'V1 Acc':<10} {'V1 SNR':<10} {'V2 Acc':<10} {'V2 SNR':<10} {'Improvement':<12}")
    print("-" * 70)

    results = []

    for num_seq in sequence_counts:
        sequences = all_sequences[:num_seq]

        # V1 test
        config_v1 = LatticeConfig(size=L, learning_rate=0.2, trace_lambda=0.8)
        lattice_v1 = PredictiveLattice(config_v1)
        # Set identical frequencies (no drift)
        lattice_v1.omega = np.ones(L) * 1.0

        for seq in sequences:
            train_sequence(lattice_v1, seq, encoding, repetitions)

        eval_v1 = evaluate_sequences(lattice_v1, sequences, encoding)
        snr_v1 = measure_snr(lattice_v1, sequences[0], encoding)

        # V2 test
        config_v2 = LatticeConfigV2(
            size=L,
            learning_rate=0.2,
            trace_lambda=0.8,
            edge_decay=0.0001,
            max_edge_weight=0.5,
            top_k_sources=5,
            top_k_targets=5,
            max_out_degree=30,
            phase_lock=True
        )
        lattice_v2 = PredictiveLatticeV2(config_v2)

        for seq in sequences:
            train_sequence(lattice_v2, seq, encoding, repetitions)

        eval_v2 = evaluate_sequences(lattice_v2, sequences, encoding)
        snr_v2 = measure_snr(lattice_v2, sequences[0], encoding)

        improvement = eval_v2['accuracy'] - eval_v1['accuracy']

        print(f"{num_seq:<10} {eval_v1['accuracy']:<10.1%} {snr_v1:<10.2f} "
              f"{eval_v2['accuracy']:<10.1%} {snr_v2:<10.2f} {improvement:+.1%}")

        results.append({
            'sequences': num_seq,
            'v1_acc': eval_v1['accuracy'],
            'v1_snr': snr_v1,
            'v2_acc': eval_v2['accuracy'],
            'v2_snr': snr_v2,
            'improvement': improvement
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_v1 = np.mean([r['v1_acc'] for r in results])
    avg_v2 = np.mean([r['v2_acc'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])

    print(f"\nAverage V1 accuracy: {avg_v1:.1%}")
    print(f"Average V2 accuracy: {avg_v2:.1%}")
    print(f"Average improvement: {avg_improvement:+.1%}")

    if avg_v2 > avg_v1:
        print("\n✓ V2 mitigations show improvement")
        print("  Key changes:")
        print("  - Per-edge decay + cap (not row normalization)")
        print("  - Top-k gated updates")
        print("  - Bounded out-degree")
        print("  - Phase-locked stimulation")
    else:
        print("\n⚠ V2 did not show significant improvement")
        print("  May need parameter tuning or deeper architectural changes")

    return results


if __name__ == '__main__':
    compare_versions()
