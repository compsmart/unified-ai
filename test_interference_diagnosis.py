#!/usr/bin/env python3
"""
Interference Mode Diagnosis Suite

Separates the three root causes of sequence prediction degradation:
1. Weight saturation / normalization dilution
2. Concept overlap (collision)
3. Phase drift

Based on diagnostic methodology for coupled oscillator systems.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from predictive_lattice import PredictiveLattice, LatticeConfig


@dataclass
class TokenEncoding:
    """Maps tokens to neuron indices."""
    token_to_neurons: Dict[str, List[int]]
    neurons_to_token: Dict[int, str]  # For readout


class DiagnosticLattice(PredictiveLattice):
    """Extended lattice with diagnostic capabilities."""

    def __init__(self, config: LatticeConfig = None, omega_spread: float = 1.5):
        super().__init__(config)
        # Control frequency spread
        if omega_spread == 0:
            # Identical frequencies (no drift)
            self.omega = np.ones(self.config.size) * 1.0
        else:
            # Random frequencies with controlled spread
            self.omega = np.random.uniform(1.0 - omega_spread/2, 1.0 + omega_spread/2, self.config.size)

    def get_coherence(self, active_indices: List[int]) -> float:
        """
        Compute phase coherence R for active neurons.
        R = |1/m * sum(e^(i*phi_k))| for k in active set
        R=1 means perfect sync, R≈0 means random phases
        """
        if not active_indices:
            return 0.0
        phases = np.angle(self.z[active_indices])
        R = np.abs(np.mean(np.exp(1j * phases)))
        return float(R)

    def get_edge_strength(self, source_neurons: List[int], target_neurons: List[int]) -> float:
        """Get average connection strength from source to target neuron groups."""
        if self.config.use_sparse:
            W = self.W.toarray()
        else:
            W = self.W

        total = 0.0
        count = 0
        for t in target_neurons:
            for s in source_neurons:
                total += np.abs(W[t, s])
                count += 1
        return total / count if count > 0 else 0.0


def create_random_encoding(num_tokens: int, K: int, L: int) -> TokenEncoding:
    """Create token encoding with random (potentially overlapping) neuron assignments."""
    token_to_neurons = {}
    neurons_to_token = {}

    for i in range(num_tokens):
        token = f"T{i}"
        neurons = np.random.choice(L, K, replace=False).tolist()
        token_to_neurons[token] = neurons
        for n in neurons:
            neurons_to_token[n] = token  # Last write wins for overlaps

    return TokenEncoding(token_to_neurons, neurons_to_token)


def create_collision_free_encoding(num_tokens: int, K: int, L: int) -> TokenEncoding:
    """Create token encoding with NO overlap between tokens."""
    if num_tokens * K > L:
        raise ValueError(f"Cannot fit {num_tokens} tokens x {K} neurons in {L} lattice")

    token_to_neurons = {}
    neurons_to_token = {}

    for i in range(num_tokens):
        token = f"T{i}"
        start = i * K
        neurons = list(range(start, start + K))
        token_to_neurons[token] = neurons
        for n in neurons:
            neurons_to_token[n] = token

    return TokenEncoding(token_to_neurons, neurons_to_token)


def calculate_overlap_stats(encoding: TokenEncoding) -> Dict:
    """Calculate overlap statistics for an encoding."""
    tokens = list(encoding.token_to_neurons.keys())
    overlaps = []

    for i, t1 in enumerate(tokens):
        for t2 in tokens[i+1:]:
            n1 = set(encoding.token_to_neurons[t1])
            n2 = set(encoding.token_to_neurons[t2])
            overlap = len(n1 & n2)
            overlaps.append(overlap)

    return {
        'mean_overlap': np.mean(overlaps) if overlaps else 0,
        'max_overlap': max(overlaps) if overlaps else 0,
        'pairs_with_overlap': sum(1 for o in overlaps if o > 0),
        'total_pairs': len(overlaps)
    }


def generate_sequences(num_sequences: int, seq_length: int, num_tokens: int) -> List[List[str]]:
    """Generate random sequences from token vocabulary."""
    sequences = []
    for _ in range(num_sequences):
        # Each sequence uses random tokens (allows overlap between sequences)
        seq = [f"T{np.random.randint(num_tokens)}" for _ in range(seq_length)]
        sequences.append(seq)
    return sequences


def train_sequence(lattice: DiagnosticLattice, sequence: List[str],
                   encoding: TokenEncoding, repetitions: int = 20):
    """Train a single sequence."""
    for _ in range(repetitions):
        lattice.reset_states(clear_trace=True)
        for token in sequence:
            neurons = encoding.token_to_neurons[token]
            for _ in range(3):  # Sustain each token
                u = np.zeros(lattice.config.size, dtype=np.complex128)
                for n in neurons:
                    u[n] = 1.0
                lattice.step(u)


def predict_next_token(lattice: DiagnosticLattice, current_token: str,
                       encoding: TokenEncoding, steps: int = 5) -> Tuple[str, float, float]:
    """
    Predict next token and return (predicted_token, margin, coherence).

    Readout rule: predicted token = argmax over token groups of summed amplitude
    """
    # Save state
    saved_z = lattice.z.copy()
    saved_e = lattice.e.copy()

    # Reset and stimulate current token
    lattice.reset_states()
    neurons = encoding.token_to_neurons[current_token]

    # Phase-aligned stimulation (all neurons at same phase)
    for n in neurons:
        lattice.z[n] = 1.5  # Real-valued = phase 0

    # Run forward
    for _ in range(steps):
        lattice.step(learn=False)

    # Readout: score each token
    token_scores = {}
    for token, token_neurons in encoding.token_to_neurons.items():
        if token == current_token:
            continue  # Skip current token
        score = sum(np.abs(lattice.z[n]) for n in token_neurons)
        token_scores[token] = score

    # Get coherence of winner's neurons
    if token_scores:
        sorted_tokens = sorted(token_scores.items(), key=lambda x: -x[1])
        predicted = sorted_tokens[0][0]
        best_score = sorted_tokens[0][1]
        second_score = sorted_tokens[1][1] if len(sorted_tokens) > 1 else 0
        margin = best_score - second_score

        # Coherence of predicted token's neurons
        winner_neurons = encoding.token_to_neurons[predicted]
        coherence = lattice.get_coherence(winner_neurons)
    else:
        predicted = None
        margin = 0
        coherence = 0

    # Restore state
    lattice.z = saved_z
    lattice.e = saved_e

    return predicted, margin, coherence


def evaluate_sequences(lattice: DiagnosticLattice, sequences: List[List[str]],
                       encoding: TokenEncoding) -> Dict:
    """Evaluate prediction accuracy on all sequences."""
    correct = 0
    total = 0
    margins = []
    coherences = []

    for seq in sequences:
        for i in range(len(seq) - 1):
            current = seq[i]
            expected = seq[i + 1]

            predicted, margin, coherence = predict_next_token(lattice, current, encoding)

            if predicted == expected:
                correct += 1
            total += 1
            margins.append(margin)
            coherences.append(coherence)

    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'mean_margin': np.mean(margins),
        'mean_coherence': np.mean(coherences),
        'std_margin': np.std(margins),
        'std_coherence': np.std(coherences)
    }


def test_interference_conditions():
    """
    Run the 4-condition interference test:
    A: random ω, random tokens (baseline)
    B: ω identical, random tokens (tests drift)
    C: random ω, collision-free tokens (tests overlap)
    D: ω identical, collision-free tokens (upper bound)
    """
    print("=" * 70)
    print("INTERFERENCE MODE SEPARATION TEST")
    print("=" * 70)

    # Parameters
    L = 256       # Lattice size
    K = 10        # Neurons per token
    V = 20        # Number of tokens (vocabulary)
    S = 10        # Number of sequences
    T = 4         # Sequence length

    print(f"\nParameters: L={L}, K={K}, V={V}, S={S}, T={T}")
    print(f"Theoretical overlap probability: {1 - (1 - K/L)**(V*K):.1%}")

    # Generate same sequences for all conditions
    np.random.seed(42)
    sequences = generate_sequences(S, T, V)

    results = {}

    conditions = [
        ("A: random ω, random tokens (baseline)", 1.5, "random"),
        ("B: ω identical, random tokens (drift test)", 0.0, "random"),
        ("C: random ω, collision-free tokens (overlap test)", 1.5, "collision_free"),
        ("D: ω identical, collision-free (upper bound)", 0.0, "collision_free"),
    ]

    for name, omega_spread, encoding_type in conditions:
        print(f"\n{name}")
        print("-" * 60)

        # Create encoding
        np.random.seed(42)  # Same random encoding for A and B
        if encoding_type == "random":
            encoding = create_random_encoding(V, K, L)
        else:
            encoding = create_collision_free_encoding(V, K, L)

        overlap_stats = calculate_overlap_stats(encoding)
        print(f"Encoding overlap: mean={overlap_stats['mean_overlap']:.2f}, "
              f"pairs with overlap={overlap_stats['pairs_with_overlap']}/{overlap_stats['total_pairs']}")

        # Create lattice
        config = LatticeConfig(size=L, learning_rate=0.2, trace_lambda=0.8)
        lattice = DiagnosticLattice(config, omega_spread=omega_spread)

        print(f"Frequency spread: {'identical' if omega_spread == 0 else f'±{omega_spread/2:.2f}'}")

        # Train all sequences
        for seq in sequences:
            train_sequence(lattice, seq, encoding, repetitions=20)

        # Evaluate
        eval_results = evaluate_sequences(lattice, sequences, encoding)
        results[name[:1]] = eval_results

        print(f"Accuracy: {eval_results['accuracy']:.1%} ({eval_results['correct']}/{eval_results['total']})")
        print(f"Mean margin: {eval_results['mean_margin']:.3f} ± {eval_results['std_margin']:.3f}")
        print(f"Mean coherence: {eval_results['mean_coherence']:.3f} ± {eval_results['std_coherence']:.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    A, B, C, D = results['A'], results['B'], results['C'], results['D']

    drift_effect = B['accuracy'] - A['accuracy']
    overlap_effect = C['accuracy'] - A['accuracy']
    combined_effect = D['accuracy'] - A['accuracy']

    print(f"\nBaseline (A): {A['accuracy']:.1%}")
    print(f"Fixing drift only (B-A): {drift_effect:+.1%}")
    print(f"Fixing overlap only (C-A): {overlap_effect:+.1%}")
    print(f"Fixing both (D-A): {combined_effect:+.1%}")

    print("\nDominant interference mode:")
    if abs(overlap_effect) > abs(drift_effect) * 1.5:
        print("  → OVERLAP is the primary bottleneck")
        print("  → Fix: Increase L, reduce K, or use collision-free encoding")
    elif abs(drift_effect) > abs(overlap_effect) * 1.5:
        print("  → PHASE DRIFT is the primary bottleneck")
        print("  → Fix: Lock frequencies, use phase reset on stimulation")
    elif combined_effect > max(drift_effect, overlap_effect) * 1.2:
        print("  → BOTH factors contribute significantly")
        print("  → Fix: Address both overlap and phase coherence")
    else:
        print("  → SATURATION/NORMALIZATION is likely dominant")
        print("  → Fix: Change normalization strategy, bound out-degree")

    return results


def test_capacity_curve():
    """Test accuracy vs number of stored sequences."""
    print("\n" + "=" * 70)
    print("CAPACITY CURVE (Accuracy vs #Sequences)")
    print("=" * 70)

    L = 512  # Larger lattice
    K = 8    # Fewer neurons per token
    V = 50   # More tokens for more sequences
    T = 4

    # Use collision-free to isolate saturation effect
    encoding = create_collision_free_encoding(V, K, L)

    print(f"\nParameters: L={L}, K={K}, V={V}, T={T}")
    print("Using collision-free encoding to isolate saturation effects\n")

    config = LatticeConfig(size=L, learning_rate=0.2, trace_lambda=0.8)
    lattice = DiagnosticLattice(config, omega_spread=0)  # No drift

    np.random.seed(42)
    all_sequences = generate_sequences(30, T, V)

    results = []

    for num_seq in [1, 2, 5, 10, 15, 20, 25, 30]:
        sequences = all_sequences[:num_seq]

        # Reset and train
        lattice.W = lattice.W * 0  # Clear weights
        for seq in sequences:
            train_sequence(lattice, seq, encoding, repetitions=20)

        # Evaluate
        eval_results = evaluate_sequences(lattice, sequences, encoding)
        results.append((num_seq, eval_results['accuracy'], eval_results['mean_margin']))

        print(f"S={num_seq:2d}: accuracy={eval_results['accuracy']:.1%}, "
              f"margin={eval_results['mean_margin']:.3f}")

    # Analyze collapse pattern
    print("\nCollapse analysis:")
    for i in range(1, len(results)):
        prev_acc = results[i-1][1]
        curr_acc = results[i][1]
        if curr_acc < prev_acc - 0.1:
            print(f"  Sharp drop at S={results[i][0]}: {prev_acc:.1%} → {curr_acc:.1%}")

    return results


def test_edge_strength_analysis():
    """Analyze weight saturation by tracking edge strengths."""
    print("\n" + "=" * 70)
    print("EDGE STRENGTH ANALYSIS")
    print("=" * 70)

    L = 256
    K = 10
    V = 20
    T = 3

    encoding = create_collision_free_encoding(V, K, L)
    config = LatticeConfig(size=L, learning_rate=0.2, trace_lambda=0.8)
    lattice = DiagnosticLattice(config, omega_spread=0)

    np.random.seed(42)
    sequences = generate_sequences(15, T, V)

    print(f"\nTracking edge strengths as sequences are added...\n")

    # Track edges for first sequence
    first_seq = sequences[0]

    for num_seq in [1, 5, 10, 15]:
        # Reset and train
        lattice.W = lattice.W * 0
        for seq in sequences[:num_seq]:
            train_sequence(lattice, seq, encoding, repetitions=20)

        # Measure edge strength for first sequence's transitions
        edge_strengths = []
        for i in range(len(first_seq) - 1):
            src_neurons = encoding.token_to_neurons[first_seq[i]]
            tgt_neurons = encoding.token_to_neurons[first_seq[i+1]]
            strength = lattice.get_edge_strength(src_neurons, tgt_neurons)
            edge_strengths.append(strength)

        # Measure competing edges (noise)
        noise_strengths = []
        src_neurons = encoding.token_to_neurons[first_seq[0]]
        for token in encoding.token_to_neurons:
            if token not in first_seq:
                tgt_neurons = encoding.token_to_neurons[token]
                strength = lattice.get_edge_strength(src_neurons, tgt_neurons)
                noise_strengths.append(strength)

        signal = np.mean(edge_strengths)
        noise = np.mean(noise_strengths) if noise_strengths else 0
        snr = signal / (noise + 1e-8)

        print(f"S={num_seq:2d}: signal={signal:.4f}, noise={noise:.4f}, SNR={snr:.2f}x")

    print("\nIf SNR decreases sharply → saturation/normalization is diluting edges")


def run_full_diagnosis():
    """Run all diagnostic tests."""
    print("\n" + "=" * 70)
    print("FULL INTERFERENCE DIAGNOSIS SUITE")
    print("=" * 70)
    print("This suite separates three root causes:")
    print("  1. Weight saturation / normalization dilution")
    print("  2. Concept overlap (collision)")
    print("  3. Phase drift")
    print("=" * 70)

    # Run tests
    interference_results = test_interference_conditions()
    capacity_results = test_capacity_curve()
    test_edge_strength_analysis()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    A = interference_results['A']['accuracy']
    D = interference_results['D']['accuracy']

    print(f"\nBaseline accuracy: {A:.1%}")
    print(f"Best case (no drift, no overlap): {D:.1%}")
    print(f"Improvement potential: {D - A:+.1%}")

    if D > 0.9:
        print("\n✓ High upper bound achieved - system CAN work with proper configuration")
    else:
        print("\n⚠ Even optimal conditions don't reach 90% - deeper architectural issues")

    return {
        'interference': interference_results,
        'capacity': capacity_results
    }


if __name__ == '__main__':
    run_full_diagnosis()
