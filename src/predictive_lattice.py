"""
PredictiveLattice - Fast Reflexes Module

The "spine" of the Unified AI - handles real-time pattern detection
using coupled oscillators with directional Hebbian learning.

Key Features:
- Complex oscillator network with eligibility traces
- Directional prediction (A activates B, not vice versa)
- Microsecond inference time
- Memory efficient (~32KB for sparse patterns)

Mathematical Foundation:
- State: z_i(t) = A_i * e^(i*phi_i) (complex oscillators)
- Eligibility: e_i(t) = lambda*e_i(t-1) + (1-lambda)*z_i(t)
- Learning: dW_ji = z_j(now) * conj(e_i(past))
- Stability: ||W_i|| <= 1 via row normalization
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
import json
import os


@dataclass
class LatticeConfig:
    """Configuration for the predictive lattice."""
    size: int = 256              # Number of oscillator nodes
    learning_rate: float = 0.15  # Causal link formation rate
    decay: float = 0.02          # Amplitude damping
    trace_lambda: float = 0.8    # Eligibility trace decay
    threshold: float = 0.1       # Activation threshold
    dt: float = 0.1              # Time step
    use_sparse: bool = True      # Use sparse weight storage


class PredictiveLattice:
    """
    Fast-reflex lattice for real-time pattern detection.

    Uses coupled oscillators with eligibility traces to create
    directional predictions. When pattern A is followed by B,
    future occurrences of A will predict/activate B.
    """

    def __init__(self, config: LatticeConfig = None):
        self.config = config or LatticeConfig()
        n = self.config.size

        # Complex oscillator states
        self.z = np.zeros(n, dtype=np.complex128)

        # Eligibility trace (sliding memory of recent activations)
        self.e = np.zeros(n, dtype=np.complex128)

        # Natural frequencies (each neuron oscillates at its own rhythm)
        self.omega = np.random.uniform(0.5, 2.0, n)

        # Weight matrix - sparse or dense
        if self.config.use_sparse:
            self.W = lil_matrix((n, n), dtype=np.complex128)
        else:
            self.W = np.zeros((n, n), dtype=np.complex128)

        # Statistics
        self.total_steps = 0
        self.active_history = []

    def step(
        self,
        u: np.ndarray = None,
        learn: bool = True
    ) -> np.ndarray:
        """
        Advance the lattice by one time step.

        Args:
            u: External input signal (complex array of size n)
            learn: Whether to update weights

        Returns:
            Amplitudes of all oscillators
        """
        n = self.config.size
        dt = self.config.dt

        if u is None:
            u = np.zeros(n, dtype=np.complex128)

        # 1. Natural Rotation - each oscillator spins at its frequency
        self.z *= np.exp(1j * self.omega * dt)

        # 2. Coupling from other neurons + external input
        if self.config.use_sparse:
            coupling = np.array(self.W.dot(self.z)).flatten()
        else:
            coupling = np.dot(self.W, self.z)

        drive = coupling + u

        # 3. Update state with damping
        self.z = (self.z + drive * dt) * (1 - self.config.decay)

        # 4. Causal learning via eligibility trace
        if learn:
            self._causal_update()

        # 5. Update eligibility trace
        lam = self.config.trace_lambda
        self.e = lam * self.e + (1 - lam) * self.z

        # 6. Stability via normalization
        self._normalize()

        self.total_steps += 1

        return np.abs(self.z)

    def _causal_update(self):
        """
        Apply causal learning rule.

        W[target, source] += lr * z[target](now) * conj(e[source](past))

        This creates directional links: if source was active in the past
        and target is active now, strengthen the source->target connection.
        """
        lr = self.config.learning_rate
        threshold = self.config.threshold
        dt = self.config.dt

        # Find active neurons
        z_amp = np.abs(self.z)
        e_amp = np.abs(self.e)

        target_mask = z_amp > threshold
        source_mask = e_amp > threshold * 0.5

        if not np.any(target_mask) or not np.any(source_mask):
            return

        # Get indices
        targets = np.where(target_mask)[0]
        sources = np.where(source_mask)[0]

        # Update only relevant connections
        for t in targets:
            for s in sources:
                if t != s:  # No self-connections
                    update = self.z[t] * np.conj(self.e[s]) * lr * dt
                    if self.config.use_sparse:
                        self.W[t, s] += update
                    else:
                        self.W[t, s] += update

        # Record active count
        self.active_history.append(len(targets))

    def _normalize(self):
        """
        Normalize weights and states to prevent explosion.
        """
        # State normalization
        max_amp = 2.0
        state_norms = np.abs(self.z)
        overflow = state_norms > max_amp
        if np.any(overflow):
            self.z[overflow] *= max_amp / (state_norms[overflow] + 1e-8)

        # Weight row normalization
        if self.config.use_sparse:
            # Convert to CSR for efficient row operations
            W_csr = self.W.tocsr()
            for i in range(self.config.size):
                row_start = W_csr.indptr[i]
                row_end = W_csr.indptr[i + 1]
                if row_end > row_start:
                    row_data = W_csr.data[row_start:row_end]
                    norm = np.sqrt(np.sum(np.abs(row_data) ** 2))
                    if norm > 1.0:
                        W_csr.data[row_start:row_end] /= norm
            self.W = W_csr.tolil()
        else:
            norms = np.linalg.norm(self.W, axis=1, keepdims=True)
            self.W = np.where(norms > 1.0, self.W / (norms + 1e-8), self.W)

    def stimulate(
        self,
        indices: List[int],
        amplitude: float = 1.0,
        phase: float = 0.0
    ):
        """Activate specific neurons."""
        for idx in indices:
            if 0 <= idx < self.config.size:
                self.z[idx] = amplitude * np.exp(1j * phase)

    def reset_states(self, clear_trace: bool = True):
        """Reset states but keep learned weights."""
        self.z = np.zeros(self.config.size, dtype=np.complex128)
        if clear_trace:
            self.e = np.zeros(self.config.size, dtype=np.complex128)

    def clear_trace(self):
        """Clear eligibility trace (between sequences)."""
        self.e = np.zeros(self.config.size, dtype=np.complex128)

    def get_amplitudes(self) -> np.ndarray:
        """Get oscillator amplitudes."""
        return np.abs(self.z)

    def get_phases(self) -> np.ndarray:
        """Get oscillator phases."""
        return np.angle(self.z)

    def get_state_vector(self) -> np.ndarray:
        """Get full complex state for bridging."""
        return self.z.copy()

    def get_top_active(self, k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k most active neurons."""
        amps = np.abs(self.z)
        indices = np.argsort(amps)[::-1][:k]
        return [(int(i), float(amps[i])) for i in indices if amps[i] > self.config.threshold]

    def predict_next(self, current_indices: List[int], steps: int = 5) -> List[Tuple[int, float]]:
        """
        Predict which neurons will activate next given current active set.
        """
        # Save current state
        saved_z = self.z.copy()
        saved_e = self.e.copy()

        # Reset and stimulate
        self.reset_states()
        self.stimulate(current_indices, amplitude=1.5)

        # Run forward
        for _ in range(steps):
            self.step(learn=False)

        # Get predictions
        predictions = self.get_top_active(k=20)

        # Filter out input neurons
        predictions = [(i, a) for i, a in predictions if i not in current_indices]

        # Restore state
        self.z = saved_z
        self.e = saved_e

        return predictions

    def get_connection_strength(self, source: int, target: int) -> float:
        """Get strength of connection from source to target."""
        if self.config.use_sparse:
            return abs(self.W[target, source])
        return abs(self.W[target, source])

    def get_sparsity(self) -> float:
        """Calculate weight matrix sparsity."""
        threshold = 1e-4
        if self.config.use_sparse:
            W_dense = self.W.toarray()
        else:
            W_dense = self.W
        total = W_dense.size
        nonzero = np.sum(np.abs(W_dense) > threshold)
        return 1.0 - (nonzero / total)

    def get_stats(self) -> Dict:
        """Get lattice statistics."""
        return {
            'size': self.config.size,
            'total_steps': self.total_steps,
            'sparsity': self.get_sparsity(),
            'active_avg': np.mean(self.active_history) if self.active_history else 0,
            'memory_bytes': self._estimate_memory()
        }

    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        n = self.config.size

        # States: z, e, omega
        state_bytes = n * 16 * 2 + n * 8  # complex128 + float64

        # Weights
        if self.config.use_sparse:
            nnz = self.W.nnz
            weight_bytes = nnz * (4 + 4 + 16)  # row, col, complex value
        else:
            weight_bytes = n * n * 16

        return state_bytes + weight_bytes

    def save(self, path: str):
        """Save lattice state."""
        if self.config.use_sparse:
            W_csr = self.W.tocsr()
            weights = {
                'data_real': W_csr.data.real.tolist(),
                'data_imag': W_csr.data.imag.tolist(),
                'indices': W_csr.indices.tolist(),
                'indptr': W_csr.indptr.tolist(),
                'shape': list(W_csr.shape)
            }
        else:
            weights = {
                'W_real': self.W.real.tolist(),
                'W_imag': self.W.imag.tolist()
            }

        data = {
            'config': {
                'size': self.config.size,
                'learning_rate': self.config.learning_rate,
                'decay': self.config.decay,
                'trace_lambda': self.config.trace_lambda,
                'threshold': self.config.threshold,
                'dt': self.config.dt,
                'use_sparse': self.config.use_sparse
            },
            'omega': self.omega.tolist(),
            'weights': weights,
            'total_steps': self.total_steps
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load lattice state."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.config = LatticeConfig(**data['config'])
        n = self.config.size

        self.z = np.zeros(n, dtype=np.complex128)
        self.e = np.zeros(n, dtype=np.complex128)
        self.omega = np.array(data['omega'])

        if self.config.use_sparse:
            w = data['weights']
            data_complex = np.array(w['data_real']) + 1j * np.array(w['data_imag'])
            W_csr = csr_matrix(
                (data_complex, w['indices'], w['indptr']),
                shape=tuple(w['shape'])
            )
            self.W = W_csr.tolil()
        else:
            w = data['weights']
            self.W = np.array(w['W_real']) + 1j * np.array(w['W_imag'])

        self.total_steps = data['total_steps']


class AnomalyDetector:
    """
    Uses the PredictiveLattice to detect anomalies in patterns.

    When the lattice's predictions don't match reality,
    that's an anomaly worth reporting to higher layers.
    """

    def __init__(self, lattice: PredictiveLattice, anomaly_threshold: float = 0.5):
        self.lattice = lattice
        self.anomaly_threshold = anomaly_threshold
        self.history = []

    def check_anomaly(
        self,
        observed_indices: List[int],
        expected_indices: List[int] = None
    ) -> Tuple[bool, float, Dict]:
        """
        Check if observed pattern is anomalous given predictions.

        Returns:
            (is_anomaly, anomaly_score, details)
        """
        if expected_indices is None:
            # Use lattice's internal prediction
            predictions = self.lattice.get_top_active(k=10)
            expected_indices = [i for i, _ in predictions]

        # Calculate overlap
        observed_set = set(observed_indices)
        expected_set = set(expected_indices)

        if not expected_set:
            # No prediction = no anomaly judgement
            return False, 0.0, {'reason': 'no_prediction'}

        overlap = len(observed_set & expected_set)
        union = len(observed_set | expected_set)

        # Jaccard similarity
        jaccard = overlap / union if union > 0 else 0

        # Anomaly if low overlap
        anomaly_score = 1.0 - jaccard
        is_anomaly = anomaly_score > self.anomaly_threshold

        details = {
            'observed': list(observed_set),
            'expected': list(expected_set),
            'overlap': overlap,
            'jaccard': jaccard,
            'anomaly_score': anomaly_score
        }

        self.history.append({
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'details': details
        })

        return is_anomaly, anomaly_score, details


def test_chain_resonance():
    """Test A->B->C chain propagation."""
    print("=" * 60)
    print("CHAIN RESONANCE TEST (A -> B -> C)")
    print("=" * 60)

    config = LatticeConfig(size=10, learning_rate=0.25, trace_lambda=0.85)
    lattice = PredictiveLattice(config)

    A, B, C = 0, 1, 2

    # Train A -> B
    print("\nTraining A -> B...")
    for _ in range(100):
        lattice.reset_states(clear_trace=True)
        for _ in range(5):
            u = np.zeros(10, dtype=np.complex128)
            u[A] = 1.0
            lattice.step(u)
        for _ in range(5):
            u = np.zeros(10, dtype=np.complex128)
            u[B] = 1.0
            lattice.step(u)

    # Train B -> C
    print("Training B -> C...")
    for _ in range(100):
        lattice.reset_states(clear_trace=True)
        for _ in range(5):
            u = np.zeros(10, dtype=np.complex128)
            u[B] = 1.0
            lattice.step(u)
        for _ in range(5):
            u = np.zeros(10, dtype=np.complex128)
            u[C] = 1.0
            lattice.step(u)

    # Test chain
    print("\nTesting chain (trigger A only)...")
    lattice.reset_states()
    lattice.stimulate([A], amplitude=1.5)

    for step in range(30):
        u = np.zeros(10, dtype=np.complex128)
        if step < 5:
            u[A] = 0.5
        lattice.step(learn=False)

        if step % 5 == 0:
            amps = lattice.get_amplitudes()
            print(f"  Step {step:2d}: A={amps[A]:.3f}, B={amps[B]:.3f}, C={amps[C]:.3f}")

    amps = lattice.get_amplitudes()
    success = amps[B] > 0.05 and amps[C] > 0.01

    print(f"\n{'SUCCESS' if success else 'FAILED'}: Chain propagation")
    return success


def test_directionality():
    """Test that learning is directional."""
    print("\n" + "=" * 60)
    print("DIRECTIONALITY TEST")
    print("=" * 60)

    config = LatticeConfig(size=5, learning_rate=0.2, trace_lambda=0.7)
    lattice = PredictiveLattice(config)

    A, B = 0, 1

    # Train A -> B only
    print("\nTraining A -> B...")
    for _ in range(100):
        lattice.reset_states(clear_trace=True)
        for _ in range(5):
            u = np.zeros(5, dtype=np.complex128)
            u[A] = 1.0
            lattice.step(u)
        for _ in range(5):
            u = np.zeros(5, dtype=np.complex128)
            u[B] = 1.0
            lattice.step(u)

    ab = lattice.get_connection_strength(A, B)
    ba = lattice.get_connection_strength(B, A)

    print(f"A->B strength: {ab:.4f}")
    print(f"B->A strength: {ba:.4f}")
    print(f"Ratio: {ab / (ba + 1e-6):.2f}x")

    success = ab > 2 * ba
    print(f"\n{'SUCCESS' if success else 'FAILED'}: Learning is directional")
    return success


if __name__ == "__main__":
    print("PREDICTIVE LATTICE - Fast Reflexes Module\n")

    test_chain_resonance()
    test_directionality()

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
