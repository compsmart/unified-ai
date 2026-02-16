"""
PredictiveLattice V2 - Improved Fast Reflexes Module

Addresses saturation/normalization issues identified in interference diagnosis:
1. Per-edge decay + cap (instead of row normalization)
2. Hard sparsity in updates (top-k gating)
3. Bounded out-degree
4. Phase reset on stimulation

Run test_mitigation_comparison.py to compare V1 vs V2 performance.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix


@dataclass
class LatticeConfigV2:
    """Configuration for the improved predictive lattice."""
    size: int = 256
    learning_rate: float = 0.15
    decay: float = 0.02
    trace_lambda: float = 0.8
    threshold: float = 0.1
    dt: float = 0.1
    use_sparse: bool = True
    # V2 additions
    edge_decay: float = 0.001      # Per-edge decay rate
    max_edge_weight: float = 0.5   # Cap individual edges
    top_k_sources: int = 5         # Sparsity in updates
    top_k_targets: int = 5         # Sparsity in updates
    max_out_degree: int = 20       # Bounded out-degree
    phase_lock: bool = True        # Lock phase on stimulation


class PredictiveLatticeV2:
    """
    Improved fast-reflex lattice addressing saturation issues.

    Key changes from V1:
    - No row normalization (prevents dilution)
    - Per-edge decay + cap (bounded without interference)
    - Top-k gated updates (reduces cross-talk)
    - Phase reset on stimulation (reliable activation)
    """

    def __init__(self, config: LatticeConfigV2 = None):
        self.config = config or LatticeConfigV2()
        n = self.config.size

        # Complex oscillator states
        self.z = np.zeros(n, dtype=np.complex128)

        # Eligibility trace
        self.e = np.zeros(n, dtype=np.complex128)

        # All neurons at same frequency (no drift)
        self.omega = np.ones(n) * 1.0

        # Weight matrix - sparse
        if self.config.use_sparse:
            self.W = lil_matrix((n, n), dtype=np.complex128)
        else:
            self.W = np.zeros((n, n), dtype=np.complex128)

        # Statistics
        self.total_steps = 0
        self.active_history = []

    def step(self, u: np.ndarray = None, learn: bool = True) -> np.ndarray:
        """Advance the lattice by one time step."""
        n = self.config.size
        dt = self.config.dt

        if u is None:
            u = np.zeros(n, dtype=np.complex128)

        # 1. Natural Rotation (no drift since omega identical)
        self.z *= np.exp(1j * self.omega * dt)

        # 2. Coupling from other neurons + external input
        if self.config.use_sparse:
            coupling = np.array(self.W.dot(self.z)).flatten()
        else:
            coupling = np.dot(self.W, self.z)

        drive = coupling + u

        # 3. Update state with damping
        self.z = (self.z + drive * dt) * (1 - self.config.decay)

        # 4. Causal learning via eligibility trace (IMPROVED)
        if learn:
            self._causal_update_v2()

        # 5. Update eligibility trace
        lam = self.config.trace_lambda
        self.e = lam * self.e + (1 - lam) * self.z

        # 6. Per-edge normalization (NOT row normalization)
        self._normalize_v2()

        self.total_steps += 1
        return np.abs(self.z)

    def _causal_update_v2(self):
        """
        Apply improved causal learning rule with top-k gating.

        Only updates edges between top-k active sources and targets,
        reducing cross-talk massively.
        """
        lr = self.config.learning_rate
        threshold = self.config.threshold
        dt = self.config.dt
        top_k_s = self.config.top_k_sources
        top_k_t = self.config.top_k_targets

        # Find active neurons (above threshold)
        z_amp = np.abs(self.z)
        e_amp = np.abs(self.e)

        # Get top-k sources (by eligibility - recent past activity)
        source_candidates = np.where(e_amp > threshold * 0.5)[0]
        if len(source_candidates) > top_k_s:
            top_sources = source_candidates[np.argsort(e_amp[source_candidates])[-top_k_s:]]
        else:
            top_sources = source_candidates

        # Get top-k targets (by current activity)
        target_candidates = np.where(z_amp > threshold)[0]
        if len(target_candidates) > top_k_t:
            top_targets = target_candidates[np.argsort(z_amp[target_candidates])[-top_k_t:]]
        else:
            top_targets = target_candidates

        if len(top_sources) == 0 or len(top_targets) == 0:
            return

        # Update only top-k x top-k edges
        for t in top_targets:
            for s in top_sources:
                if t != s:
                    update = self.z[t] * np.conj(self.e[s]) * lr * dt
                    if self.config.use_sparse:
                        self.W[t, s] += update
                    else:
                        self.W[t, s] += update

        self.active_history.append(len(top_targets))

    def _normalize_v2(self):
        """
        Improved normalization: per-edge decay + cap.

        Does NOT use row normalization, which was causing dilution.
        """
        # 1. State normalization (prevent amplitude explosion)
        max_amp = 2.0
        state_norms = np.abs(self.z)
        overflow = state_norms > max_amp
        if np.any(overflow):
            self.z[overflow] *= max_amp / (state_norms[overflow] + 1e-8)

        # 2. Per-edge decay (gradual forgetting, not dilution)
        self.W *= (1 - self.config.edge_decay)

        # 3. Per-edge cap (prevent single edge explosion)
        max_w = self.config.max_edge_weight
        if self.config.use_sparse:
            W_csr = self.W.tocsr()
            overflow = np.abs(W_csr.data) > max_w
            if np.any(overflow):
                W_csr.data[overflow] *= max_w / (np.abs(W_csr.data[overflow]) + 1e-8)
            self.W = W_csr.tolil()
        else:
            overflow = np.abs(self.W) > max_w
            self.W[overflow] *= max_w / (np.abs(self.W[overflow]) + 1e-8)

        # 4. Bounded out-degree (prune weak edges)
        self._prune_edges()

    def _prune_edges(self):
        """Limit number of outgoing edges per source neuron."""
        max_out = self.config.max_out_degree

        if self.config.use_sparse:
            W_csr = self.W.tocsr()
            # Check each column (outgoing from source)
            for s in range(self.config.size):
                col = np.abs(W_csr[:, s].toarray().flatten())
                nonzero = np.count_nonzero(col > 1e-6)
                if nonzero > max_out:
                    threshold = np.sort(col)[-max_out]
                    weak = col < threshold
                    for t in np.where(weak)[0]:
                        W_csr[t, s] = 0
            self.W = W_csr.tolil()

    def stimulate(self, indices: List[int], amplitude: float = 1.0, phase: float = None):
        """
        Activate specific neurons with optional phase locking.

        V2 improvement: Phase reset for reliable activation.
        """
        if phase is None and self.config.phase_lock:
            phase = 0.0  # Lock to phase 0

        for idx in indices:
            if 0 <= idx < self.config.size:
                if phase is not None:
                    self.z[idx] = amplitude * np.exp(1j * phase)
                else:
                    # Keep existing phase, just boost amplitude
                    current_phase = np.angle(self.z[idx]) if np.abs(self.z[idx]) > 0 else 0
                    self.z[idx] = amplitude * np.exp(1j * current_phase)

    def reset_states(self, clear_trace: bool = True):
        """Reset states but keep learned weights."""
        self.z = np.zeros(self.config.size, dtype=np.complex128)
        if clear_trace:
            self.e = np.zeros(self.config.size, dtype=np.complex128)

    def clear_trace(self):
        """Clear eligibility trace (between sequences)."""
        self.e = np.zeros(self.config.size, dtype=np.complex128)

    def get_amplitudes(self) -> np.ndarray:
        return np.abs(self.z)

    def get_top_active(self, k: int = 10) -> List[Tuple[int, float]]:
        amps = np.abs(self.z)
        indices = np.argsort(amps)[::-1][:k]
        return [(int(i), float(amps[i])) for i in indices if amps[i] > self.config.threshold]

    def predict_next(self, current_indices: List[int], steps: int = 5) -> List[Tuple[int, float]]:
        """Predict which neurons will activate next."""
        saved_z = self.z.copy()
        saved_e = self.e.copy()

        self.reset_states()
        self.stimulate(current_indices, amplitude=1.5)

        for _ in range(steps):
            self.step(learn=False)

        predictions = self.get_top_active(k=20)
        predictions = [(i, a) for i, a in predictions if i not in current_indices]

        self.z = saved_z
        self.e = saved_e

        return predictions

    def get_edge_strength(self, source: int, target: int) -> float:
        """Get strength of connection from source to target."""
        return float(np.abs(self.W[target, source]))

    def get_sparsity(self) -> float:
        """Calculate weight matrix sparsity."""
        threshold = 1e-6
        if self.config.use_sparse:
            W_dense = self.W.toarray()
        else:
            W_dense = self.W
        total = W_dense.size
        nonzero = np.sum(np.abs(W_dense) > threshold)
        return 1.0 - (nonzero / total)

    def get_stats(self) -> Dict:
        return {
            'version': 'V2',
            'size': self.config.size,
            'total_steps': self.total_steps,
            'sparsity': self.get_sparsity(),
            'active_avg': np.mean(self.active_history) if self.active_history else 0,
        }


def test_v2_basic():
    """Basic functionality test for V2."""
    print("=" * 60)
    print("PREDICTIVE LATTICE V2 - BASIC TEST")
    print("=" * 60)

    config = LatticeConfigV2(size=32, learning_rate=0.2)
    lattice = PredictiveLatticeV2(config)

    A, B, C = 0, 1, 2

    # Train A → B → C
    print("\nTraining sequence A → B → C...")
    for _ in range(50):
        lattice.reset_states(clear_trace=True)
        for step, neuron in enumerate([A, B, C]):
            for _ in range(3):
                u = np.zeros(32, dtype=np.complex128)
                u[neuron] = 1.0
                lattice.step(u)

    # Test prediction
    preds = lattice.predict_next([A], steps=10)
    pred_indices = [i for i, _ in preds]

    print(f"From A, predicted: {pred_indices[:5]}")
    success = B in pred_indices
    print(f"B in predictions: {'SUCCESS' if success else 'FAILED'}")

    # Check edge strengths
    ab = lattice.get_edge_strength(A, B)
    bc = lattice.get_edge_strength(B, C)
    print(f"\nEdge A→B: {ab:.4f}")
    print(f"Edge B→C: {bc:.4f}")

    print(f"\nStats: {lattice.get_stats()}")

    return success


if __name__ == '__main__':
    test_v2_basic()
