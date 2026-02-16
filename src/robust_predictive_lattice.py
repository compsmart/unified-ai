"""
RobustPredictiveLattice - Production-Ready Fast Reflexes

Addresses saturation through:
1. Dual Gating (top-k sources AND targets with thresholds)
2. Bounded Out-Degree (structural sparsity)
3. Per-Edge Decay + Cap (no row normalization)
4. Phase Anchor on Stimulation (reliable activation)

This replaces global row normalization which was causing dilution.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RobustLatticeConfig:
    """Configuration for RobustPredictiveLattice."""
    n: int = 256                  # Number of neurons
    lr: float = 0.2               # Learning rate
    decay_w: float = 0.01         # Per-edge decay rate
    w_max: float = 1.0            # Hard cap on edge magnitude
    max_out: int = 4              # Bounded out-degree per source
    k_tgt: int = 5                # Max targets to update per step
    k_src: int = 5                # Max sources to update per step
    tau_z: float = 0.1            # Activation threshold for targets
    tau_e: float = 0.1            # Trace threshold for sources
    trace_lambda: float = 0.8     # Eligibility trace decay


class RobustPredictiveLattice:
    """
    Production-ready lattice with saturation mitigations.

    Key improvements over V1/V2:
    - Sparse edge storage (dict-based, not matrix)
    - O(k^2) updates instead of O(n^2)
    - No row normalization = no dilution
    - Phase anchor on stimulation = reliable triggering
    """

    def __init__(self, config: RobustLatticeConfig = None):
        self.config = config or RobustLatticeConfig()
        n = self.config.n

        # Complex oscillator states
        self.z = np.zeros(n, dtype=np.complex64)

        # Eligibility trace (recent past activity)
        self.e = np.zeros(n, dtype=np.complex64)

        # Natural frequencies (identical to prevent drift)
        self.omega = np.ones(n, dtype=np.float32) * 1.0

        # Sparse edge storage: List of dicts {target_idx: complex_weight}
        # W[src][tgt] = weight from src to tgt
        self.W: List[Dict[int, complex]] = [{} for _ in range(n)]

        # Statistics
        self.total_steps = 0
        self.update_count = 0

    def stimulate(self, indices: List[int], amplitude: float = 1.0):
        """
        Phase reset on stimulation: provides stable phase anchor (phi = 0).

        This is critical for reliable activation - all stimulated neurons
        start with the same phase, ensuring coherent propagation.
        """
        for idx in indices:
            if 0 <= idx < self.config.n:
                self.z[idx] = np.complex64(amplitude * np.exp(1j * 0.0))

    def step(self, dt: float = 0.1, learn: bool = True) -> np.ndarray:
        """
        Advance lattice by one time step.

        Returns amplitudes of all oscillators.
        """
        n = self.config.n

        # 1. Physical Rotation (all same frequency = no drift)
        self.z *= np.exp(1j * self.omega * dt).astype(np.complex64)

        # 2. Sparse Drive Computation
        drive = np.zeros(n, dtype=np.complex64)
        for src in range(n):
            if np.abs(self.z[src]) < 1e-3:
                continue
            for tgt, weight in self.W[src].items():
                drive[tgt] += weight * self.z[src]

        # 3. Integrate State with damping
        self.z = (self.z + drive * dt) * 0.98  # Light damping

        # 4. Learning with Dual Gating
        if learn:
            self._dual_gated_update(dt)

        # 5. Per-Edge Maintenance (decay + cap)
        self._maintain_edges()

        # 6. Trace Update
        lam = self.config.trace_lambda
        self.e = (lam * self.e + (1 - lam) * self.z).astype(np.complex64)

        self.total_steps += 1
        return np.abs(self.z)

    def _dual_gated_update(self, dt: float):
        """
        Apply learning with dual gating: top-k sources AND top-k targets.

        This is O(k^2) instead of O(n^2), drastically reducing cross-talk.
        """
        k_tgt = self.config.k_tgt
        k_src = self.config.k_src
        tau_z = self.config.tau_z
        tau_e = self.config.tau_e
        lr = self.config.lr
        max_out = self.config.max_out

        amps_z = np.abs(self.z)
        amps_e = np.abs(self.e)

        # Find top-k targets (currently active)
        tgt_candidates = np.argsort(amps_z)[-k_tgt:]
        valid_tgt = [i for i in tgt_candidates if amps_z[i] > tau_z]

        # Find top-k sources (recently active via trace)
        src_candidates = np.argsort(amps_e)[-k_src:]
        valid_src = [i for i in src_candidates if amps_e[i] > tau_e]

        if not valid_tgt or not valid_src:
            return

        # O(k^2) edge updates
        for src in valid_src:
            conj_e = np.conj(self.e[src])

            for tgt in valid_tgt:
                if tgt == src:
                    continue  # No self-connections

                # Hebbian update: strengthen src→tgt if both active
                dw = lr * self.z[tgt] * conj_e * dt
                current = self.W[src].get(tgt, 0j)
                self.W[src][tgt] = np.complex64(current + dw)

            # Bounded out-degree pruning
            if len(self.W[src]) > max_out:
                sorted_edges = sorted(
                    self.W[src].items(),
                    key=lambda kv: np.abs(kv[1]),
                    reverse=True
                )
                self.W[src] = dict(sorted_edges[:max_out])

        self.update_count += 1

    def _maintain_edges(self):
        """
        Per-edge decay and cap (replaces global row normalization).

        This prevents saturation without diluting strong edges.
        """
        decay = self.config.decay_w
        w_max = self.config.w_max

        for src in range(self.config.n):
            for tgt in list(self.W[src].keys()):
                # Apply decay
                w_val = self.W[src][tgt] * (1.0 - decay)

                # Apply cap
                mag = np.abs(w_val)
                if mag > w_max:
                    w_val *= (w_max / mag)
                elif mag < 1e-6:
                    # Remove negligible edges
                    del self.W[src][tgt]
                    continue

                self.W[src][tgt] = np.complex64(w_val)

    def reset_states(self, clear_trace: bool = True):
        """Reset oscillator states but keep learned edges."""
        self.z = np.zeros(self.config.n, dtype=np.complex64)
        if clear_trace:
            self.e = np.zeros(self.config.n, dtype=np.complex64)

    def get_amplitudes(self) -> np.ndarray:
        """Get current oscillator amplitudes."""
        return np.abs(self.z)

    def get_edge_strength(self, source: int, target: int) -> float:
        """Get strength of connection from source to target."""
        return float(np.abs(self.W[source].get(target, 0j)))

    def get_total_edges(self) -> int:
        """Count total number of edges."""
        return sum(len(edges) for edges in self.W)

    def get_stats(self) -> Dict:
        """Get lattice statistics."""
        return {
            'version': 'Robust',
            'neurons': self.config.n,
            'total_edges': self.get_total_edges(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'max_out_degree': self.config.max_out,
            'k_tgt': self.config.k_tgt,
            'k_src': self.config.k_src,
        }

    def predict_next(
        self,
        current_indices: List[int],
        steps: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Predict which neurons will activate next.

        Returns list of (neuron_index, amplitude) tuples.
        """
        # Save state
        saved_z = self.z.copy()
        saved_e = self.e.copy()

        # Reset and stimulate with phase anchor
        self.reset_states()
        self.stimulate(current_indices, amplitude=1.0)

        # Propagate
        for _ in range(steps):
            self.step(learn=False)

        # Get predictions
        amps = np.abs(self.z)
        threshold = self.config.tau_z

        # Sort by amplitude, exclude input neurons
        predictions = []
        for idx in np.argsort(amps)[::-1]:
            if idx in current_indices:
                continue
            if amps[idx] > threshold:
                predictions.append((int(idx), float(amps[idx])))

        # Restore state
        self.z = saved_z
        self.e = saved_e

        return predictions[:20]  # Top 20


def test_robust_lattice():
    """Basic test for RobustPredictiveLattice."""
    print("=" * 60)
    print("ROBUST PREDICTIVE LATTICE TEST")
    print("=" * 60)

    config = RobustLatticeConfig(n=64, lr=0.25, max_out=4, k_tgt=3, k_src=3)
    lattice = RobustPredictiveLattice(config)

    # Train A → B → C (neurons 0, 10, 20)
    A, B, C = [0, 1, 2], [10, 11, 12], [20, 21, 22]

    print("\nTraining sequence A → B → C...")
    for rep in range(30):
        lattice.reset_states(clear_trace=True)

        # Present A
        lattice.stimulate(A, amplitude=1.0)
        for _ in range(5):
            lattice.step()

        # Present B
        lattice.stimulate(B, amplitude=1.0)
        for _ in range(5):
            lattice.step()

        # Present C
        lattice.stimulate(C, amplitude=1.0)
        for _ in range(5):
            lattice.step()

    print(f"Stats: {lattice.get_stats()}")

    # Test prediction
    print("\nTesting prediction from A...")
    lattice.reset_states()
    preds = lattice.predict_next(A, steps=8)

    pred_indices = set(i for i, _ in preds[:5])
    B_set = set(B)

    overlap = len(pred_indices & B_set)
    print(f"Predicted indices: {list(pred_indices)}")
    print(f"Expected B indices: {B}")
    print(f"Overlap: {overlap}/{len(B)}")

    success = overlap >= 2
    print(f"\n{'SUCCESS' if success else 'PARTIAL'}: Robust lattice prediction")

    return success


if __name__ == '__main__':
    test_robust_lattice()
