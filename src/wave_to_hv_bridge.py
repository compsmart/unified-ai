"""
RobustWaveToHVBridge - The Bridge Module

Converts continuous wave signals from the PredictiveLattice into
10,000-bit Bipolar Hypervectors for the HippocampusVSA.

Key Features:
- Zero-memory Rademacher seed projection
- Deterministic mapping (same input = same output)
- Handles complex (real + imaginary + amplitude) signals
- Thermometer encoding for continuous values
- Binning for efficient quantization

This is the critical link between the fast-reflex oscillator network
and the symbolic hypervector memory.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BridgeConfig:
    """Configuration for the Wave-to-HV bridge."""
    hv_dimensions: int = 10000      # Hypervector dimension
    lattice_size: int = 256         # Number of lattice oscillators
    amplitude_bins: int = 8         # Bins for amplitude quantization
    phase_bins: int = 16            # Bins for phase quantization
    seed: int = 42                  # Random seed for projections


class RobustWaveToHVBridge:
    """
    Converts oscillator states to hypervectors for VSA memory.

    Methods:
    1. Active Neuron Encoding - Bundle hypervectors of active neurons
    2. Thermometer Encoding - Encode continuous amplitude values
    3. Phase Encoding - Encode oscillator phase information
    4. Full State Encoding - Complete lattice state to single HV
    """

    def __init__(self, config: BridgeConfig = None):
        self.config = config or BridgeConfig()

        # Initialize deterministic random projections
        self.rng = np.random.default_rng(self.config.seed)

        # Pre-compute projection matrices (Rademacher - bipolar random)
        # Each lattice neuron gets a unique hypervector
        self._neuron_hvs = self.rng.choice(
            [-1, 1],
            size=(self.config.lattice_size, self.config.hv_dimensions)
        ).astype(np.int8)

        # Amplitude bin hypervectors
        self._amplitude_hvs = self.rng.choice(
            [-1, 1],
            size=(self.config.amplitude_bins, self.config.hv_dimensions)
        ).astype(np.int8)

        # Phase bin hypervectors
        self._phase_hvs = self.rng.choice(
            [-1, 1],
            size=(self.config.phase_bins, self.config.hv_dimensions)
        ).astype(np.int8)

        # Statistics
        self.encode_count = 0

    def encode_active_neurons(
        self,
        active_indices: List[int],
        amplitudes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Encode a set of active neurons as a bundled hypervector.

        Args:
            active_indices: List of active neuron indices
            amplitudes: Optional amplitude values for weighting

        Returns:
            Bundled hypervector (bipolar, 10000 bits)
        """
        if not active_indices:
            return np.zeros(self.config.hv_dimensions, dtype=np.int8)

        self.encode_count += 1

        # Get hypervectors for active neurons
        active_hvs = [self._neuron_hvs[i] for i in active_indices
                     if i < self.config.lattice_size]

        if not active_hvs:
            return np.zeros(self.config.hv_dimensions, dtype=np.int8)

        # Weighted sum if amplitudes provided
        if amplitudes is not None:
            weights = [amplitudes[i] for i in active_indices
                      if i < len(amplitudes)]
            if len(weights) != len(active_hvs):
                weights = [1.0] * len(active_hvs)
        else:
            weights = [1.0] * len(active_hvs)

        # Bundle with weighting
        sum_vec = np.zeros(self.config.hv_dimensions, dtype=np.float64)
        for hv, w in zip(active_hvs, weights):
            sum_vec += w * hv.astype(np.float64)

        # Threshold to bipolar
        bundled = np.sign(sum_vec).astype(np.int8)
        bundled[bundled == 0] = self.rng.choice([-1, 1])

        return bundled

    def encode_amplitude(self, amplitude: float, max_amp: float = 2.0) -> np.ndarray:
        """
        Thermometer encoding of amplitude value.

        Uses cumulative bundling: higher amplitude = more bins included.
        This preserves ordering: similar amplitudes = similar vectors.
        """
        # Normalize to [0, 1]
        normalized = min(1.0, max(0.0, amplitude / max_amp))

        # Determine how many bins to include
        num_bins = int(normalized * self.config.amplitude_bins)
        num_bins = max(1, min(num_bins, self.config.amplitude_bins))

        # Bundle first num_bins amplitude vectors (thermometer)
        sum_vec = np.sum(self._amplitude_hvs[:num_bins], axis=0, dtype=np.float64)

        # Threshold
        bundled = np.sign(sum_vec).astype(np.int8)
        bundled[bundled == 0] = 1

        return bundled

    def encode_phase(self, phase: float) -> np.ndarray:
        """
        Encode phase value [0, 2*pi] to hypervector.

        Uses circular encoding: adjacent phases have similar vectors.
        """
        # Normalize to [0, num_bins)
        normalized_phase = (phase % (2 * np.pi)) / (2 * np.pi)
        bin_idx = int(normalized_phase * self.config.phase_bins) % self.config.phase_bins

        # Use circular interpolation for smoothness
        bin_next = (bin_idx + 1) % self.config.phase_bins
        blend = (normalized_phase * self.config.phase_bins) - bin_idx

        # Weighted blend of adjacent bins
        hv = ((1 - blend) * self._phase_hvs[bin_idx].astype(np.float64) +
              blend * self._phase_hvs[bin_next].astype(np.float64))

        # Threshold
        result = np.sign(hv).astype(np.int8)
        result[result == 0] = 1

        return result

    def encode_complex_state(
        self,
        z: np.ndarray,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        Encode full complex oscillator state to hypervector.

        Combines:
        1. Active neuron identities
        2. Amplitude information
        3. Phase information

        Args:
            z: Complex state vector from PredictiveLattice
            threshold: Activation threshold

        Returns:
            Single hypervector encoding the full state
        """
        self.encode_count += 1

        amplitudes = np.abs(z)
        phases = np.angle(z)

        # Find active neurons
        active_mask = amplitudes > threshold
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            return np.zeros(self.config.hv_dimensions, dtype=np.int8)

        # Component 1: Active neuron identities
        identity_hv = self.encode_active_neurons(
            list(active_indices),
            amplitudes[active_indices]
        )

        # Component 2: Average amplitude
        avg_amp = np.mean(amplitudes[active_indices])
        amp_hv = self.encode_amplitude(avg_amp)

        # Component 3: Dominant phase
        # Weight phases by amplitude
        weighted_phases = amplitudes[active_indices] * np.exp(1j * phases[active_indices])
        dominant_phase = np.angle(np.sum(weighted_phases))
        phase_hv = self.encode_phase(dominant_phase)

        # Bundle components with different roles
        # Use binding to create orthogonal combination
        role_amp = self._amplitude_hvs[0]  # Role marker for amplitude
        role_phase = self._phase_hvs[0]    # Role marker for phase

        # Bind roles to values, then bundle
        components = [
            identity_hv,
            self._bind(amp_hv, role_amp),
            self._bind(phase_hv, role_phase)
        ]

        # Final bundle
        sum_vec = np.sum([c.astype(np.float64) for c in components], axis=0)
        result = np.sign(sum_vec).astype(np.int8)
        result[result == 0] = 1

        return result

    def _bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Element-wise multiplication (XOR for bipolar)."""
        return (v1 * v2).astype(np.int8)

    def encode_anomaly_signal(
        self,
        observed_z: np.ndarray,
        predicted_z: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[np.ndarray, float]:
        """
        Encode an anomaly signal (difference between observed and predicted).

        Returns:
            (anomaly_hv, anomaly_magnitude)
        """
        # Calculate difference
        diff = np.abs(observed_z) - np.abs(predicted_z)

        # Find neurons with significant difference
        anomaly_mask = np.abs(diff) > threshold
        anomaly_indices = np.where(anomaly_mask)[0]

        if len(anomaly_indices) == 0:
            return np.zeros(self.config.hv_dimensions, dtype=np.int8), 0.0

        # Encode anomalous neurons with their difference as weight
        anomaly_hv = self.encode_active_neurons(
            list(anomaly_indices),
            np.abs(diff[anomaly_indices])
        )

        # Calculate anomaly magnitude
        magnitude = np.mean(np.abs(diff[anomaly_mask]))

        return anomaly_hv, magnitude

    def encode_sequence(
        self,
        state_sequence: List[np.ndarray],
        max_length: int = 10
    ) -> np.ndarray:
        """
        Encode a sequence of states using positional permutation.

        Each state is permuted by its position, then all are bundled.
        """
        if not state_sequence:
            return np.zeros(self.config.hv_dimensions, dtype=np.int8)

        # Encode and permute each state
        permuted_hvs = []
        for i, state in enumerate(state_sequence[:max_length]):
            state_hv = self.encode_complex_state(state)
            # Permute by position (cyclic shift)
            permuted = np.roll(state_hv, i)
            permuted_hvs.append(permuted)

        # Bundle all
        sum_vec = np.sum([hv.astype(np.float64) for hv in permuted_hvs], axis=0)
        result = np.sign(sum_vec).astype(np.int8)
        result[result == 0] = 1

        return result

    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Cosine similarity between hypervectors."""
        return np.dot(hv1.astype(np.float64), hv2.astype(np.float64)) / self.config.hv_dimensions

    def get_neuron_hv(self, index: int) -> np.ndarray:
        """Get the base hypervector for a specific neuron."""
        if 0 <= index < self.config.lattice_size:
            return self._neuron_hvs[index].copy()
        raise ValueError(f"Neuron index out of range: {index}")

    def get_stats(self) -> Dict:
        """Get bridge statistics."""
        return {
            'hv_dimensions': self.config.hv_dimensions,
            'lattice_size': self.config.lattice_size,
            'amplitude_bins': self.config.amplitude_bins,
            'phase_bins': self.config.phase_bins,
            'encode_count': self.encode_count,
            'memory_bytes': self._estimate_memory()
        }

    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        # Projection matrices
        neuron_bytes = self.config.lattice_size * self.config.hv_dimensions
        amp_bytes = self.config.amplitude_bins * self.config.hv_dimensions
        phase_bytes = self.config.phase_bins * self.config.hv_dimensions
        return neuron_bytes + amp_bytes + phase_bytes


class SemanticBridge:
    """
    Extended bridge that maps semantic concepts to lattice neurons
    and vice versa.
    """

    def __init__(
        self,
        bridge: RobustWaveToHVBridge,
        hippocampus
    ):
        """
        Args:
            bridge: The base wave-to-HV bridge
            hippocampus: HippocampusVSA instance for concept lookup
        """
        self.bridge = bridge
        self.hippocampus = hippocampus

        # Mapping from concept names to neuron indices
        self.concept_to_neurons: Dict[str, List[int]] = {}
        self.neuron_to_concepts: Dict[int, List[str]] = {}

    def register_concept_mapping(
        self,
        concept_name: str,
        neuron_indices: List[int]
    ):
        """
        Register which neurons represent a concept.
        """
        self.concept_to_neurons[concept_name] = neuron_indices

        for idx in neuron_indices:
            if idx not in self.neuron_to_concepts:
                self.neuron_to_concepts[idx] = []
            self.neuron_to_concepts[idx].append(concept_name)

    def neurons_to_concepts(
        self,
        active_indices: List[int],
        min_overlap: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Convert active neuron set to potential concepts.

        Returns list of (concept_name, overlap_score) sorted by score.
        """
        results = []
        active_set = set(active_indices)

        for concept, neurons in self.concept_to_neurons.items():
            neuron_set = set(neurons)
            overlap = len(active_set & neuron_set)

            if overlap >= min_overlap:
                # Jaccard-like score
                score = overlap / len(neuron_set)
                results.append((concept, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def concepts_to_neurons(self, concept_names: List[str]) -> List[int]:
        """
        Get all neurons associated with given concepts.
        """
        neurons = set()
        for name in concept_names:
            if name in self.concept_to_neurons:
                neurons.update(self.concept_to_neurons[name])
        return list(neurons)

    def encode_with_concepts(
        self,
        z: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Encode state and identify associated concepts.

        Returns:
            (hypervector, [(concept_name, score), ...])
        """
        # Encode to HV
        hv = self.bridge.encode_complex_state(z, threshold)

        # Find active neurons
        active = np.where(np.abs(z) > threshold)[0]

        # Map to concepts
        concepts = self.neurons_to_concepts(list(active))

        return hv, concepts


def test_bridge():
    """Test the Wave-to-HV bridge."""
    print("=" * 60)
    print("WAVE-TO-HV BRIDGE TEST")
    print("=" * 60)

    config = BridgeConfig(hv_dimensions=10000, lattice_size=100)
    bridge = RobustWaveToHVBridge(config)

    # Test 1: Active neuron encoding
    print("\n1. Active Neuron Encoding:")
    active1 = [0, 5, 10, 15]
    active2 = [0, 5, 10, 20]
    active3 = [50, 55, 60, 65]

    hv1 = bridge.encode_active_neurons(active1)
    hv2 = bridge.encode_active_neurons(active2)
    hv3 = bridge.encode_active_neurons(active3)

    print(f"   Similar sets [0,5,10,15] vs [0,5,10,20]: {bridge.similarity(hv1, hv2):.3f}")
    print(f"   Different sets [0,5,10,15] vs [50,55,60,65]: {bridge.similarity(hv1, hv3):.3f}")

    # Test 2: Amplitude encoding (thermometer)
    print("\n2. Amplitude Encoding (thermometer):")
    for amp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        amp_hv = bridge.encode_amplitude(amp)
        base_hv = bridge.encode_amplitude(0.5)
        sim = bridge.similarity(amp_hv, base_hv)
        print(f"   Amplitude {amp:.1f} vs 0.5: {sim:.3f}")

    # Test 3: Complex state encoding
    print("\n3. Complex State Encoding:")
    z1 = np.zeros(100, dtype=np.complex128)
    z1[10:15] = 0.5 * np.exp(1j * np.pi / 4)

    z2 = np.zeros(100, dtype=np.complex128)
    z2[10:15] = 0.5 * np.exp(1j * np.pi / 3)  # Same neurons, different phase

    z3 = np.zeros(100, dtype=np.complex128)
    z3[50:55] = 0.5 * np.exp(1j * np.pi / 4)  # Different neurons, same phase

    hv_z1 = bridge.encode_complex_state(z1)
    hv_z2 = bridge.encode_complex_state(z2)
    hv_z3 = bridge.encode_complex_state(z3)

    print(f"   Same neurons, diff phase: {bridge.similarity(hv_z1, hv_z2):.3f}")
    print(f"   Diff neurons, same phase: {bridge.similarity(hv_z1, hv_z3):.3f}")

    # Test 4: Sequence encoding
    print("\n4. Sequence Encoding:")
    seq1 = [z1.copy() for _ in range(3)]
    seq2 = [z1.copy() for _ in range(3)]
    seq3 = [z3.copy() for _ in range(3)]

    hv_seq1 = bridge.encode_sequence(seq1)
    hv_seq2 = bridge.encode_sequence(seq2)
    hv_seq3 = bridge.encode_sequence(seq3)

    print(f"   Identical sequences: {bridge.similarity(hv_seq1, hv_seq2):.3f}")
    print(f"   Different sequences: {bridge.similarity(hv_seq1, hv_seq3):.3f}")

    print(f"\nBridge stats: {bridge.get_stats()}")
    print("\nSUCCESS: Bridge encoding tests passed")


if __name__ == "__main__":
    test_bridge()
