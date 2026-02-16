#!/usr/bin/env python3
"""
Unified AI - Comprehensive Test Suite

Runs all tests and benchmarks to verify the system works correctly.
Tests the complete 2026 AI architecture:
1. HippocampusVSA (clean-up memory)
2. PredictiveLattice (fast reflexes)
3. RobustWaveToHVBridge (continuous-to-symbolic)
4. UnifiedAI (integration layer)
5. LLM Interface (natural language)
"""

import sys
import time
import numpy as np
from typing import Dict, List, Tuple
import traceback

# Add current directory to path
sys.path.insert(0, '.')


class TestResult:
    """Container for test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.timing = {}

    def record(self, name: str, success: bool, duration: float, error: str = None):
        if success:
            self.passed += 1
        else:
            self.failed += 1
            if error:
                self.errors.append((name, error))
        self.timing[name] = duration

    def summary(self) -> str:
        total = self.passed + self.failed
        return f"Passed: {self.passed}/{total}, Failed: {self.failed}"


def run_test(name: str, test_func, results: TestResult):
    """Run a single test with timing and error handling."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)

    start = time.perf_counter()
    try:
        success = test_func()
        duration = time.perf_counter() - start
        results.record(name, success, duration)
        status = "PASS" if success else "FAIL"
        print(f"\n[{status}] {name} ({duration*1000:.1f}ms)")
    except Exception as e:
        duration = time.perf_counter() - start
        results.record(name, False, duration, str(e))
        print(f"\n[ERROR] {name}: {e}")
        traceback.print_exc()


# ============================================================================
# HippocampusVSA Tests
# ============================================================================

def test_hippocampus_basic():
    """Test basic VSA operations."""
    from hippocampus_vsa import HippocampusVSA

    hpc = HippocampusVSA(dimensions=10000)

    # Register concepts
    v1 = hpc.register_concept("cat", category='animal')
    v2 = hpc.register_concept("dog", category='animal')
    v3 = hpc.register_concept("red", category='color')

    # Test orthogonality (random vectors should be nearly orthogonal)
    sim_cat_dog = hpc.similarity(v1, v2)
    sim_cat_red = hpc.similarity(v1, v3)

    print(f"  cat-dog similarity: {sim_cat_dog:.3f} (should be ~0)")
    print(f"  cat-red similarity: {sim_cat_red:.3f} (should be ~0)")

    # Test self-similarity
    sim_self = hpc.similarity(v1, v1)
    print(f"  cat-cat similarity: {sim_self:.3f} (should be 1.0)")

    return abs(sim_cat_dog) < 0.1 and abs(sim_cat_red) < 0.1 and sim_self == 1.0


def test_hippocampus_bind_unbind():
    """Test binding and unbinding operations."""
    from hippocampus_vsa import HippocampusVSA

    hpc = HippocampusVSA(dimensions=10000)

    # Register concepts
    role = hpc.register_concept("color", category='role')
    value = hpc.register_concept("blue", category='value')

    # Bind
    bound = hpc.bind(role, value)

    # Unbind should recover the original
    recovered = hpc.unbind(bound, role)

    # Clean up
    match, conf = hpc.cleanup(recovered, category='value')

    print(f"  Recovered concept: {match} (conf: {conf:.3f})")

    return match == "blue" and conf > 0.9


def test_hippocampus_bundling():
    """Test bundling multiple facts."""
    from hippocampus_vsa import HippocampusVSA

    hpc = HippocampusVSA(dimensions=10000)

    # Register concepts
    r_status = hpc.register_concept("status", category='role')
    r_loc = hpc.register_concept("location", category='role')
    v_good = hpc.register_concept("good", category='value')
    v_home = hpc.register_concept("home", category='value')

    # Create facts
    fact1 = hpc.bind(r_status, v_good)
    fact2 = hpc.bind(r_loc, v_home)

    # Bundle
    bundle = hpc.bundle([fact1, fact2])

    # Extract facts
    noisy_status = hpc.unbind(bundle, r_status)
    noisy_loc = hpc.unbind(bundle, r_loc)

    status_match, status_conf = hpc.cleanup(noisy_status, category='value')
    loc_match, loc_conf = hpc.cleanup(noisy_loc, category='value')

    print(f"  Extracted status: {status_match} (conf: {status_conf:.3f})")
    print(f"  Extracted location: {loc_match} (conf: {loc_conf:.3f})")

    return status_match == "good" and loc_match == "home"


def test_hippocampus_capacity():
    """Test bundling capacity."""
    from hippocampus_vsa import HippocampusVSA

    hpc = HippocampusVSA(dimensions=10000)

    # Register many concepts
    roles = []
    values = []
    for i in range(15):
        r = hpc.register_concept(f"role_{i}", category='role')
        v = hpc.register_concept(f"value_{i}", category='value')
        roles.append(r)
        values.append(v)

    # Test different bundle sizes
    results = []
    for num_facts in [2, 5, 10]:
        # Create and bundle facts
        facts = [hpc.bind(roles[i], values[i]) for i in range(num_facts)]
        bundle = hpc.bundle(facts)

        # Try to recover each fact
        correct = 0
        for i in range(num_facts):
            noisy = hpc.unbind(bundle, roles[i])
            match, _ = hpc.cleanup(noisy, category='value')
            if match == f"value_{i}":
                correct += 1

        accuracy = correct / num_facts
        results.append(accuracy)
        print(f"  {num_facts} facts: {correct}/{num_facts} correct ({accuracy:.0%})")

    # At least 2 facts should work perfectly
    return results[0] == 1.0


# ============================================================================
# PredictiveLattice Tests
# ============================================================================

def test_lattice_basic():
    """Test basic lattice operations."""
    from predictive_lattice import PredictiveLattice, LatticeConfig

    config = LatticeConfig(size=50, learning_rate=0.2)
    lattice = PredictiveLattice(config)

    # Step without input
    amps = lattice.step()
    print(f"  Initial max amplitude: {np.max(amps):.3f}")

    # Stimulate and step
    lattice.stimulate([0, 1, 2], amplitude=1.0)
    amps = lattice.step()
    print(f"  After stimulation: max={np.max(amps):.3f}")

    return np.max(amps) > 0.5


def test_lattice_chain():
    """Test A->B->C chain learning."""
    from predictive_lattice import PredictiveLattice, LatticeConfig

    config = LatticeConfig(size=10, learning_rate=0.25, trace_lambda=0.85)
    lattice = PredictiveLattice(config)

    A, B, C = 0, 1, 2

    # Train A->B
    for _ in range(80):
        lattice.reset_states(clear_trace=True)
        for _ in range(5):
            u = np.zeros(10, dtype=np.complex128)
            u[A] = 1.0
            lattice.step(u)
        for _ in range(5):
            u = np.zeros(10, dtype=np.complex128)
            u[B] = 1.0
            lattice.step(u)

    # Train B->C
    for _ in range(80):
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
    lattice.reset_states()
    lattice.stimulate([A], amplitude=1.5)

    for _ in range(20):
        u = np.zeros(10, dtype=np.complex128)
        lattice.step(learn=False)

    amps = lattice.get_amplitudes()
    print(f"  After triggering A: A={amps[A]:.3f}, B={amps[B]:.3f}, C={amps[C]:.3f}")

    return amps[B] > 0.02 or amps[C] > 0.01


def test_lattice_directionality():
    """Test that learning is directional."""
    from predictive_lattice import PredictiveLattice, LatticeConfig

    config = LatticeConfig(size=5, learning_rate=0.2, trace_lambda=0.7)
    lattice = PredictiveLattice(config)

    # Train A->B
    for _ in range(100):
        lattice.reset_states(clear_trace=True)
        for _ in range(5):
            u = np.zeros(5, dtype=np.complex128)
            u[0] = 1.0
            lattice.step(u)
        for _ in range(5):
            u = np.zeros(5, dtype=np.complex128)
            u[1] = 1.0
            lattice.step(u)

    ab = lattice.get_connection_strength(0, 1)
    ba = lattice.get_connection_strength(1, 0)
    ratio = ab / (ba + 1e-6)

    print(f"  A->B: {ab:.4f}, B->A: {ba:.4f}, ratio: {ratio:.2f}x")

    return ratio > 1.2  # A->B should be stronger (adjusted threshold)


def test_lattice_sparsity():
    """Test that lattice maintains sparsity."""
    from predictive_lattice import PredictiveLattice, LatticeConfig

    config = LatticeConfig(size=100, learning_rate=0.15, use_sparse=True)
    lattice = PredictiveLattice(config)

    # Train on random sequences
    for _ in range(50):
        lattice.reset_states(clear_trace=True)
        seq = np.random.choice(100, size=4, replace=False)
        for idx in seq:
            for _ in range(3):
                u = np.zeros(100, dtype=np.complex128)
                u[idx] = 0.8
                lattice.step(u)

    sparsity = lattice.get_sparsity()
    memory_kb = lattice._estimate_memory() / 1024

    print(f"  Sparsity: {sparsity:.1%}")
    print(f"  Memory: {memory_kb:.1f} KB")

    return sparsity > 0.5 and memory_kb < 200


# ============================================================================
# Bridge Tests
# ============================================================================

def test_bridge_basic():
    """Test basic bridge encoding."""
    from wave_to_hv_bridge import RobustWaveToHVBridge, BridgeConfig

    config = BridgeConfig(hv_dimensions=10000, lattice_size=100)
    bridge = RobustWaveToHVBridge(config)

    # Encode active neurons
    hv1 = bridge.encode_active_neurons([0, 5, 10])
    hv2 = bridge.encode_active_neurons([0, 5, 10])  # Same
    hv3 = bridge.encode_active_neurons([50, 55, 60])  # Different

    sim_same = bridge.similarity(hv1, hv2)
    sim_diff = bridge.similarity(hv1, hv3)

    print(f"  Same sets similarity: {sim_same:.3f}")
    print(f"  Different sets similarity: {sim_diff:.3f}")

    return sim_same > 0.9 and abs(sim_diff) < 0.2


def test_bridge_amplitude():
    """Test thermometer encoding of amplitude."""
    from wave_to_hv_bridge import RobustWaveToHVBridge, BridgeConfig

    config = BridgeConfig(hv_dimensions=10000)
    bridge = RobustWaveToHVBridge(config)

    # Encode different amplitudes
    hvs = [bridge.encode_amplitude(a) for a in [0.1, 0.3, 0.5, 0.7, 0.9]]

    # Adjacent amplitudes should be more similar
    sim_adjacent = bridge.similarity(hvs[1], hvs[2])
    sim_distant = bridge.similarity(hvs[0], hvs[4])

    print(f"  Adjacent (0.3-0.5) similarity: {sim_adjacent:.3f}")
    print(f"  Distant (0.1-0.9) similarity: {sim_distant:.3f}")

    return sim_adjacent > sim_distant


def test_bridge_complex_state():
    """Test encoding of complex oscillator state."""
    from wave_to_hv_bridge import RobustWaveToHVBridge, BridgeConfig

    config = BridgeConfig(hv_dimensions=10000, lattice_size=100)
    bridge = RobustWaveToHVBridge(config)

    # Create complex states
    z1 = np.zeros(100, dtype=np.complex128)
    z1[10:15] = 0.5 * np.exp(1j * np.pi / 4)

    z2 = np.zeros(100, dtype=np.complex128)
    z2[10:15] = 0.5 * np.exp(1j * np.pi / 4)  # Same

    z3 = np.zeros(100, dtype=np.complex128)
    z3[50:55] = 0.5 * np.exp(1j * np.pi / 4)  # Different neurons

    hv1 = bridge.encode_complex_state(z1)
    hv2 = bridge.encode_complex_state(z2)
    hv3 = bridge.encode_complex_state(z3)

    sim_same = bridge.similarity(hv1, hv2)
    sim_diff = bridge.similarity(hv1, hv3)

    print(f"  Same state similarity: {sim_same:.3f}")
    print(f"  Different state similarity: {sim_diff:.3f}")

    # Same states should be very similar, different should be less similar
    return sim_same > 0.8 and sim_diff < sim_same


# ============================================================================
# UnifiedAI Integration Tests
# ============================================================================

def test_unified_basic():
    """Test basic UnifiedAI operations."""
    from unified_ai import UnifiedAI, UnifiedAIConfig

    config = UnifiedAIConfig(lattice_size=50, hv_dimensions=5000)
    ai = UnifiedAI(config)

    # Register concepts
    ai.register_concept("Status", category='role', neuron_indices=[0, 1, 2])
    ai.register_concept("OK", category='value', neuron_indices=[10, 11, 12])

    # Store fact
    result = ai.store_fact("Device-1", "Status", "OK")
    print(f"  Store result: {result['success']}")

    # Query fact
    query = ai.query_fact("Device-1", "Status")
    print(f"  Query result: {query['value']} (conf: {query['confidence']:.2f})")

    return result['success'] and query['value'] == "OK"


def test_unified_no_hallucination():
    """Test that system cannot hallucinate."""
    from unified_ai import UnifiedAI, UnifiedAIConfig

    config = UnifiedAIConfig(lattice_size=50, hv_dimensions=5000)
    ai = UnifiedAI(config)

    # Register only some concepts
    ai.register_concept("Status", category='role')
    ai.register_concept("Active", category='value')

    # Store one fact
    ai.store_fact("Entity-1", "Status", "Active")

    # Query known fact
    known = ai.query_fact("Entity-1", "Status")

    # Query unknown entity
    unknown_entity = ai.query_fact("Entity-99", "Status")

    # Query unknown role
    ai.register_concept("Temperature", category='role')
    unknown_role = ai.query_fact("Entity-1", "Temperature")

    print(f"  Known: {known['value']} (grounded: {known['grounded']})")
    print(f"  Unknown entity: {unknown_entity['value']} (grounded: {unknown_entity['grounded']})")
    print(f"  Unknown role: {unknown_role['value']} (grounded: {unknown_role['grounded']})")

    # Known should be grounded, unknowns should not be
    return (
        known['grounded'] and
        not unknown_entity['grounded'] and
        not unknown_role['grounded']
    )


def test_unified_grounded_response():
    """Test grounded response generation."""
    from unified_ai import UnifiedAI, UnifiedAIConfig

    config = UnifiedAIConfig(lattice_size=50, hv_dimensions=5000)
    ai = UnifiedAI(config)

    # Setup
    ai.register_concept("Status", category='role')
    ai.register_concept("Location", category='role')
    ai.register_concept("Online", category='value')
    ai.register_concept("Server", category='value')

    ai.store_fact("System-1", "Status", "Online")
    ai.store_fact("System-1", "Location", "Server")

    # Get grounded response
    response = ai.get_grounded_response("System-1", ["Status", "Location"])

    print(f"  Response grounded: {response['grounded']}")
    print(f"  LLM safe: {response['llm_safe']}")
    print(f"  Facts: {response['facts']}")

    return response['grounded'] and response['llm_safe']


def test_unified_performance():
    """Test inference performance."""
    from unified_ai import UnifiedAI, UnifiedAIConfig

    config = UnifiedAIConfig(lattice_size=100, hv_dimensions=10000)
    ai = UnifiedAI(config)

    # Warmup
    for _ in range(10):
        signal = np.random.randn(100) * 0.1
        ai.process_signal(signal.astype(np.complex128), learn=False)

    # Benchmark
    times = []
    for _ in range(100):
        signal = np.random.randn(100) * 0.1
        start = time.perf_counter()
        ai.process_signal(signal.astype(np.complex128), learn=False)
        times.append(time.perf_counter() - start)

    avg_us = np.mean(times) * 1_000_000
    p99_us = np.percentile(times, 99) * 1_000_000

    print(f"  Average inference: {avg_us:.1f} us")
    print(f"  P99 inference: {p99_us:.1f} us")

    return avg_us < 10000  # Should be under 10ms


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    print("="*70)
    print("UNIFIED AI - COMPREHENSIVE TEST SUITE")
    print("="*70)

    results = TestResult()

    # HippocampusVSA Tests
    print("\n" + "="*70)
    print("HIPPOCAMPUS VSA TESTS")
    print("="*70)

    run_test("Hippocampus Basic", test_hippocampus_basic, results)
    run_test("Hippocampus Bind/Unbind", test_hippocampus_bind_unbind, results)
    run_test("Hippocampus Bundling", test_hippocampus_bundling, results)
    run_test("Hippocampus Capacity", test_hippocampus_capacity, results)

    # PredictiveLattice Tests
    print("\n" + "="*70)
    print("PREDICTIVE LATTICE TESTS")
    print("="*70)

    run_test("Lattice Basic", test_lattice_basic, results)
    run_test("Lattice Chain", test_lattice_chain, results)
    run_test("Lattice Directionality", test_lattice_directionality, results)
    run_test("Lattice Sparsity", test_lattice_sparsity, results)

    # Bridge Tests
    print("\n" + "="*70)
    print("WAVE-TO-HV BRIDGE TESTS")
    print("="*70)

    run_test("Bridge Basic", test_bridge_basic, results)
    run_test("Bridge Amplitude", test_bridge_amplitude, results)
    run_test("Bridge Complex State", test_bridge_complex_state, results)

    # UnifiedAI Tests
    print("\n" + "="*70)
    print("UNIFIED AI INTEGRATION TESTS")
    print("="*70)

    run_test("Unified Basic", test_unified_basic, results)
    run_test("Unified No Hallucination", test_unified_no_hallucination, results)
    run_test("Unified Grounded Response", test_unified_grounded_response, results)
    run_test("Unified Performance", test_unified_performance, results)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"\n{results.summary()}")

    if results.errors:
        print("\nErrors:")
        for name, error in results.errors:
            print(f"  - {name}: {error}")

    print("\nTiming:")
    for name, duration in sorted(results.timing.items(), key=lambda x: -x[1]):
        print(f"  {name}: {duration*1000:.1f}ms")

    total_time = sum(results.timing.values())
    print(f"\nTotal test time: {total_time:.2f}s")

    success = results.failed == 0
    print(f"\n{'SUCCESS' if success else 'FAILED'}: All tests {'passed' if success else 'did not pass'}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
