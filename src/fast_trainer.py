#!/usr/bin/env python3
"""
Fast Trainer - Optimized Scale Testing

Faster version with:
- Reduced sequence training (main bottleneck)
- Vectorized operations
- Progressive scaling with early stopping
- Resource monitoring
"""

import numpy as np
import os
import sys
import json
import time
import gc
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Disable output buffering
sys.stdout.reconfigure(line_buffering=True)

from hippocampus_vsa import HippocampusVSA, FactMemory
from predictive_lattice import PredictiveLattice, LatticeConfig
from wave_to_hv_bridge import RobustWaveToHVBridge, BridgeConfig
from unified_ai import UnifiedAI, UnifiedAIConfig


def get_memory_mb():
    """Get current process memory in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except:
        return 0


@dataclass
class ScaleResult:
    """Results from a scale test."""
    scale: int
    concepts: int
    facts: int
    accuracy: float
    latency_us: float
    memory_mb: float
    train_time_s: float


def test_scale(
    scale: int,
    lattice_size: int = 256,
    facts_per_entity: int = 3,
    verbose: bool = True
) -> ScaleResult:
    """
    Quick scale test focused on hippocampus (main storage).

    Args:
        scale: Number of entities
        lattice_size: Oscillator count
        facts_per_entity: Facts per entity
        verbose: Print progress
    """
    start_time = time.time()
    start_mem = get_memory_mb()

    if verbose:
        print(f"\n{'='*50}")
        print(f"SCALE TEST: {scale} entities")
        print(f"{'='*50}")

    # Create system
    config = UnifiedAIConfig(
        lattice_size=lattice_size,
        hv_dimensions=10000,
        learning_rate=0.15,
        similarity_threshold=0.12
    )
    ai = UnifiedAI(config)

    # Register base roles
    roles = ['Status', 'Location', 'Type', 'Priority', 'Owner']
    for role in roles:
        ai.register_concept(role, category='role')

    # Generate and register values
    values = []
    for i in range(min(scale // 5, 200)):  # Cap values at 200
        val = f"Value_{i}"
        ai.register_concept(val, category='value')
        values.append(val)

    if not values:
        values = ['Active', 'Idle', 'Unknown']
        for v in values:
            ai.register_concept(v, category='value')

    if verbose:
        print(f"  Registered {len(roles)} roles, {len(values)} values")

    # Store facts
    facts_stored = []
    rng = np.random.default_rng(42)

    for i in range(scale):
        entity = f"Entity_{i:05d}"
        # Store multiple facts per entity
        for _ in range(min(facts_per_entity, len(roles))):
            role = rng.choice(roles)
            value = rng.choice(values)
            result = ai.store_fact(entity, role, value)
            if result['success']:
                facts_stored.append((entity, role, value))

    if verbose:
        print(f"  Stored {len(facts_stored)} facts")

    # Test accuracy on sample
    test_sample = facts_stored[:min(200, len(facts_stored))]
    correct = 0
    times = []

    for entity, role, expected in test_sample:
        start = time.perf_counter()
        result = ai.query_fact(entity, role)
        times.append(time.perf_counter() - start)

        if result['value'] == expected:
            correct += 1

    accuracy = correct / len(test_sample) if test_sample else 0
    avg_latency = np.mean(times) * 1_000_000 if times else 0

    # Memory
    end_mem = get_memory_mb()
    train_time = time.time() - start_time

    result = ScaleResult(
        scale=scale,
        concepts=len(roles) + len(values),
        facts=len(facts_stored),
        accuracy=accuracy,
        latency_us=avg_latency,
        memory_mb=end_mem,
        train_time_s=train_time
    )

    if verbose:
        print(f"\n  Results:")
        print(f"    Accuracy:  {accuracy*100:.1f}%")
        print(f"    Latency:   {avg_latency:.1f} us")
        print(f"    Memory:    {end_mem:.1f} MB (+{end_mem-start_mem:.1f})")
        print(f"    Time:      {train_time:.1f}s")

    # Cleanup
    del ai
    gc.collect()

    return result


def test_learning_retention(verbose: bool = True) -> Dict:
    """Test that learning persists over iterations."""
    if verbose:
        print(f"\n{'='*50}")
        print("LEARNING RETENTION TEST")
        print(f"{'='*50}")

    config = UnifiedAIConfig(
        lattice_size=256,
        hv_dimensions=10000
    )
    ai = UnifiedAI(config)

    # Register concepts
    ai.register_concept("Status", category='role')

    all_facts = []
    retention_history = []

    for iteration in range(5):
        # Add new concepts
        for i in range(5):
            val = f"Val_i{iteration}_{i}"
            ai.register_concept(val, category='value')

        # Add new facts
        for i in range(10):
            entity = f"E_i{iteration}_{i}"
            value = f"Val_i{iteration}_{i % 5}"
            ai.store_fact(entity, "Status", value)
            all_facts.append((entity, "Status", value))

        # Test ALL facts (retention)
        correct = 0
        for entity, role, expected in all_facts:
            result = ai.query_fact(entity, role)
            if result['value'] == expected:
                correct += 1

        retention = correct / len(all_facts)
        retention_history.append({
            'iteration': iteration + 1,
            'total_facts': len(all_facts),
            'retention': retention
        })

        if verbose:
            print(f"  Iteration {iteration+1}: {len(all_facts)} facts, {retention*100:.1f}% retained")

    return {
        'final_retention': retention_history[-1]['retention'],
        'history': retention_history
    }


def test_sequence_learning(verbose: bool = True) -> Dict:
    """Test sequence prediction capability."""
    if verbose:
        print(f"\n{'='*50}")
        print("SEQUENCE LEARNING TEST")
        print(f"{'='*50}")

    config = UnifiedAIConfig(
        lattice_size=128,
        hv_dimensions=5000
    )
    ai = UnifiedAI(config)

    # Register sequence states
    states = ['Start', 'Process', 'Validate', 'Complete', 'End']
    for i, state in enumerate(states):
        ai.register_concept(state, category='state', neuron_indices=list(range(i*20, i*20+20)))

    # Train sequence (reduced reps for speed)
    ai.train_sequence(states, repetitions=20, steps_per_concept=3)

    # Test prediction
    prediction = ai.predict_next(current_concepts=['Start'], steps=10)

    if verbose:
        print(f"  Sequence: {' -> '.join(states)}")
        print(f"  Trained with 20 repetitions")
        print(f"  Prediction from 'Start': {prediction['predicted_concepts'][:3]}")

    expected_next = 'Process'
    success = any(c == expected_next for c, _ in prediction['predicted_concepts'][:3])

    return {
        'sequence': states,
        'predictions': prediction['predicted_concepts'][:5],
        'success': success
    }


def run_progressive_scale_test(
    max_scale: int = 10000,
    steps: List[int] = None,
    memory_limit_mb: float = 2000,
    verbose: bool = True
) -> List[ScaleResult]:
    """Run progressive scale tests with resource limits."""

    if steps is None:
        steps = [50, 100, 250, 500, 1000, 2000, 5000, 10000]

    steps = [s for s in steps if s <= max_scale]

    results = []
    start_mem = get_memory_mb()

    if verbose:
        print("\n" + "=" * 60)
        print("PROGRESSIVE SCALE TEST")
        print("=" * 60)
        print(f"  Scales: {steps}")
        print(f"  Memory limit: {memory_limit_mb} MB")
        print(f"  Starting memory: {start_mem:.1f} MB")

    for scale in steps:
        # Adjust lattice size based on scale
        if scale <= 100:
            lattice_size = 128
        elif scale <= 500:
            lattice_size = 256
        elif scale <= 2000:
            lattice_size = 512
        else:
            lattice_size = 1024

        result = test_scale(scale, lattice_size=lattice_size, verbose=verbose)
        results.append(result)

        # Check memory limit
        if result.memory_mb > memory_limit_mb:
            if verbose:
                print(f"\n  WARNING: Memory limit reached ({result.memory_mb:.1f} MB)")
            break

        gc.collect()

    return results


def save_benchmark_report(
    results: List[ScaleResult],
    retention: Dict,
    sequence: Dict,
    path: str
):
    """Save comprehensive benchmark report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'scale_tests': [asdict(r) for r in results],
        'retention_test': retention,
        'sequence_test': sequence,
        'summary': {
            'max_scale': max(r.scale for r in results) if results else 0,
            'max_facts': max(r.facts for r in results) if results else 0,
            'avg_accuracy': np.mean([r.accuracy for r in results]) if results else 0,
            'avg_latency_us': np.mean([r.latency_us for r in results]) if results else 0,
            'peak_memory_mb': max(r.memory_mb for r in results) if results else 0,
            'retention_rate': retention['final_retention'],
            'sequence_learning': sequence['success']
        }
    }

    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def print_summary(results: List[ScaleResult], retention: Dict, sequence: Dict):
    """Print formatted summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"\n{'Scale':<8} {'Facts':<8} {'Accuracy':<10} {'Latency':<12} {'Memory':<10} {'Time':<8}")
    print("-" * 70)

    for r in results:
        print(f"{r.scale:<8} {r.facts:<8} {r.accuracy*100:>6.1f}%    "
              f"{r.latency_us:>8.1f}us  {r.memory_mb:>6.1f}MB   {r.train_time_s:>5.1f}s")

    print("\nKey Metrics:")
    if results:
        print(f"  Max scale achieved: {max(r.scale for r in results)} entities")
        print(f"  Max facts stored:   {max(r.facts for r in results)}")
        print(f"  Average accuracy:   {np.mean([r.accuracy for r in results])*100:.1f}%")
        print(f"  Average latency:    {np.mean([r.latency_us for r in results]):.1f} us")
        print(f"  Peak memory:        {max(r.memory_mb for r in results):.1f} MB")

    print(f"\nLearning Retention: {retention['final_retention']*100:.1f}%")
    print(f"Sequence Learning:  {'PASS' if sequence['success'] else 'FAIL'}")


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("UNIFIED AI - FAST SCALE BENCHMARK")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Initial memory: {get_memory_mb():.1f} MB")

    # Run scale tests
    results = run_progressive_scale_test(
        max_scale=10000,
        steps=[50, 100, 250, 500, 1000, 2000, 5000],
        memory_limit_mb=1500,
        verbose=True
    )

    # Run retention test
    retention = test_learning_retention(verbose=True)

    # Run sequence test
    sequence = test_sequence_learning(verbose=True)

    # Print summary
    print_summary(results, retention, sequence)

    # Save report
    report_path = '/home/compsmart/htdocs/cloud.compsmart.co.uk/poc/unified-ai/models/benchmark_report.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = save_benchmark_report(results, retention, sequence, report_path)

    print(f"\nReport saved: {report_path}")
    print(f"Finished: {datetime.now().isoformat()}")

    return results, retention, sequence


if __name__ == "__main__":
    main()
