#!/usr/bin/env python3
"""
Stress Test - Push to Maximum Scale

Tests:
1. Scale to 10K, 20K, 50K, 100K entities
2. Memory limits and degradation
3. Persist and reload
4. Query performance at scale
"""

import numpy as np
import os
import sys
import json
import time
import gc
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

from hippocampus_vsa import HippocampusVSA
from unified_ai import UnifiedAI, UnifiedAIConfig


def get_memory_mb():
    """Get current process memory in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except:
        return 0


def stress_test_hippocampus(max_concepts: int = 100000, verbose: bool = True):
    """
    Stress test just the HippocampusVSA (the core storage).

    This tests the fundamental limit of the VSA approach.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("HIPPOCAMPUS STRESS TEST")
        print("=" * 60)

    hpc = HippocampusVSA(dimensions=10000, seed=42)
    start_mem = get_memory_mb()

    results = []
    scales = [1000, 5000, 10000, 20000, 50000, 100000]

    for target in scales:
        if target > max_concepts:
            break

        # Register concepts
        print(f"\n  Registering {target} concepts...")
        start_time = time.time()

        for i in range(target):
            hpc.register_concept(f"Concept_{i:06d}", category='test')

        reg_time = time.time() - start_time
        mem = get_memory_mb()

        # Test retrieval speed
        test_names = [f"Concept_{i:06d}" for i in np.random.randint(0, target, size=100)]
        times = []

        for name in test_names:
            start = time.perf_counter()
            vec = hpc.get_concept(name)
            times.append(time.perf_counter() - start)

        avg_retrieve = np.mean(times) * 1_000_000

        # Test cleanup accuracy (find nearest concept)
        correct = 0
        cleanup_times = []

        for i in range(100):
            name = f"Concept_{np.random.randint(0, target):06d}"
            vec = hpc.get_concept(name)
            if vec is not None:
                # Add some noise
                noisy = vec.copy().astype(np.float64)
                noisy += np.random.normal(0, 0.1, len(noisy))
                noisy = np.sign(noisy).astype(np.int8)

                start = time.perf_counter()
                match, conf = hpc.cleanup(noisy)
                cleanup_times.append(time.perf_counter() - start)

                if match == name:
                    correct += 1

        accuracy = correct / 100
        avg_cleanup = np.mean(cleanup_times) * 1_000_000 if cleanup_times else 0

        results.append({
            'concepts': target,
            'reg_time_s': reg_time,
            'memory_mb': mem,
            'retrieve_us': avg_retrieve,
            'cleanup_us': avg_cleanup,
            'accuracy': accuracy
        })

        if verbose:
            print(f"    Registered in {reg_time:.1f}s")
            print(f"    Memory: {mem:.1f} MB ({(mem-start_mem)/target*1000:.2f} KB/concept)")
            print(f"    Retrieve: {avg_retrieve:.1f} us")
            print(f"    Cleanup: {avg_cleanup:.1f} us, Accuracy: {accuracy*100:.1f}%")

        # Memory check
        if mem > 3000:
            print(f"    WARNING: Memory limit approaching, stopping")
            break

        gc.collect()

    return results


def stress_test_unified(max_entities: int = 50000, verbose: bool = True):
    """
    Stress test the full UnifiedAI system.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("UNIFIED AI STRESS TEST")
        print("=" * 60)

    results = []
    scales = [1000, 5000, 10000, 20000, 50000]

    for target in scales:
        if target > max_entities:
            break

        if verbose:
            print(f"\n  Testing {target} entities...")

        start_time = time.time()
        start_mem = get_memory_mb()

        # Create system
        config = UnifiedAIConfig(
            lattice_size=512,
            hv_dimensions=10000,
            similarity_threshold=0.10
        )
        ai = UnifiedAI(config)

        # Register concepts
        roles = ['Status', 'Type', 'Priority']
        for role in roles:
            ai.register_concept(role, category='role')

        num_values = min(target // 10, 500)
        values = []
        for i in range(num_values):
            val = f"Val_{i:04d}"
            ai.register_concept(val, category='value')
            values.append(val)

        # Store facts
        rng = np.random.default_rng(42)
        facts = []

        for i in range(target):
            entity = f"Entity_{i:06d}"
            role = rng.choice(roles)
            value = rng.choice(values)
            result = ai.store_fact(entity, role, value)
            if result['success']:
                facts.append((entity, role, value))

        store_time = time.time() - start_time

        # Test retrieval accuracy
        test_facts = facts[:min(500, len(facts))]
        correct = 0
        query_times = []

        for entity, role, expected in test_facts:
            start = time.perf_counter()
            result = ai.query_fact(entity, role)
            query_times.append(time.perf_counter() - start)

            if result['value'] == expected:
                correct += 1

        accuracy = correct / len(test_facts) if test_facts else 0
        avg_query = np.mean(query_times) * 1_000_000 if query_times else 0
        p99_query = np.percentile(query_times, 99) * 1_000_000 if query_times else 0

        mem = get_memory_mb()
        total_time = time.time() - start_time

        results.append({
            'entities': target,
            'facts': len(facts),
            'values': num_values,
            'accuracy': accuracy,
            'avg_query_us': avg_query,
            'p99_query_us': p99_query,
            'memory_mb': mem,
            'store_time_s': store_time,
            'total_time_s': total_time
        })

        if verbose:
            print(f"    Stored {len(facts)} facts in {store_time:.1f}s")
            print(f"    Accuracy: {accuracy*100:.1f}%")
            print(f"    Query: avg={avg_query:.1f}us, p99={p99_query:.1f}us")
            print(f"    Memory: {mem:.1f} MB")

        # Save for persistence test
        if target == 10000:
            save_path = '/home/compsmart/htdocs/cloud.compsmart.co.uk/poc/unified-ai/models/stress_10k'
            ai.save(save_path)
            if verbose:
                print(f"    Saved checkpoint: {save_path}")

        del ai
        gc.collect()

        if mem > 2000:
            print(f"    Memory limit reached")
            break

    return results


def test_persistence(verbose: bool = True):
    """Test save/load and continued learning."""
    if verbose:
        print("\n" + "=" * 60)
        print("PERSISTENCE TEST")
        print("=" * 60)

    save_path = '/home/compsmart/htdocs/cloud.compsmart.co.uk/poc/unified-ai/models/stress_10k'

    if not os.path.exists(save_path):
        print("  No checkpoint found, skipping")
        return None

    # Load saved model
    print("  Loading checkpoint...")
    config = UnifiedAIConfig(lattice_size=512, hv_dimensions=10000)
    ai = UnifiedAI(config)
    ai.load(save_path)

    # Test some queries
    correct = 0
    for i in range(100):
        entity = f"Entity_{np.random.randint(0, 10000):06d}"
        result = ai.query_fact(entity, 'Status')
        if result['value'] is not None:
            correct += 1

    print(f"  Queries on loaded model: {correct}/100 returned values")

    # Add new facts
    print("  Adding 1000 new facts...")
    ai.register_concept("NewValue", category='value')

    for i in range(1000):
        entity = f"NewEntity_{i:04d}"
        ai.store_fact(entity, "Status", "NewValue")

    # Test new facts
    correct_new = 0
    for i in range(100):
        entity = f"NewEntity_{i:04d}"
        result = ai.query_fact(entity, "Status")
        if result['value'] == "NewValue":
            correct_new += 1

    print(f"  New fact accuracy: {correct_new}%")

    # Save again
    new_save_path = save_path + "_extended"
    ai.save(new_save_path)
    print(f"  Saved extended model: {new_save_path}")

    return {
        'loaded_queries': correct,
        'new_fact_accuracy': correct_new / 100
    }


def print_stress_summary(hpc_results, unified_results, persistence):
    """Print stress test summary."""
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)

    if hpc_results:
        print("\nHippocampus (Pure VSA Storage):")
        print(f"  {'Concepts':<12} {'Memory':<12} {'Retrieve':<12} {'Cleanup':<12} {'Accuracy':<10}")
        print("-" * 60)
        for r in hpc_results:
            print(f"  {r['concepts']:<12} {r['memory_mb']:>8.1f}MB  "
                  f"{r['retrieve_us']:>8.1f}us  {r['cleanup_us']:>8.1f}us  {r['accuracy']*100:>6.1f}%")

    if unified_results:
        print("\nUnified AI (Full System):")
        print(f"  {'Entities':<10} {'Facts':<10} {'Accuracy':<10} {'Query':<12} {'Memory':<10}")
        print("-" * 60)
        for r in unified_results:
            print(f"  {r['entities']:<10} {r['facts']:<10} {r['accuracy']*100:>6.1f}%    "
                  f"{r['avg_query_us']:>8.1f}us  {r['memory_mb']:>6.1f}MB")

    if persistence:
        print("\nPersistence:")
        print(f"  Loaded queries working: {persistence['loaded_queries']}/100")
        print(f"  New fact accuracy: {persistence['new_fact_accuracy']*100:.0f}%")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if hpc_results:
        max_hpc = hpc_results[-1]
        print(f"  Hippocampus max scale: {max_hpc['concepts']:,} concepts")
        print(f"  Memory efficiency: ~{max_hpc['memory_mb']/max_hpc['concepts']*1000:.2f} KB/concept")

    if unified_results:
        max_uni = unified_results[-1]
        print(f"  Unified AI max scale: {max_uni['entities']:,} entities, {max_uni['facts']:,} facts")
        print(f"  Accuracy at scale: {max_uni['accuracy']*100:.1f}%")
        print(f"  Query latency: {max_uni['avg_query_us']:.0f}us avg")
        print(f"  Memory at scale: {max_uni['memory_mb']:.0f} MB")


def main():
    """Run stress tests."""
    print("=" * 70)
    print("UNIFIED AI - STRESS TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Initial memory: {get_memory_mb():.1f} MB")

    # Run tests
    hpc_results = stress_test_hippocampus(max_concepts=50000)
    gc.collect()

    unified_results = stress_test_unified(max_entities=50000)
    gc.collect()

    persistence = test_persistence()

    # Summary
    print_stress_summary(hpc_results, unified_results, persistence)

    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'hippocampus': hpc_results,
        'unified': unified_results,
        'persistence': persistence
    }

    report_path = '/home/compsmart/htdocs/cloud.compsmart.co.uk/poc/unified-ai/models/stress_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved: {report_path}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
