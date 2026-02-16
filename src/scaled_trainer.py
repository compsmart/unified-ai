#!/usr/bin/env python3
"""
Scaled Trainer - Train Unified AI at Scale

This module:
1. Generates training data (entities, facts, relationships)
2. Scales up the system progressively
3. Monitors resource usage
4. Tests learning retention
5. Benchmarks at each scale level
"""

import numpy as np
import os
import sys
import json
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from hippocampus_vsa import HippocampusVSA, FactMemory
from predictive_lattice import PredictiveLattice, LatticeConfig
from wave_to_hv_bridge import RobustWaveToHVBridge, BridgeConfig
from unified_ai import UnifiedAI, UnifiedAIConfig


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources."""
    timestamp: str
    memory_mb: float
    memory_percent: float
    cpu_percent: float
    model_memory_kb: float


@dataclass
class ScaleTestResult:
    """Results from a scale test."""
    scale: int
    num_concepts: int
    num_facts: int
    num_sequences: int
    training_time_s: float
    query_accuracy: float
    avg_inference_us: float
    memory_mb: float
    memory_per_concept_kb: float


class ResourceMonitor:
    """Monitor system resources during training."""

    def __init__(self):
        self.process = psutil.Process()
        self.snapshots: List[ResourceSnapshot] = []
        self.start_memory = self._get_memory_mb()

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def snapshot(self, model_memory_kb: float = 0) -> ResourceSnapshot:
        """Take a resource snapshot."""
        snap = ResourceSnapshot(
            timestamp=datetime.now().isoformat(),
            memory_mb=self._get_memory_mb(),
            memory_percent=self.process.memory_percent(),
            cpu_percent=self.process.cpu_percent(interval=0.1),
            model_memory_kb=model_memory_kb
        )
        self.snapshots.append(snap)
        return snap

    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if not self.snapshots:
            return self._get_memory_mb()
        return max(s.memory_mb for s in self.snapshots)

    def get_memory_delta(self) -> float:
        """Get memory change since start in MB."""
        return self._get_memory_mb() - self.start_memory


class TrainingDataGenerator:
    """Generate synthetic training data at scale."""

    # Domain templates for realistic data
    DOMAINS = {
        'devices': {
            'entities': ['Drone', 'Robot', 'Sensor', 'Camera', 'Vehicle', 'Machine'],
            'roles': ['Status', 'Location', 'Battery', 'Speed', 'Mode', 'Temperature', 'Signal'],
            'status_values': ['Operational', 'Offline', 'Maintenance', 'Error', 'Standby', 'Active'],
            'location_values': ['Warehouse', 'Field', 'Base', 'Transit', 'Docking', 'Charging'],
            'level_values': ['Critical', 'Low', 'Medium', 'High', 'Full', 'Unknown']
        },
        'people': {
            'entities': ['User', 'Admin', 'Operator', 'Technician', 'Manager', 'Guest'],
            'roles': ['Role', 'Department', 'Access', 'Status', 'Location', 'Clearance'],
            'role_values': ['Engineer', 'Analyst', 'Supervisor', 'Specialist', 'Director'],
            'status_values': ['Active', 'Inactive', 'OnLeave', 'Remote', 'OnSite'],
            'access_values': ['Full', 'Limited', 'ReadOnly', 'Admin', 'Guest', 'None']
        },
        'systems': {
            'entities': ['Server', 'Database', 'Network', 'API', 'Service', 'Container'],
            'roles': ['Status', 'Load', 'Health', 'Version', 'Region', 'Tier'],
            'status_values': ['Running', 'Stopped', 'Degraded', 'Healthy', 'Critical'],
            'load_values': ['Idle', 'Light', 'Moderate', 'Heavy', 'Overloaded'],
            'region_values': ['US_East', 'US_West', 'EU', 'Asia', 'Global']
        }
    }

    # Sequence patterns for lattice training
    SEQUENCE_PATTERNS = [
        # State transitions
        ['Standby', 'Active', 'Running', 'Complete'],
        ['Offline', 'Booting', 'Ready', 'Operational'],
        ['Idle', 'Processing', 'Busy', 'Idle'],
        ['Error', 'Diagnosing', 'Repairing', 'Testing', 'Operational'],
        # Location transitions
        ['Base', 'Transit', 'Destination', 'Working', 'Transit', 'Base'],
        ['Docking', 'Charging', 'Ready', 'Deployed'],
        # Level progressions
        ['Critical', 'Low', 'Medium', 'High', 'Full'],
        ['Unknown', 'Detecting', 'Confirmed', 'Verified'],
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_concepts(self, scale: int) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate concepts at given scale.

        Returns dict with 'roles' and 'values' lists of (name, category) tuples.
        """
        concepts = {'roles': [], 'values': []}

        # Base concepts from domains
        for domain_name, domain in self.DOMAINS.items():
            for role in domain['roles']:
                concepts['roles'].append((f"{role}", 'role'))

            for key, values in domain.items():
                if key.endswith('_values'):
                    for val in values:
                        concepts['values'].append((f"{val}", 'value'))

        # Scale up with generated concepts
        num_extra = max(0, scale - len(concepts['values']))

        for i in range(num_extra // 2):
            concepts['roles'].append((f"Attr_{i}", 'role'))

        for i in range(num_extra):
            concepts['values'].append((f"Val_{i}", 'value'))

        return concepts

    def generate_entities(self, scale: int) -> List[str]:
        """Generate entity names at scale."""
        entities = []

        for domain_name, domain in self.DOMAINS.items():
            for entity_type in domain['entities']:
                # Generate multiple instances of each type
                num_instances = max(1, scale // (len(self.DOMAINS) * 6))
                for i in range(num_instances):
                    entities.append(f"{entity_type}_{i:04d}")

        return entities[:scale]

    def generate_facts(
        self,
        entities: List[str],
        roles: List[str],
        values: List[str],
        facts_per_entity: int = 3
    ) -> List[Tuple[str, str, str]]:
        """Generate fact triplets (entity, role, value)."""
        facts = []

        for entity in entities:
            # Select random roles and values for this entity
            selected_roles = self.rng.choice(
                roles,
                size=min(facts_per_entity, len(roles)),
                replace=False
            )

            for role in selected_roles:
                value = self.rng.choice(values)
                facts.append((entity, role, value))

        return facts

    def generate_sequences(self, scale: int) -> List[List[str]]:
        """Generate training sequences at scale."""
        sequences = []

        # Include base patterns
        sequences.extend(self.SEQUENCE_PATTERNS)

        # Generate additional sequences
        num_extra = max(0, scale // 10)

        for _ in range(num_extra):
            # Random length sequence
            length = self.rng.integers(3, 7)
            seq = [f"State_{self.rng.integers(0, 100)}" for _ in range(length)]
            sequences.append(seq)

        return sequences


class ScaledTrainer:
    """Train and benchmark the Unified AI at scale."""

    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir or '/home/compsmart/htdocs/cloud.compsmart.co.uk/poc/unified-ai/models'
        self.data_gen = TrainingDataGenerator()
        self.monitor = ResourceMonitor()
        self.results: List[ScaleTestResult] = []

        os.makedirs(self.save_dir, exist_ok=True)

    def create_scaled_ai(
        self,
        lattice_size: int = 512,
        hv_dimensions: int = 10000
    ) -> UnifiedAI:
        """Create a scaled UnifiedAI instance."""
        config = UnifiedAIConfig(
            lattice_size=lattice_size,
            hv_dimensions=hv_dimensions,
            learning_rate=0.12,
            similarity_threshold=0.12
        )
        return UnifiedAI(config)

    def train_at_scale(
        self,
        scale: int,
        lattice_size: int = 512,
        hv_dimensions: int = 10000,
        facts_per_entity: int = 3,
        sequence_reps: int = 30,
        verbose: bool = True
    ) -> ScaleTestResult:
        """
        Train the system at a given scale.

        Args:
            scale: Target number of concepts/entities
            lattice_size: Lattice oscillator count
            hv_dimensions: Hypervector dimensions
            facts_per_entity: Facts to store per entity
            sequence_reps: Repetitions for sequence training
            verbose: Print progress

        Returns:
            ScaleTestResult with metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING AT SCALE: {scale}")
            print(f"{'='*60}")

        start_time = time.time()
        self.monitor.snapshot()

        # Create AI
        ai = self.create_scaled_ai(lattice_size, hv_dimensions)

        # Generate data
        if verbose:
            print(f"\n1. Generating training data...")

        concepts = self.data_gen.generate_concepts(scale)
        entities = self.data_gen.generate_entities(scale)
        sequences = self.data_gen.generate_sequences(scale)

        # Register concepts
        if verbose:
            print(f"   Registering {len(concepts['roles'])} roles + {len(concepts['values'])} values...")

        role_names = []
        value_names = []

        # Assign neurons to concepts
        neurons_per_concept = max(2, lattice_size // (len(concepts['roles']) + len(concepts['values']) + 1))

        neuron_idx = 0
        for name, category in concepts['roles']:
            neurons = list(range(neuron_idx, min(neuron_idx + neurons_per_concept, lattice_size)))
            ai.register_concept(name, category='role', neuron_indices=neurons)
            role_names.append(name)
            neuron_idx = (neuron_idx + neurons_per_concept) % lattice_size

        for name, category in concepts['values']:
            neurons = list(range(neuron_idx, min(neuron_idx + neurons_per_concept, lattice_size)))
            ai.register_concept(name, category='value', neuron_indices=neurons)
            value_names.append(name)
            neuron_idx = (neuron_idx + neurons_per_concept) % lattice_size

        # Also register sequence concepts
        for seq in sequences:
            for state in seq:
                if state not in value_names:
                    neurons = list(range(neuron_idx, min(neuron_idx + neurons_per_concept, lattice_size)))
                    ai.register_concept(state, category='state', neuron_indices=neurons)
                    value_names.append(state)
                    neuron_idx = (neuron_idx + neurons_per_concept) % lattice_size

        self.monitor.snapshot()

        # Generate and store facts
        if verbose:
            print(f"   Generating facts for {len(entities)} entities...")

        facts = self.data_gen.generate_facts(entities, role_names, value_names, facts_per_entity)

        if verbose:
            print(f"   Storing {len(facts)} facts...")

        stored = 0
        for entity, role, value in facts:
            result = ai.store_fact(entity, role, value)
            if result['success']:
                stored += 1

        self.monitor.snapshot()

        # Train sequences
        if verbose:
            print(f"   Training {len(sequences)} sequences ({sequence_reps} reps each)...")

        for seq in sequences:
            # Filter to registered concepts only
            valid_seq = [s for s in seq if s in value_names]
            if len(valid_seq) >= 2:
                ai.train_sequence(valid_seq, repetitions=sequence_reps, steps_per_concept=3)

        self.monitor.snapshot()

        training_time = time.time() - start_time

        # Test accuracy
        if verbose:
            print(f"\n2. Testing accuracy...")

        correct = 0
        total = 0
        inference_times = []

        # Sample facts for testing
        test_facts = facts[:min(100, len(facts))]

        for entity, role, value in test_facts:
            start = time.perf_counter()
            result = ai.query_fact(entity, role)
            inference_times.append(time.perf_counter() - start)

            total += 1
            if result['value'] == value:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        avg_inference = np.mean(inference_times) * 1_000_000  # microseconds

        # Get memory stats
        stats = ai.get_stats()
        model_memory = stats['memory']['total_kb']

        self.monitor.snapshot(model_memory)

        # Create result
        result = ScaleTestResult(
            scale=scale,
            num_concepts=len(concepts['roles']) + len(concepts['values']),
            num_facts=stored,
            num_sequences=len(sequences),
            training_time_s=training_time,
            query_accuracy=accuracy,
            avg_inference_us=avg_inference,
            memory_mb=self.monitor.get_peak_memory(),
            memory_per_concept_kb=model_memory / max(1, len(concepts['roles']) + len(concepts['values']))
        )

        self.results.append(result)

        if verbose:
            print(f"\n3. Results:")
            print(f"   Concepts:    {result.num_concepts}")
            print(f"   Facts:       {result.num_facts}")
            print(f"   Sequences:   {result.num_sequences}")
            print(f"   Accuracy:    {result.query_accuracy:.1%}")
            print(f"   Inference:   {result.avg_inference_us:.1f} us")
            print(f"   Memory:      {result.memory_mb:.1f} MB")
            print(f"   Time:        {result.training_time_s:.1f}s")

        # Save model
        model_path = os.path.join(self.save_dir, f'scale_{scale}')
        ai.save(model_path)
        if verbose:
            print(f"\n   Saved to: {model_path}")

        # Cleanup
        del ai
        gc.collect()

        return result

    def test_learning_retention(
        self,
        scale: int = 100,
        iterations: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Test that learning accumulates over time.

        Trains incrementally and verifies old knowledge is retained.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"LEARNING RETENTION TEST")
            print(f"{'='*60}")

        ai = self.create_scaled_ai(lattice_size=256, hv_dimensions=10000)

        # Base concepts
        ai.register_concept("Status", category='role')
        ai.register_concept("Location", category='role')

        retention_results = []
        all_facts = []

        for iteration in range(iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{iterations} ---")

            # Add new concepts and facts
            for i in range(10):
                val_name = f"Value_iter{iteration}_{i}"
                ai.register_concept(val_name, category='value')

            # Add facts
            new_facts = []
            for i in range(20):
                entity = f"Entity_{iteration}_{i}"
                value = f"Value_iter{iteration}_{i % 10}"
                ai.store_fact(entity, "Status", value)
                new_facts.append((entity, "Status", value))

            all_facts.extend(new_facts)

            # Test ALL previous facts (retention)
            correct = 0
            for entity, role, expected_value in all_facts:
                result = ai.query_fact(entity, role)
                if result['value'] == expected_value:
                    correct += 1

            retention = correct / len(all_facts)

            retention_results.append({
                'iteration': iteration + 1,
                'total_facts': len(all_facts),
                'correct': correct,
                'retention': retention
            })

            if verbose:
                print(f"   Total facts: {len(all_facts)}, Retention: {retention:.1%}")

        # Save final model
        model_path = os.path.join(self.save_dir, 'retention_test')
        ai.save(model_path)

        return {
            'iterations': iterations,
            'final_facts': len(all_facts),
            'final_retention': retention_results[-1]['retention'],
            'results': retention_results
        }

    def run_scale_benchmark(
        self,
        scales: List[int] = [100, 500, 1000, 2000, 5000],
        verbose: bool = True
    ) -> List[ScaleTestResult]:
        """Run benchmarks at multiple scales."""
        if verbose:
            print("\n" + "=" * 70)
            print("SCALE BENCHMARK")
            print("=" * 70)

        for scale in scales:
            # Adjust lattice size based on scale
            if scale <= 100:
                lattice_size = 256
            elif scale <= 500:
                lattice_size = 512
            elif scale <= 2000:
                lattice_size = 1024
            else:
                lattice_size = 2048

            self.train_at_scale(
                scale=scale,
                lattice_size=lattice_size,
                hv_dimensions=10000,
                verbose=verbose
            )

            # Check memory pressure
            snap = self.monitor.snapshot()
            if snap.memory_percent > 80:
                print(f"\nWARNING: Memory usage at {snap.memory_percent:.1f}%, stopping scale-up")
                break

            gc.collect()

        return self.results

    def print_summary(self):
        """Print summary of all results."""
        print("\n" + "=" * 70)
        print("SCALE BENCHMARK SUMMARY")
        print("=" * 70)

        print(f"\n{'Scale':<10} {'Concepts':<10} {'Facts':<10} {'Accuracy':<10} {'Latency':<12} {'Memory':<10}")
        print("-" * 70)

        for r in self.results:
            print(f"{r.scale:<10} {r.num_concepts:<10} {r.num_facts:<10} "
                  f"{r.query_accuracy*100:>6.1f}%    {r.avg_inference_us:>8.1f}us  {r.memory_mb:>6.1f}MB")

        # Memory scaling analysis
        if len(self.results) > 1:
            print("\nMemory Scaling:")
            for r in self.results:
                print(f"  Scale {r.scale}: {r.memory_per_concept_kb:.2f} KB/concept")

    def save_results(self, path: str = None):
        """Save results to JSON."""
        path = path or os.path.join(self.save_dir, 'benchmark_results.json')

        data = {
            'timestamp': datetime.now().isoformat(),
            'results': [asdict(r) for r in self.results],
            'resource_snapshots': [asdict(s) for s in self.monitor.snapshots]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {path}")


def main():
    """Run the scaled training benchmark."""
    print("=" * 70)
    print("UNIFIED AI - SCALED TRAINING & VERIFICATION")
    print("=" * 70)

    # Initial resource check
    print("\n1. Initial Resource Check:")
    monitor = ResourceMonitor()
    snap = monitor.snapshot()
    print(f"   Memory: {snap.memory_mb:.1f} MB ({snap.memory_percent:.1f}%)")

    # Create trainer
    trainer = ScaledTrainer()

    # Run scale benchmarks
    print("\n2. Running Scale Benchmarks...")
    scales = [100, 500, 1000, 2000, 5000]
    trainer.run_scale_benchmark(scales)

    # Test learning retention
    print("\n3. Testing Learning Retention...")
    retention = trainer.test_learning_retention(scale=100, iterations=5)
    print(f"\n   Final retention: {retention['final_retention']:.1%} over {retention['final_facts']} facts")

    # Print summary
    trainer.print_summary()

    # Save results
    trainer.save_results()

    # Final resource check
    print("\n4. Final Resource Check:")
    snap = monitor.snapshot()
    print(f"   Memory: {snap.memory_mb:.1f} MB ({snap.memory_percent:.1f}%)")
    print(f"   Memory delta: +{monitor.get_memory_delta():.1f} MB")

    print("\n" + "=" * 70)
    print("SCALED TRAINING COMPLETE")
    print("=" * 70)

    return trainer.results


if __name__ == "__main__":
    results = main()
