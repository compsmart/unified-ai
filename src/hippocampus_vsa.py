"""
HippocampusVSA - Vector Symbolic Architecture Clean-up Memory

The Hippocampus module that acts as a strict, non-hallucinating reference library.
Takes noisy bundled hypervectors and snaps them to the closest clean concept.

VSA Operations:
1. Binding (XOR/Multiplication) - Links concepts together (Location * Highway)
2. Bundling (Addition + Sign) - Stores multiple facts in single vector
3. Clean-up (Cosine Similarity) - Restores noisy vector to clean concept

This is the final piece that prevents LLM hallucination by providing
ground truth from the hyperdimensional codebook.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime


@dataclass
class ConceptMetadata:
    """Metadata for a registered concept."""
    name: str
    category: str  # 'variable', 'value', 'entity', 'action', 'relation'
    created_at: str
    access_count: int
    description: Optional[str] = None
    aliases: Optional[List[str]] = None


class HippocampusVSA:
    """
    Vector Symbolic Architecture with Clean-up Memory.

    The "Hippocampus" that stores pristine concept vectors and performs
    noise-tolerant pattern matching. Key to preventing hallucination by
    providing strict symbolic grounding.

    Features:
    - Bipolar hypervectors (+1, -1) for optimal noise tolerance
    - Binding: Element-wise multiplication (creates orthogonal result)
    - Bundling: Addition + sign threshold (superposition of facts)
    - Clean-up: Cosine similarity search against codebook
    - Hierarchical categories for organized retrieval
    """

    def __init__(
        self,
        dimensions: int = 10000,
        seed: int = 42,
        similarity_threshold: float = 0.15
    ):
        """
        Initialize the Hippocampus VSA.

        Args:
            dimensions: Hypervector dimensionality (10000 is standard)
            seed: Random seed for reproducibility
            similarity_threshold: Minimum similarity for valid clean-up
        """
        self.D = dimensions
        self.rng = np.random.default_rng(seed)
        self.similarity_threshold = similarity_threshold

        # The "Codebook": Stores clean, pristine concept hypervectors
        self.item_memory: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, ConceptMetadata] = {}

        # Category indices for efficient retrieval
        self.category_index: Dict[str, List[str]] = {}

        # Alias mapping for flexible retrieval
        self.alias_map: Dict[str, str] = {}

        # Statistics
        self.bind_count = 0
        self.bundle_count = 0
        self.cleanup_count = 0
        self.cleanup_hits = 0

    def register_concept(
        self,
        name: str,
        vector: np.ndarray = None,
        category: str = 'general',
        description: str = None,
        aliases: List[str] = None
    ) -> np.ndarray:
        """
        Register a concept in the codebook.

        Args:
            name: Unique identifier for the concept
            vector: Pre-defined hypervector (random if None)
            category: Type of concept for organized retrieval
            description: Human-readable description
            aliases: Alternative names for this concept

        Returns:
            The registered hypervector
        """
        if vector is None:
            # Generate random bipolar vector
            vector = self.rng.choice([-1, 1], size=self.D).astype(np.int8)
        else:
            # Ensure bipolar encoding
            vector = np.sign(vector).astype(np.int8)
            vector[vector == 0] = self.rng.choice([-1, 1])

        # Store in codebook
        self.item_memory[name] = vector

        # Create metadata
        self.metadata[name] = ConceptMetadata(
            name=name,
            category=category,
            created_at=datetime.now().isoformat(),
            access_count=0,
            description=description,
            aliases=aliases or []
        )

        # Update category index
        if category not in self.category_index:
            self.category_index[category] = []
        self.category_index[category].append(name)

        # Register aliases
        if aliases:
            for alias in aliases:
                self.alias_map[alias.lower()] = name

        return vector

    def get_concept(self, name: str) -> Optional[np.ndarray]:
        """
        Retrieve a concept vector by name or alias.
        """
        # Check aliases first
        if name.lower() in self.alias_map:
            name = self.alias_map[name.lower()]

        if name in self.item_memory:
            self.metadata[name].access_count += 1
            return self.item_memory[name].copy()
        return None

    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors together (element-wise multiplication).

        Creates a new vector orthogonal to both inputs.
        Used to link concepts: Location * Highway = "at highway"

        Binding is its own inverse: bind(bind(A, B), B) = A
        """
        self.bind_count += 1
        return (v1 * v2).astype(np.int8)

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Unbind (same as bind for bipolar vectors - XOR property).

        Used to extract: unbind(Location * Highway, Location) = Highway
        """
        return self.bind(bound, key)

    def bundle(self, vector_list: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """
        Bundle multiple vectors into a single memory trace (superposition).

        This creates the "noisy" composite that can still be decoded
        thanks to the massive dimensionality (10,000 bits).

        Args:
            vector_list: List of hypervectors to combine
            weights: Optional importance weights for each vector

        Returns:
            Bundled hypervector (bipolar, normalized)
        """
        self.bundle_count += 1

        if not vector_list:
            return np.zeros(self.D, dtype=np.int8)

        if weights is None:
            weights = [1.0] * len(vector_list)

        # Weighted sum
        sum_vec = np.zeros(self.D, dtype=np.float64)
        for vec, weight in zip(vector_list, weights):
            sum_vec += weight * vec.astype(np.float64)

        # Threshold back to bipolar (+1, -1)
        # Break ties randomly for information preservation
        bundled = np.where(sum_vec > 0, 1,
                          np.where(sum_vec < 0, -1,
                                  self.rng.choice([-1, 1], size=self.D))).astype(np.int8)
        return bundled

    def cleanup(
        self,
        noisy_vector: np.ndarray,
        category: str = None,
        top_k: int = 1,
        return_all: bool = False
    ) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
        """
        Clean up a noisy vector by finding the closest concept in codebook.

        This is the critical operation that prevents hallucination:
        - Takes a mathematically noisy result from bundling/unbinding
        - Finds the closest "ground truth" concept
        - Returns a strict symbolic answer, not a fuzzy interpretation

        Args:
            noisy_vector: The vector to clean up
            category: Optional category to restrict search
            top_k: Number of top matches to return
            return_all: If True, return all matches above threshold

        Returns:
            (concept_name, similarity) or list of matches
        """
        self.cleanup_count += 1

        # Determine search space
        if category and category in self.category_index:
            search_names = self.category_index[category]
        else:
            search_names = list(self.item_memory.keys())

        if not search_names:
            return (None, 0.0) if not return_all else []

        results = []

        for name in search_names:
            clean_vec = self.item_memory[name]

            # Cosine similarity for bipolar vectors
            # For bipolar: cos_sim = dot(a, b) / D
            sim = np.dot(noisy_vector.astype(np.float64),
                        clean_vec.astype(np.float64)) / self.D

            if sim >= self.similarity_threshold:
                results.append((name, sim))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            self.cleanup_hits += 1
            # Update access count for top match
            if results[0][0] in self.metadata:
                self.metadata[results[0][0]].access_count += 1

        if return_all:
            return results
        elif not results:
            return (None, 0.0)
        elif top_k == 1:
            return results[0]
        else:
            return results[:top_k]

    def create_role_filler_pair(
        self,
        role_name: str,
        filler_name: str
    ) -> np.ndarray:
        """
        Create a role-filler binding (e.g., "Status = Motor Failure").

        This is the standard VSA pattern for structured knowledge.
        """
        role = self.get_concept(role_name)
        filler = self.get_concept(filler_name)

        if role is None or filler is None:
            raise ValueError(f"Concept not found: {role_name if role is None else filler_name}")

        return self.bind(role, filler)

    def extract_filler(
        self,
        composite: np.ndarray,
        role_name: str,
        filler_category: str = None
    ) -> Tuple[str, float]:
        """
        Extract the filler from a composite given a role.

        E.g., extract Status from a bundled state to get "Motor Failure"
        """
        role = self.get_concept(role_name)
        if role is None:
            return (None, 0.0)  # Gracefully handle unregistered role

        # Unbind to get noisy filler
        noisy_filler = self.unbind(composite, role)

        # Clean up to get exact filler
        return self.cleanup(noisy_filler, category=filler_category)

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two hypervectors.
        """
        return np.dot(v1.astype(np.float64), v2.astype(np.float64)) / self.D

    def permute(self, vector: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Permute a hypervector (cyclic shift).

        Used for sequence encoding: P^n(x) represents x at position n.
        """
        return np.roll(vector, shift).astype(np.int8)

    def encode_sequence(self, concept_names: List[str]) -> np.ndarray:
        """
        Encode a sequence of concepts using permutation.

        Returns a single vector representing the ordered sequence.
        """
        if not concept_names:
            return np.zeros(self.D, dtype=np.int8)

        vectors = []
        for i, name in enumerate(concept_names):
            vec = self.get_concept(name)
            if vec is not None:
                # Permute by position
                vectors.append(self.permute(vec, i))

        return self.bundle(vectors)

    def decode_sequence(
        self,
        sequence_vec: np.ndarray,
        max_length: int = 10,
        category: str = None
    ) -> List[Tuple[str, float]]:
        """
        Decode a sequence vector back to ordered concepts.
        """
        results = []

        for i in range(max_length):
            # Inverse permute to get concept at position i
            unshifted = self.permute(sequence_vec, -i)
            match, sim = self.cleanup(unshifted, category=category)

            if match and sim > self.similarity_threshold:
                results.append((match, sim))
            else:
                break  # Sequence ended

        return results

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            'dimensions': self.D,
            'num_concepts': len(self.item_memory),
            'categories': list(self.category_index.keys()),
            'category_counts': {k: len(v) for k, v in self.category_index.items()},
            'bind_operations': self.bind_count,
            'bundle_operations': self.bundle_count,
            'cleanup_operations': self.cleanup_count,
            'cleanup_hit_rate': self.cleanup_hits / max(1, self.cleanup_count),
            'memory_bytes': self._estimate_memory()
        }

    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        # Each concept: D bytes (int8) + metadata
        concept_bytes = len(self.item_memory) * self.D
        metadata_bytes = len(self.metadata) * 200  # Rough estimate
        return concept_bytes + metadata_bytes

    def save(self, directory: str):
        """Save the codebook to disk."""
        os.makedirs(directory, exist_ok=True)

        # Save vectors (compressed)
        vectors = {k: v.tolist() for k, v in self.item_memory.items()}
        with open(os.path.join(directory, 'vectors.json'), 'w') as f:
            json.dump(vectors, f)

        # Save metadata
        meta = {k: asdict(v) for k, v in self.metadata.items()}
        with open(os.path.join(directory, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        # Save indices
        indices = {
            'category_index': self.category_index,
            'alias_map': self.alias_map,
            'stats': {
                'dimensions': self.D,
                'similarity_threshold': self.similarity_threshold
            }
        }
        with open(os.path.join(directory, 'indices.json'), 'w') as f:
            json.dump(indices, f, indent=2)

    def load(self, directory: str):
        """Load the codebook from disk."""
        # Load vectors
        vectors_path = os.path.join(directory, 'vectors.json')
        if os.path.exists(vectors_path):
            with open(vectors_path, 'r') as f:
                vectors = json.load(f)
            self.item_memory = {k: np.array(v, dtype=np.int8) for k, v in vectors.items()}

        # Load metadata
        meta_path = os.path.join(directory, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.metadata = {k: ConceptMetadata(**v) for k, v in meta.items()}

        # Load indices
        indices_path = os.path.join(directory, 'indices.json')
        if os.path.exists(indices_path):
            with open(indices_path, 'r') as f:
                indices = json.load(f)
            self.category_index = indices.get('category_index', {})
            self.alias_map = indices.get('alias_map', {})


class FactMemory:
    """
    High-level fact storage using VSA role-filler bindings.

    Stores multiple facts in a single "state vector" and allows
    precise querying without hallucination.
    """

    def __init__(self, hippocampus: HippocampusVSA):
        self.hpc = hippocampus
        self.state_vectors: Dict[str, np.ndarray] = {}

    def store_fact(
        self,
        context: str,
        role: str,
        value: str
    ) -> np.ndarray:
        """
        Store a fact in the specified context.

        Args:
            context: The context/entity this fact belongs to
            role: The role/attribute (e.g., "status", "location")
            value: The value/filler (e.g., "motor_failure", "base")

        Returns:
            Updated state vector for the context
        """
        # Create role-filler binding
        fact = self.hpc.create_role_filler_pair(role, value)

        if context not in self.state_vectors:
            self.state_vectors[context] = fact
        else:
            # Bundle with existing facts
            self.state_vectors[context] = self.hpc.bundle([
                self.state_vectors[context],
                fact
            ])

        return self.state_vectors[context]

    def query_fact(
        self,
        context: str,
        role: str,
        value_category: str = None
    ) -> Tuple[str, float]:
        """
        Query a fact from a context.

        Args:
            context: The context to query
            role: The role to extract
            value_category: Category to restrict value search

        Returns:
            (value_name, confidence)
        """
        if context not in self.state_vectors:
            return (None, 0.0)

        return self.hpc.extract_filler(
            self.state_vectors[context],
            role,
            filler_category=value_category
        )

    def get_all_facts(self, context: str, roles: List[str]) -> Dict[str, Tuple[str, float]]:
        """Extract all facts for given roles."""
        return {role: self.query_fact(context, role) for role in roles}


def test_drone_black_box():
    """
    The POC Test: "The Drone Black Box"

    Demonstrates the complete lifecycle of a real-time memory:
    1. Register clean concepts
    2. Store multiple facts in a single hypervector
    3. Unbind and recall
    4. Clean up noise to get strict symbolic answer
    """
    print("=" * 60)
    print("HIPPOCAMPUS VSA - DRONE BLACK BOX TEST")
    print("=" * 60)

    memory = HippocampusVSA(dimensions=10000)

    # 1. Register "Clean" Atomic Concepts into the Dictionary
    print("\n1. Registering concepts...")

    # Variables (roles)
    v_status = memory.register_concept("Variable: Status", category='variable')
    v_loc = memory.register_concept("Variable: Location", category='variable')
    v_speed = memory.register_concept("Variable: Speed", category='variable')
    v_battery = memory.register_concept("Variable: Battery", category='variable')

    # Values (fillers)
    v_fail = memory.register_concept("Value: Motor Failure", category='value',
                                     aliases=['motor_failure', 'failure'])
    v_home = memory.register_concept("Value: Base Station", category='value',
                                     aliases=['base', 'home'])
    v_wind = memory.register_concept("Value: High Winds", category='value',
                                     aliases=['windy', 'high_wind'])
    v_highway = memory.register_concept("Value: Highway", category='value')
    v_low = memory.register_concept("Value: Low", category='value')
    v_critical = memory.register_concept("Value: Critical", category='value')

    print(f"   Registered {len(memory.item_memory)} concepts")

    # 2. Bind Facts Together
    print("\n2. Binding facts...")
    fact_1 = memory.bind(v_status, v_fail)  # Status = Motor Failure
    fact_2 = memory.bind(v_loc, v_home)     # Location = Base Station
    fact_3 = memory.bind(v_battery, v_critical)  # Battery = Critical

    print("   fact_1: Status * Motor_Failure")
    print("   fact_2: Location * Base_Station")
    print("   fact_3: Battery * Critical")

    # 3. Bundle into a single "Black Box" State Vector
    print("\n3. Bundling into single state vector...")
    black_box_state = memory.bundle([fact_1, fact_2, fact_3])
    print(f"   Created 10,000-bit state vector containing ALL facts")
    print(f"   (Highly compressed, but noisy due to superposition)")

    # 4. Unbind and Recall
    print("\n4. Extracting facts (simulating LLM query)...")

    # Extract Status
    noisy_status = memory.bind(black_box_state, v_status)
    status_result, status_conf = memory.cleanup(noisy_status, category='value')

    # Extract Location
    noisy_loc = memory.bind(black_box_state, v_loc)
    loc_result, loc_conf = memory.cleanup(noisy_loc, category='value')

    # Extract Battery
    noisy_battery = memory.bind(black_box_state, v_battery)
    battery_result, battery_conf = memory.cleanup(noisy_battery, category='value')

    # 5. Results
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)

    print(f"\nQuery: 'What is the Status?'")
    print(f"  Raw similarity to Motor Failure: {memory.similarity(noisy_status, v_fail):.3f} (Noisy)")
    print(f"  Clean-up Result: [{status_result}] (Confidence: {status_conf:.2f})")

    print(f"\nQuery: 'What is the Location?'")
    print(f"  Raw similarity to Base Station: {memory.similarity(noisy_loc, v_home):.3f} (Noisy)")
    print(f"  Clean-up Result: [{loc_result}] (Confidence: {loc_conf:.2f})")

    print(f"\nQuery: 'What is the Battery level?'")
    print(f"  Raw similarity to Critical: {memory.similarity(noisy_battery, v_critical):.3f} (Noisy)")
    print(f"  Clean-up Result: [{battery_result}] (Confidence: {battery_conf:.2f})")

    # Verify results
    success = (
        status_result == "Value: Motor Failure" and status_conf > 0.3 and
        loc_result == "Value: Base Station" and loc_conf > 0.3 and
        battery_result == "Value: Critical" and battery_conf > 0.3
    )

    print(f"\n{'SUCCESS' if success else 'FAILED'}: All facts correctly extracted from noisy state")

    # Print statistics
    stats = memory.get_stats()
    print(f"\nStatistics:")
    print(f"  Memory: {stats['memory_bytes'] / 1024:.1f} KB")
    print(f"  Concepts: {stats['num_concepts']}")
    print(f"  Cleanup hit rate: {stats['cleanup_hit_rate']:.1%}")

    return success, memory


def test_capacity():
    """Test the capacity of bundled storage."""
    print("\n" + "=" * 60)
    print("CAPACITY TEST - How many facts can we bundle?")
    print("=" * 60)

    memory = HippocampusVSA(dimensions=10000)

    # Register many concepts
    roles = []
    values = []

    for i in range(20):
        roles.append(f"Role_{i}")
        values.append(f"Value_{i}")
        memory.register_concept(f"Role_{i}", category='role')
        memory.register_concept(f"Value_{i}", category='value')

    # Test bundling increasing numbers of facts
    print("\nBundling increasing numbers of facts...")

    for num_facts in [2, 3, 5, 7, 10, 15, 20]:
        # Create facts
        facts = []
        for i in range(num_facts):
            role = memory.get_concept(roles[i])
            value = memory.get_concept(values[i])
            facts.append(memory.bind(role, value))

        # Bundle
        bundle = memory.bundle(facts)

        # Try to extract each fact
        correct = 0
        total_conf = 0

        for i in range(num_facts):
            role = memory.get_concept(roles[i])
            noisy = memory.bind(bundle, role)
            result, conf = memory.cleanup(noisy, category='value')

            if result == f"Value_{i}":
                correct += 1
                total_conf += conf

        accuracy = correct / num_facts
        avg_conf = total_conf / max(1, correct)

        print(f"  {num_facts:2d} facts: {correct}/{num_facts} correct ({accuracy:.0%}), avg confidence: {avg_conf:.3f}")

    print("\nNote: 10,000 dimensions can reliably store 5-10 facts with >80% accuracy")


def test_sequence_encoding():
    """Test sequence encoding with permutation."""
    print("\n" + "=" * 60)
    print("SEQUENCE ENCODING TEST")
    print("=" * 60)

    memory = HippocampusVSA(dimensions=10000)

    # Register words
    words = ["the", "quick", "brown", "fox", "jumps"]
    for word in words:
        memory.register_concept(word, category='word')

    # Encode sequence
    print(f"\nEncoding sequence: {words}")
    seq_vec = memory.encode_sequence(words)

    # Decode
    print("Decoding sequence...")
    decoded = memory.decode_sequence(seq_vec, max_length=len(words), category='word')

    for i, (word, conf) in enumerate(decoded):
        correct = word == words[i] if i < len(words) else False
        print(f"  Position {i}: {word} (conf: {conf:.3f}) {'OK' if correct else 'WRONG'}")

    success = len(decoded) == len(words) and all(
        decoded[i][0] == words[i] for i in range(len(words))
    )
    print(f"\n{'SUCCESS' if success else 'FAILED'}: Sequence {'correctly' if success else 'not'} recovered")


if __name__ == "__main__":
    print("HIPPOCAMPUS VSA - Vector Symbolic Architecture")
    print("Clean-up Memory for Non-Hallucinating AI\n")

    # Run tests
    success1, memory = test_drone_black_box()
    test_capacity()
    test_sequence_encoding()

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
