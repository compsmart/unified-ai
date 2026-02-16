"""
Reasoning Engine

Handles:
- Direct fact lookup
- Transitive reasoning (A→B→C)
- Type inference
- Confidence propagation
"""

import sys
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_ai import UnifiedAI


@dataclass
class ReasoningResult:
    """Result of a reasoning query."""
    found: bool
    value: Any
    confidence: float
    reasoning_chain: List[str]
    source: str  # "direct", "inferred", "predicted", "unknown"

    def explain(self) -> str:
        if not self.found:
            return "I don't know that."
        if self.source == "direct":
            return f"{self.value}"
        elif self.source == "inferred":
            chain = " → ".join(self.reasoning_chain)
            return f"{self.value} (via: {chain})"
        elif self.source == "predicted":
            return f"Probably {self.value} (prediction)"
        return str(self.value)


class Reasoner:
    """
    Reasoning engine built on UnifiedAI.

    Provides:
    - Direct lookup
    - Transitive queries
    - Type-based inference
    """

    def __init__(self, ai: UnifiedAI):
        self.ai = ai
        self.max_chain_depth = 5

    def query(
        self,
        entity: str,
        role: str,
        follow_chain: bool = True
    ) -> ReasoningResult:
        """
        Query for a fact, optionally following chains.

        Examples:
            query("Alice", "job") → "Engineer"
            query("Alice", "location") with chain:
                Alice.workplace = Acme
                Acme.location = London
                → "London" (via Alice → Acme → London)
        """
        # Normalize case
        entity = entity.lower()
        role = role.lower()

        # Direct lookup first
        result = self.ai.query_fact(entity, role)

        if result['value']:
            return ReasoningResult(
                found=True,
                value=result['value'],
                confidence=result['confidence'],
                reasoning_chain=[f"{entity}.{role}"],
                source="direct"
            )

        if not follow_chain:
            return ReasoningResult(
                found=False,
                value=None,
                confidence=0.0,
                reasoning_chain=[],
                source="unknown"
            )

        # Try transitive reasoning
        chain_result = self._follow_chain(entity, role, depth=0, visited=set())
        if chain_result:
            return chain_result

        return ReasoningResult(
            found=False,
            value=None,
            confidence=0.0,
            reasoning_chain=[],
            source="unknown"
        )

    def _follow_chain(
        self,
        entity: str,
        target_role: str,
        depth: int,
        visited: set
    ) -> Optional[ReasoningResult]:
        """Follow a chain of facts to find the answer."""
        if depth >= self.max_chain_depth:
            return None

        entity = entity.lower()
        target_role = target_role.lower()

        if entity in visited:
            return None

        visited.add(entity)

        # Get all known facts about this entity
        known_roles = ['type', 'job', 'workplace', 'location', 'company', 'belongs_to', 'part_of', 'is']

        for role in known_roles:
            result = self.ai.query_fact(entity, role)
            if result['value']:
                intermediate = result['value']

                # Check if intermediate has the target role
                final = self.ai.query_fact(intermediate, target_role)
                if final['value']:
                    return ReasoningResult(
                        found=True,
                        value=final['value'],
                        confidence=result['confidence'] * final['confidence'] * 0.9,
                        reasoning_chain=[
                            f"{entity}.{role}={intermediate}",
                            f"{intermediate}.{target_role}={final['value']}"
                        ],
                        source="inferred"
                    )

                # Recurse
                deeper = self._follow_chain(intermediate, target_role, depth + 1, visited)
                if deeper:
                    deeper.reasoning_chain.insert(0, f"{entity}.{role}={intermediate}")
                    deeper.confidence *= 0.9
                    return deeper

        return None

    def query_all(self, entity: str) -> Dict[str, ReasoningResult]:
        """Get all known facts about an entity."""
        results = {}
        known_roles = ['type', 'job', 'location', 'workplace', 'status',
                       'has', 'knows', 'name', 'belongs_to']

        for role in known_roles:
            result = self.query(entity, role, follow_chain=False)
            if result.found:
                results[role] = result

        return results

    def check_type(self, entity: str, expected_type: str) -> ReasoningResult:
        """Check if an entity is of a certain type."""
        result = self.query(entity, "type")

        if result.found:
            matches = result.value.lower() == expected_type.lower()
            return ReasoningResult(
                found=True,
                value=matches,
                confidence=result.confidence,
                reasoning_chain=result.reasoning_chain + [f"matches {expected_type}: {matches}"],
                source=result.source
            )

        return ReasoningResult(
            found=False,
            value=None,
            confidence=0.0,
            reasoning_chain=[],
            source="unknown"
        )

    def predict_next(self, context: str) -> ReasoningResult:
        """Predict what comes next in a sequence."""
        # Normalize to lowercase
        context = context.lower()

        # Check if concept exists and has neurons
        neurons = self.ai.semantic_bridge.concepts_to_neurons([context])

        if not neurons:
            return ReasoningResult(
                found=False,
                value=None,
                confidence=0.0,
                reasoning_chain=[],
                source="unknown"
            )

        result = self.ai.predict_next(current_concepts=[context], steps=5)
        predictions = result.get('predicted_concepts', [])

        if predictions:
            best_concept, best_conf = predictions[0]
            return ReasoningResult(
                found=True,
                value=best_concept,
                confidence=best_conf,
                reasoning_chain=[f"after {context}"],
                source="predicted"
            )

        return ReasoningResult(
            found=False,
            value=None,
            confidence=0.0,
            reasoning_chain=[],
            source="unknown"
        )

    def store_fact(
        self,
        entity: str,
        role: str,
        value: str,
        auto_register: bool = True
    ) -> bool:
        """Store a fact, auto-registering concepts if needed."""
        # Normalize case
        entity = entity.lower()
        role = role.lower()
        value = value.lower()

        if auto_register:
            # Register role if new
            if role not in self.ai.hippocampus.item_memory:
                self.ai.register_concept(role, category='role')

            # Register value if new
            if value not in self.ai.hippocampus.item_memory:
                self.ai.register_concept(value, category='value')

        result = self.ai.store_fact(entity, role, value)
        return result.get('success', False)

    def learn_sequence(self, sequence: List[str]) -> bool:
        """Learn a sequence of concepts."""
        # Normalize to lowercase
        sequence = [s.lower() for s in sequence]

        lattice_size = self.ai.config.lattice_size
        neurons_per_concept = min(10, lattice_size // max(len(sequence) * 2, 1))

        # Find next available neuron range (avoid overlap with existing concepts)
        used_neurons = set()
        for neurons in self.ai.semantic_bridge.concept_to_neurons.values():
            used_neurons.update(neurons)

        next_start = max(used_neurons) + 1 if used_neurons else 0

        for i, concept in enumerate(sequence):
            if concept not in self.ai.hippocampus.item_memory:
                # Allocate new neuron range for this concept
                start = (next_start + i * neurons_per_concept) % lattice_size
                indices = [(start + j) % lattice_size for j in range(neurons_per_concept)]
                self.ai.register_concept(concept, category='state', neuron_indices=indices)

        self.ai.train_sequence(sequence, repetitions=20)
        return True
