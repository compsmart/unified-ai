#!/usr/bin/env python3
"""
Automated Training & Testing for Conversational Agent

This script:
1. Loads training data (facts, sequences)
2. Trains the agent
3. Runs intelligence tests
4. Reports results
5. Saves the trained model
"""

import sys
import os
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from conversation.agent import ConversationalAgent
from conversation.reasoner import ReasoningResult


@dataclass
class TestResult:
    name: str
    passed: bool
    expected: str
    actual: str
    details: str = ""


class AgentTrainer:
    """Trains and tests the conversational agent."""

    def __init__(self, data_path: str = None):
        self.data_path = data_path or os.path.join(
            os.path.dirname(__file__), 'data', 'training_data.json'
        )
        self.agent = ConversationalAgent()
        self.results: List[TestResult] = []

    def load_training_data(self) -> Dict:
        """Load training data from JSON."""
        with open(self.data_path) as f:
            return json.load(f)

    def train_facts(self, facts: List[Dict], verbose: bool = True) -> int:
        """Train agent on facts."""
        count = 0
        for fact in facts:
            success = self.agent.reasoner.store_fact(
                fact['entity'],
                fact['role'],
                fact['value']
            )
            if success:
                count += 1
                if verbose:
                    print(f"  + {fact['entity']}.{fact['role']} = {fact['value']}")
        return count

    def train_sequences(self, sequences: List[List[str]], verbose: bool = True) -> int:
        """Train agent on sequences."""
        count = 0
        for seq in sequences:
            self.agent.reasoner.learn_sequence(seq)
            count += 1
            if verbose:
                print(f"  + {' -> '.join(seq)}")
        return count

    def test_direct_queries(self, tests: List[Dict], verbose: bool = True) -> List[TestResult]:
        """Test direct fact retrieval."""
        results = []

        for test in tests:
            response = self.agent.chat(test['input'])
            response_lower = response.text.lower()

            passed = False
            expected = ""

            if 'expected_contains' in test:
                expected = test['expected_contains']
                passed = expected.lower() in response_lower

            if 'expected_entity' in test:
                expected = test['expected_entity']
                passed = expected.lower() in response_lower

            result = TestResult(
                name=f"Query: {test['input']}",
                passed=passed,
                expected=expected,
                actual=response.text[:50]
            )
            results.append(result)

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {test['input']}")
                if not passed:
                    print(f"       Expected: {expected}")
                    print(f"       Got: {response.text[:80]}")

        return results

    def test_reasoning(self, tests: List[Dict], verbose: bool = True) -> List[TestResult]:
        """Test transitive reasoning."""
        results = []

        for test in tests:
            # Setup facts for this test
            for fact in test['setup']:
                self.agent.reasoner.store_fact(
                    fact['entity'],
                    fact['role'],
                    fact['value']
                )

            # Query
            query = test['query']
            result = self.agent.reasoner.query(
                query['entity'],
                query['role'],
                follow_chain=True
            )

            passed = result.found and result.value.lower() == test['expected'].lower()

            test_result = TestResult(
                name=test['description'],
                passed=passed,
                expected=test['expected'],
                actual=result.value if result.found else "NOT FOUND",
                details=str(result.reasoning_chain) if result.found else ""
            )
            results.append(test_result)

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {test['description']}")
                if not passed:
                    print(f"       Expected: {test['expected']}")
                    print(f"       Got: {result.value}")

        return results

    def test_conversation(self, tests: List[Dict], verbose: bool = True) -> List[TestResult]:
        """Test multi-turn conversation."""
        results = []

        for test in tests:
            dialogue = test['dialogue']

            for turn in dialogue:
                response = self.agent.chat(turn['user'])

                passed = True
                expected = ""

                if turn.get('expect_learned'):
                    passed = response.learned
                    expected = "should learn"

                if 'expect_contains' in turn:
                    expected = turn['expect_contains']
                    passed = expected.lower() in response.text.lower()

                result = TestResult(
                    name=f"Conv: {turn['user'][:30]}...",
                    passed=passed,
                    expected=expected,
                    actual=response.text[:50]
                )
                results.append(result)

                if verbose:
                    status = "PASS" if passed else "FAIL"
                    print(f"  [{status}] User: {turn['user'][:40]}")
                    if not passed:
                        print(f"       Expected: {expected}")
                        print(f"       Got: {response.text[:60]}")

        return results

    def test_predictions(self, sequences: List[List[str]], verbose: bool = True) -> List[TestResult]:
        """Test sequence prediction."""
        results = []

        for seq in sequences:
            if len(seq) < 2:
                continue

            # Test prediction from first element
            first = seq[0]
            expected_next = seq[1]

            result = self.agent.reasoner.predict_next(first)

            # Compare case-insensitively since we normalize to lowercase
            passed = result.found and result.value.lower() == expected_next.lower()

            test_result = TestResult(
                name=f"Predict: {first} -> ?",
                passed=passed,
                expected=expected_next,
                actual=result.value if result.found else "NONE"
            )
            results.append(test_result)

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] After '{first}' expect '{expected_next}'")
                if not passed:
                    print(f"       Got: {result.value}")

        return results

    def run_full_training(self, verbose: bool = True) -> Dict:
        """Run complete training and testing cycle."""
        print("=" * 60)
        print("UNIFIED AI - AUTOMATED TRAINING")
        print("=" * 60)
        print()

        data = self.load_training_data()
        start_time = time.time()

        # Phase 1: Train Facts
        print("PHASE 1: Learning Facts")
        print("-" * 40)
        facts_learned = self.train_facts(data.get('facts', []), verbose)
        print(f"\nLearned {facts_learned} facts")
        print()

        # Phase 2: Train Sequences
        print("PHASE 2: Learning Sequences")
        print("-" * 40)
        seqs_learned = self.train_sequences(data.get('sequences', []), verbose)
        print(f"\nLearned {seqs_learned} sequences")
        print()

        # Phase 3: Test Direct Queries
        print("PHASE 3: Testing Direct Queries")
        print("-" * 40)
        query_results = self.test_direct_queries(data.get('test_queries', []), verbose)
        self.results.extend(query_results)
        print()

        # Phase 4: Test Reasoning
        print("PHASE 4: Testing Reasoning")
        print("-" * 40)
        reasoning_results = self.test_reasoning(data.get('reasoning_tests', []), verbose)
        self.results.extend(reasoning_results)
        print()

        # Phase 5: Test Conversation
        print("PHASE 5: Testing Conversation")
        print("-" * 40)
        conv_results = self.test_conversation(data.get('conversation_tests', []), verbose)
        self.results.extend(conv_results)
        print()

        # Phase 6: Test Predictions
        print("PHASE 6: Testing Predictions")
        print("-" * 40)
        pred_results = self.test_predictions(data.get('sequences', []), verbose)
        self.results.extend(pred_results)
        print()

        # Summary
        total_time = time.time() - start_time
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Facts learned:     {facts_learned}")
        print(f"Sequences learned: {seqs_learned}")
        print(f"Tests passed:      {passed}/{total} ({100*passed/total:.1f}%)")
        print(f"Training time:     {total_time:.2f}s")
        print()

        # Save model
        save_path = os.path.join(os.path.dirname(__file__), 'models', 'trained_agent')
        os.makedirs(save_path, exist_ok=True)
        self.agent.save(save_path)
        print(f"Model saved to: {save_path}")

        return {
            'facts_learned': facts_learned,
            'sequences_learned': seqs_learned,
            'tests_passed': passed,
            'tests_total': total,
            'accuracy': passed / total if total > 0 else 0,
            'time_seconds': total_time,
            'save_path': save_path
        }


def main():
    trainer = AgentTrainer()
    results = trainer.run_full_training(verbose=True)

    # Exit with error code if tests failed
    if results['accuracy'] < 1.0:
        print(f"\nWARNING: {results['tests_total'] - results['tests_passed']} tests failed")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
