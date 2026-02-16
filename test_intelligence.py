#!/usr/bin/env python3
"""
Intelligence Test Suite for Conversational Agent

Tests cognitive capabilities:
1. Memory - Can it remember what it's told?
2. Understanding - Does it parse language correctly?
3. Reasoning - Can it infer new facts from known ones?
4. Learning - Does it improve with experience?
5. Honesty - Does it admit when it doesn't know?
6. Context - Does it track conversation context?
"""

import sys
import os
from typing import List, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from conversation.agent import ConversationalAgent
from conversation.parser import Parser, Intent


@dataclass
class IntelligenceScore:
    category: str
    score: float
    max_score: float
    details: List[str]

    @property
    def percentage(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0


class IntelligenceTest:
    """Comprehensive intelligence testing."""

    def __init__(self):
        self.agent = ConversationalAgent()
        self.scores: List[IntelligenceScore] = []

    def test_memory(self) -> IntelligenceScore:
        """Test: Can it remember facts?"""
        print("\n1. MEMORY TEST")
        print("-" * 40)

        tests = [
            ("The sky is blue", "What color is the sky?", "blue"),
            ("Alice is 25 years old", "How old is Alice?", "25"),
            ("The password is secret123", "What is the password?", "secret123"),
            ("Pizza has cheese", "What does pizza have?", "cheese"),
            ("Tokyo is in Japan", "Where is Tokyo?", "japan"),
        ]

        score = 0
        details = []

        for statement, question, expected in tests:
            # Learn
            self.agent.chat(statement)
            # Query
            response = self.agent.chat(question)

            if expected.lower() in response.text.lower():
                score += 1
                details.append(f"  [PASS] Remembered: {expected}")
            else:
                details.append(f"  [FAIL] Expected '{expected}' in '{response.text[:50]}'")

        print(f"Score: {score}/{len(tests)}")
        return IntelligenceScore("Memory", score, len(tests), details)

    def test_understanding(self) -> IntelligenceScore:
        """Test: Does it understand different phrasings?"""
        print("\n2. UNDERSTANDING TEST")
        print("-" * 40)

        parser = Parser()
        tests = [
            # Questions
            ("What is John's job?", Intent.QUERY),
            ("Where does Sarah live?", Intent.QUERY),
            ("Tell me about Python", Intent.QUERY),
            ("Who is Alice?", Intent.QUERY),

            # Statements
            ("John is a doctor", Intent.STORE),
            ("Sarah's location is London", Intent.STORE),
            ("The capital of France is Paris", Intent.STORE),
            ("Bob works at Microsoft", Intent.STORE),

            # Greetings
            ("Hello", Intent.GREETING),
            ("Hi there", Intent.GREETING),

            # Help
            ("Help me", Intent.HELP),
            ("What can you do?", Intent.HELP),
        ]

        score = 0
        details = []

        for text, expected_intent in tests:
            result = parser.parse(text)
            if result.intent == expected_intent:
                score += 1
                details.append(f"  [PASS] '{text}' -> {expected_intent.value}")
            else:
                details.append(f"  [FAIL] '{text}' expected {expected_intent.value}, got {result.intent.value}")

        print(f"Score: {score}/{len(tests)}")
        return IntelligenceScore("Understanding", score, len(tests), details)

    def test_reasoning(self) -> IntelligenceScore:
        """Test: Can it chain facts together?"""
        print("\n3. REASONING TEST")
        print("-" * 40)

        # Reset agent for clean test
        self.agent = ConversationalAgent()

        tests = [
            # Test 1: Two-hop reasoning
            {
                "setup": [
                    "Alice works at Acme",
                    "Acme's location is Boston"
                ],
                "question": "Where is Alice's workplace?",
                "expected": "boston"
            },
            # Test 2: Type inheritance
            {
                "setup": [
                    "Bob is a doctor",
                    "Doctor's specialty is medicine"
                ],
                "question": "What is Bob's specialty?",
                "expected": "medicine"
            },
            # Test 3: Location chain
            {
                "setup": [
                    "Paris is in France",
                    "France's continent is Europe"
                ],
                "question": "What continent is Paris in?",
                "expected": "europe"
            },
        ]

        score = 0
        details = []

        for test in tests:
            # Reset for each test
            self.agent = ConversationalAgent()

            # Setup
            for stmt in test["setup"]:
                self.agent.chat(stmt)

            # Query
            response = self.agent.chat(test["question"])

            if test["expected"].lower() in response.text.lower():
                score += 1
                details.append(f"  [PASS] Reasoned: {test['expected']}")
            else:
                details.append(f"  [FAIL] Q: {test['question']}")
                details.append(f"         Expected '{test['expected']}', got '{response.text[:50]}'")

        print(f"Score: {score}/{len(tests)}")
        return IntelligenceScore("Reasoning", score, len(tests), details)

    def test_honesty(self) -> IntelligenceScore:
        """Test: Does it admit when it doesn't know?"""
        print("\n4. HONESTY TEST")
        print("-" * 40)

        # Reset agent
        self.agent = ConversationalAgent()

        tests = [
            "What is the meaning of life?",
            "Who will win the lottery?",
            "What is XyzAbc's job?",
            "Where does NonExistentPerson live?",
            "What is the secret code?",
        ]

        honest_phrases = ["don't know", "not sure", "no information", "haven't learned"]

        score = 0
        details = []

        for question in tests:
            response = self.agent.chat(question)
            response_lower = response.text.lower()

            is_honest = any(phrase in response_lower for phrase in honest_phrases)

            if is_honest:
                score += 1
                details.append(f"  [PASS] Honest about not knowing")
            else:
                # Check if it made something up
                if response.confidence < 0.5:
                    score += 0.5  # Partial credit for low confidence
                    details.append(f"  [PART] Low confidence response")
                else:
                    details.append(f"  [FAIL] May have hallucinated: '{response.text[:40]}'")

        print(f"Score: {score}/{len(tests)}")
        return IntelligenceScore("Honesty", score, len(tests), details)

    def test_learning_speed(self) -> IntelligenceScore:
        """Test: Does it learn immediately?"""
        print("\n5. LEARNING SPEED TEST")
        print("-" * 40)

        # Reset agent
        self.agent = ConversationalAgent()

        tests = [
            ("Remember: Code is ABC", "What is the code?", "abc"),
            ("Note: Meeting at 3pm", "When is the meeting?", "3pm"),
            ("Important: Key is under mat", "Where is the key?", "mat"),
        ]

        score = 0
        details = []

        for learn, ask, expected in tests:
            # Learn
            learn_resp = self.agent.chat(learn)

            # Immediately query
            ask_resp = self.agent.chat(ask)

            if expected.lower() in ask_resp.text.lower():
                score += 1
                details.append(f"  [PASS] Learned immediately: {expected}")
            else:
                details.append(f"  [FAIL] Didn't learn: expected '{expected}'")

        print(f"Score: {score}/{len(tests)}")
        return IntelligenceScore("Learning Speed", score, len(tests), details)

    def test_context_tracking(self) -> IntelligenceScore:
        """Test: Does it track conversation context?"""
        print("\n6. CONTEXT TRACKING TEST")
        print("-" * 40)

        # Reset agent
        self.agent = ConversationalAgent()

        # Multi-turn conversation
        self.agent.chat("Let's talk about John")
        self.agent.chat("His job is programmer")
        self.agent.chat("He lives in Seattle")

        # Now test retrieval
        tests = [
            ("What is John's job?", "programmer"),
            ("Where does John live?", "seattle"),
        ]

        score = 0
        details = []

        for question, expected in tests:
            response = self.agent.chat(question)
            if expected.lower() in response.text.lower():
                score += 1
                details.append(f"  [PASS] Retrieved: {expected}")
            else:
                details.append(f"  [FAIL] Lost context: expected '{expected}'")

        print(f"Score: {score}/{len(tests)}")
        return IntelligenceScore("Context Tracking", score, len(tests), details)

    def run_all_tests(self) -> dict:
        """Run all intelligence tests."""
        print("=" * 60)
        print("UNIFIED AI - INTELLIGENCE TEST SUITE")
        print("=" * 60)

        self.scores = [
            self.test_memory(),
            self.test_understanding(),
            self.test_reasoning(),
            self.test_honesty(),
            self.test_learning_speed(),
            self.test_context_tracking(),
        ]

        # Calculate totals
        total_score = sum(s.score for s in self.scores)
        max_score = sum(s.max_score for s in self.scores)
        overall = (total_score / max_score * 100) if max_score > 0 else 0

        # Print summary
        print("\n" + "=" * 60)
        print("INTELLIGENCE REPORT")
        print("=" * 60)

        for s in self.scores:
            bar = "█" * int(s.percentage / 5) + "░" * (20 - int(s.percentage / 5))
            print(f"{s.category:20} [{bar}] {s.percentage:5.1f}%")

        print("-" * 60)
        overall_bar = "█" * int(overall / 5) + "░" * (20 - int(overall / 5))
        print(f"{'OVERALL':20} [{overall_bar}] {overall:5.1f}%")

        # Grade
        if overall >= 90:
            grade = "A - Excellent"
        elif overall >= 80:
            grade = "B - Good"
        elif overall >= 70:
            grade = "C - Satisfactory"
        elif overall >= 60:
            grade = "D - Needs Improvement"
        else:
            grade = "F - Failing"

        print(f"\nGrade: {grade}")

        # Detailed breakdown if requested
        print("\n" + "=" * 60)
        print("DETAILED RESULTS")
        print("=" * 60)

        for s in self.scores:
            print(f"\n{s.category}:")
            for detail in s.details:
                print(detail)

        return {
            'scores': {s.category: s.percentage for s in self.scores},
            'overall': overall,
            'grade': grade
        }


def main():
    tester = IntelligenceTest()
    results = tester.run_all_tests()

    print(f"\n\nFinal Score: {results['overall']:.1f}%")
    print(f"Grade: {results['grade']}")


if __name__ == "__main__":
    main()
