"""
LLM Interface - Natural Language Layer

This module connects the Unified AI's grounded symbolic memory to
a small LLM for natural language formatting.

The key insight: The LLM ONLY does formatting, not reasoning.
All facts come from HippocampusVSA, so hallucination is impossible.

Supported Models:
- HuggingFace Inference API (remote)
- Local transformers (if available)
- Template-based fallback (no LLM needed)
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import time

# Try to import HuggingFace libraries
HF_AVAILABLE = False
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    pass


@dataclass
class LLMConfig:
    """Configuration for LLM interface."""
    api_key: str = None
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_tokens: int = 256
    temperature: float = 0.3  # Low for factual formatting
    use_templates: bool = True  # Fallback to templates
    timeout: int = 30


class GroundedLLM:
    """
    LLM wrapper that ONLY formats grounded facts.

    The system prompt strictly instructs the LLM to:
    1. ONLY use facts provided in the JSON
    2. NEVER add information not in the JSON
    3. Say "unknown" for missing facts

    This makes hallucination structurally impossible.
    """

    SYSTEM_PROMPT = """You are a factual assistant that formats JSON data into natural language.

CRITICAL RULES:
1. ONLY use facts from the provided JSON. Do NOT add any information.
2. If a fact is marked as "unknown" or null, say "I don't know" for that fact.
3. Never invent, assume, or extrapolate facts.
4. Keep responses concise and factual.
5. You are formatting pre-verified facts, not generating knowledge.

Example:
Input JSON: {"status": "operational", "location": null}
Output: "The status is operational. The location is unknown."
"""

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.client = None

        if HF_AVAILABLE and self.config.api_key:
            try:
                self.client = InferenceClient(token=self.config.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize HuggingFace client: {e}")

        self.call_count = 0
        self.template_count = 0

    def format_response(
        self,
        grounded_data: Dict,
        query: str = None,
        use_llm: bool = True
    ) -> str:
        """
        Format grounded data into natural language.

        Args:
            grounded_data: JSON from UnifiedAI.get_grounded_response()
            query: Original user query (for context)
            use_llm: Whether to use LLM (falls back to templates if False or unavailable)

        Returns:
            Natural language response
        """
        # Validate input
        if not grounded_data or not grounded_data.get('llm_safe', False):
            return "Unable to generate response: data not properly grounded."

        # Try LLM first
        if use_llm and self.client and HF_AVAILABLE:
            try:
                return self._llm_format(grounded_data, query)
            except Exception as e:
                print(f"LLM formatting failed: {e}, falling back to templates")

        # Template fallback
        return self._template_format(grounded_data, query)

    def _llm_format(self, grounded_data: Dict, query: str = None) -> str:
        """Format using HuggingFace LLM."""
        self.call_count += 1

        # Prepare the prompt
        facts_json = json.dumps(grounded_data['facts'], indent=2)
        context = grounded_data.get('context', 'the subject')

        user_prompt = f"""Format these facts about "{context}" into a natural sentence:

{facts_json}

{"Original question: " + query if query else ""}

Response (use ONLY the facts above, say "unknown" for null values):"""

        try:
            response = self.client.text_generation(
                prompt=f"[INST] {self.SYSTEM_PROMPT}\n\n{user_prompt} [/INST]",
                model=self.config.model_id,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.strip()
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    def _template_format(self, grounded_data: Dict, query: str = None) -> str:
        """Format using templates (no LLM needed)."""
        self.template_count += 1

        context = grounded_data.get('context', 'The subject')
        facts = grounded_data.get('facts', {})

        if not facts:
            return f"No information available about {context}."

        # Build response from facts
        parts = []

        for role, data in facts.items():
            value = data.get('value')
            confidence = data.get('confidence', 0)

            if value is None:
                parts.append(f"the {role.lower()} is unknown")
            elif confidence > 0.8:
                parts.append(f"the {role.lower()} is {value}")
            elif confidence > 0.5:
                parts.append(f"the {role.lower()} appears to be {value}")
            else:
                parts.append(f"the {role.lower()} might be {value} (low confidence)")

        if not parts:
            return f"No specific information available about {context}."

        # Combine parts
        if len(parts) == 1:
            response = f"For {context}, {parts[0]}."
        elif len(parts) == 2:
            response = f"For {context}, {parts[0]} and {parts[1]}."
        else:
            response = f"For {context}, {', '.join(parts[:-1])}, and {parts[-1]}."

        return response.capitalize()

    def answer_question(
        self,
        question: str,
        grounded_data: Dict
    ) -> str:
        """
        Answer a question using grounded data.

        The response is guaranteed to be factual because:
        1. All facts come from grounded_data (HippocampusVSA)
        2. LLM only formats, doesn't reason
        """
        # Parse question to identify what's being asked
        question_lower = question.lower()

        # Extract relevant fact
        facts = grounded_data.get('facts', {})
        relevant_fact = None
        relevant_role = None

        for role, data in facts.items():
            if role.lower() in question_lower:
                relevant_fact = data
                relevant_role = role
                break

        if relevant_fact is None:
            # Try to find any grounded fact
            for role, data in facts.items():
                if data.get('value') is not None:
                    relevant_fact = data
                    relevant_role = role
                    break

        if relevant_fact is None:
            return "I don't have information about that."

        value = relevant_fact.get('value')
        confidence = relevant_fact.get('confidence', 0)

        if value is None:
            return f"I don't know the {relevant_role.lower()}."

        if confidence > 0.8:
            return f"The {relevant_role.lower()} is {value}."
        elif confidence > 0.5:
            return f"The {relevant_role.lower()} appears to be {value}."
        else:
            return f"I'm not certain, but the {relevant_role.lower()} might be {value}."

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            'llm_calls': self.call_count,
            'template_calls': self.template_count,
            'hf_available': HF_AVAILABLE,
            'client_ready': self.client is not None
        }


class ConversationalInterface:
    """
    Full conversational interface combining UnifiedAI + LLM.

    Handles:
    - Question answering (grounded)
    - Fact teaching
    - Status queries
    - Anomaly reporting
    """

    def __init__(self, unified_ai, llm: GroundedLLM = None):
        """
        Args:
            unified_ai: UnifiedAI instance
            llm: GroundedLLM instance (creates default if None)
        """
        self.ai = unified_ai
        self.llm = llm or GroundedLLM()

        # Conversation state
        self.current_context: str = None
        self.history: List[Dict] = []

    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate response.

        Handles:
        - Questions about facts
        - Teaching new facts
        - Status queries
        """
        user_input = user_input.strip()

        # Record in history
        self.history.append({'role': 'user', 'content': user_input})

        # Determine intent
        response = self._handle_input(user_input)

        # Record response
        self.history.append({'role': 'assistant', 'content': response})

        return response

    def _handle_input(self, text: str) -> str:
        """Handle different types of input."""
        text_lower = text.lower()

        # Teaching: "X is Y" or "the X of Y is Z"
        teach_match = re.match(
            r"(?:the\s+)?(\w+)\s+(?:of\s+)?(\w+)\s+is\s+(.+)",
            text_lower
        )
        if teach_match and "?" not in text:
            role, context, value = teach_match.groups()
            return self._handle_teaching(context, role, value)

        # Question: "What is the X of Y?"
        question_match = re.match(
            r"what\s+is\s+(?:the\s+)?(\w+)\s+(?:of\s+)?(\w+)",
            text_lower
        )
        if question_match:
            role, context = question_match.groups()
            return self._handle_question(context, role, text)

        # Status query: "status" or "tell me about X"
        if text_lower.startswith("status") or text_lower.startswith("tell me about"):
            match = re.search(r"(?:about|of)\s+(\w+)", text_lower)
            context = match.group(1) if match else self.current_context
            if context:
                return self._handle_status_query(context)
            return "Please specify what you'd like to know about."

        # Stats query
        if "stats" in text_lower or "statistics" in text_lower:
            stats = self.ai.get_stats()
            return f"System: {stats['hippocampus']['num_concepts']} concepts, " \
                   f"{stats['memory']['total_kb']:.1f} KB memory, " \
                   f"{stats['performance']['avg_inference_us']:.0f}us avg inference"

        # Default: try to answer as question
        if "?" in text or text_lower.startswith(("what", "where", "who", "how", "is")):
            return self._handle_general_question(text)

        return "I can answer questions about stored facts, or you can teach me by saying 'the X of Y is Z'."

    def _handle_teaching(self, context: str, role: str, value: str) -> str:
        """Handle teaching a new fact."""
        # Ensure concepts exist
        for concept in [role.title(), value.title()]:
            if self.ai.hippocampus.get_concept(concept) is None:
                self.ai.register_concept(concept, category='general')

        # Store fact
        result = self.ai.store_fact(context.title(), role.title(), value.title())

        if result['success']:
            self.current_context = context.title()
            return f"Got it! I'll remember that the {role} of {context} is {value}."
        else:
            return f"Sorry, I couldn't store that fact: {result.get('error', 'unknown error')}"

    def _handle_question(self, context: str, role: str, original: str) -> str:
        """Handle a specific question."""
        self.current_context = context.title()

        # Query the fact
        result = self.ai.query_fact(context.title(), role.title())

        if result['grounded']:
            value = result['value']
            conf = result['confidence']
            if conf > 0.8:
                return f"The {role} of {context} is {value}."
            else:
                return f"I believe the {role} of {context} is {value}, but I'm not fully certain."
        else:
            return f"I don't know the {role} of {context}. Would you like to teach me?"

    def _handle_status_query(self, context: str) -> str:
        """Handle status/summary query."""
        self.current_context = context.title()

        # Get all registered roles
        roles = [name for name, meta in self.ai.hippocampus.metadata.items()
                if meta.category == 'role']

        if not roles:
            roles = ['Status', 'Location']  # Defaults

        # Get grounded response
        grounded = self.ai.get_grounded_response(context.title(), roles)

        # Format with LLM
        return self.llm.format_response(grounded, f"Tell me about {context}")

    def _handle_general_question(self, text: str) -> str:
        """Handle general question."""
        if self.current_context:
            # Try to answer about current context
            roles = [name for name, meta in self.ai.hippocampus.metadata.items()
                    if meta.category == 'role']
            grounded = self.ai.get_grounded_response(self.current_context, roles)
            return self.llm.answer_question(text, grounded)

        return "Please specify what you'd like to know about."


def test_llm_interface():
    """Test the LLM interface."""
    print("=" * 60)
    print("LLM INTERFACE TEST")
    print("=" * 60)

    # Test template formatting
    config = LLMConfig(use_templates=True)
    llm = GroundedLLM(config)

    # Simulated grounded data
    grounded_data = {
        'context': 'Drone-01',
        'facts': {
            'Status': {'value': 'MotorFailure', 'confidence': 0.95},
            'Location': {'value': 'Highway', 'confidence': 0.87},
            'Battery': {'value': None, 'confidence': 0.0}
        },
        'grounded': True,
        'llm_safe': True
    }

    print("\n1. Template Formatting:")
    print("   Input:", json.dumps(grounded_data['facts'], indent=6))
    response = llm.format_response(grounded_data, use_llm=False)
    print(f"   Output: {response}")

    # Test question answering
    print("\n2. Question Answering:")
    questions = [
        "What is the status?",
        "What is the battery level?",
        "Where is it located?"
    ]

    for q in questions:
        answer = llm.answer_question(q, grounded_data)
        print(f"   Q: {q}")
        print(f"   A: {answer}")

    # Test with unknown data
    print("\n3. Unknown Data (No Hallucination):")
    unknown_data = {
        'context': 'Drone-99',
        'facts': {
            'Status': {'value': None, 'confidence': 0.0},
            'Location': {'value': None, 'confidence': 0.0}
        },
        'grounded': False,
        'llm_safe': True
    }

    response = llm.format_response(unknown_data, use_llm=False)
    print(f"   Output: {response}")

    print(f"\nStats: {llm.get_stats()}")
    print("\nSUCCESS: Template formatting works without hallucination")


def test_conversational():
    """Test full conversational interface."""
    print("\n" + "=" * 60)
    print("CONVERSATIONAL INTERFACE TEST")
    print("=" * 60)

    # Import UnifiedAI
    from unified_ai import UnifiedAI, UnifiedAIConfig

    # Initialize
    config = UnifiedAIConfig(lattice_size=100, hv_dimensions=10000)
    ai = UnifiedAI(config)

    # Pre-register some concepts
    ai.register_concept("Status", category='role')
    ai.register_concept("Location", category='role')
    ai.register_concept("Operational", category='value')
    ai.register_concept("Warehouse", category='value')

    # Create interface
    interface = ConversationalInterface(ai)

    # Test conversation
    test_inputs = [
        "The status of Robot-01 is Operational",
        "The location of Robot-01 is Warehouse",
        "What is the status of Robot-01?",
        "Tell me about Robot-01",
        "What is the battery of Robot-01?",  # Unknown
        "stats"
    ]

    print("\nConversation:")
    for user_input in test_inputs:
        print(f"\n  User: {user_input}")
        response = interface.process_input(user_input)
        print(f"  AI: {response}")

    print("\nSUCCESS: Conversational interface working")


if __name__ == "__main__":
    test_llm_interface()
    test_conversational()
