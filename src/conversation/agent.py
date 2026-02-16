"""
Conversational Agent

The main agent that:
- Understands natural language (parser)
- Reasons about knowledge (reasoner)
- Learns from conversation
- Responds naturally
"""

import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_ai import UnifiedAI, UnifiedAIConfig
from conversation.parser import Parser, Intent, ParseResult
from conversation.reasoner import Reasoner, ReasoningResult


@dataclass
class Response:
    """Agent response."""
    text: str
    learned: bool = False
    confidence: float = 1.0
    action: str = "respond"


class ConversationalAgent:
    """
    An agent that can converse, learn, and reason.

    Design principles:
    - Honest: Says "I don't know" rather than guess
    - Learning: Remembers what it's told
    - Reasoning: Can follow chains of facts
    - Helpful: Explains its reasoning when useful
    """

    def __init__(self, config: UnifiedAIConfig = None):
        self.config = config or UnifiedAIConfig(
            lattice_size=256,
            hv_dimensions=10000,
            similarity_threshold=0.10
        )
        self.ai = UnifiedAI(self.config)
        self.parser = Parser()
        self.reasoner = Reasoner(self.ai)

        # Conversation state
        self.context = {}  # Current topic/entity
        self.last_intent = None
        self.pending_confirmation = None

        # Knowledge sources (for future expansion)
        self.sources: Dict[str, str] = {}

        # Initialize with some base concepts
        self._initialize_base_concepts()

    def _initialize_base_concepts(self):
        """Set up common roles."""
        base_roles = ['type', 'job', 'location', 'workplace', 'status',
                      'name', 'has', 'knows', 'belongs_to', 'capital']
        for role in base_roles:
            self.ai.register_concept(role, category='role')

    def chat(self, user_input: str) -> Response:
        """
        Main conversation handler.

        Takes user input, processes it, updates state, returns response.
        """
        # Parse the input
        parsed = self.parser.parse(user_input)
        self.last_intent = parsed.intent

        # Route to appropriate handler
        handlers = {
            Intent.QUERY: self._handle_query,
            Intent.STORE: self._handle_store,
            Intent.TEACH: self._handle_teach,
            Intent.PREDICT: self._handle_predict,
            Intent.LIST: self._handle_list,
            Intent.GREETING: self._handle_greeting,
            Intent.FAREWELL: self._handle_farewell,
            Intent.HELP: self._handle_help,
            Intent.CONFIRM: self._handle_confirm,
            Intent.DENY: self._handle_deny,
            Intent.CONTEXT: self._handle_context,
            Intent.UNKNOWN: self._handle_unknown,
        }

        handler = handlers.get(parsed.intent, self._handle_unknown)
        return handler(parsed)

    def _handle_query(self, parsed: ParseResult) -> Response:
        """Handle questions."""
        entities = parsed.entities

        # Get entity and role from parsed result
        entity = entities.get('entity')
        role = entities.get('role')

        if not entity:
            return Response(
                text="I'm not sure what you're asking about. Can you be more specific?",
                confidence=0.3
            )

        # Wildcard query: tell me about X
        if role == '*':
            return self._handle_about(entity)

        # Specific query
        if role:
            result = self.reasoner.query(entity, role)

            if result.found:
                # Update context
                self.context['entity'] = entity
                self.context['role'] = role

                if result.source == "direct":
                    return Response(
                        text=f"{entity}'s {role} is {result.value}.",
                        confidence=result.confidence
                    )
                elif result.source == "inferred":
                    chain = " → ".join(result.reasoning_chain)
                    return Response(
                        text=f"{result.value}. (I figured this out: {chain})",
                        confidence=result.confidence
                    )
                else:
                    return Response(
                        text=result.explain(),
                        confidence=result.confidence
                    )

        # Check if we're asking about a type
        expected_type = entities.get('expected')
        if expected_type:
            result = self.reasoner.check_type(entity, expected_type)
            if result.found:
                if result.value:
                    return Response(f"Yes, {entity} is a {expected_type}.")
                else:
                    type_result = self.reasoner.query(entity, "type")
                    if type_result.found:
                        return Response(f"No, {entity} is a {type_result.value}, not a {expected_type}.")
                    return Response(f"I'm not sure if {entity} is a {expected_type}.")

        return Response(
            text=f"I don't know {entity}'s {role}.",
            confidence=0.0
        )

    def _handle_about(self, entity: str) -> Response:
        """Handle 'tell me about X' queries."""
        facts = self.reasoner.query_all(entity)

        if not facts:
            return Response(f"I don't know anything about {entity}.")

        lines = [f"Here's what I know about {entity}:"]
        for role, result in facts.items():
            lines.append(f"  - {role}: {result.value}")

        return Response("\n".join(lines))

    def _handle_store(self, parsed: ParseResult) -> Response:
        """Handle statements (learning facts)."""
        entities = parsed.entities

        # Check for raw statement that needs re-parsing
        if 'raw_statement' in entities:
            reparsed = self.parser.parse(entities['raw_statement'])
            if reparsed.intent == Intent.STORE:
                entities = reparsed.entities

        entity = entities.get('entity')
        role = entities.get('role')
        value = entities.get('value')

        # Handle context-dependent entity (his/her)
        if entity == '_context_':
            entity = self.context.get('entity')
            if not entity:
                return Response("Who are you referring to?", confidence=0.3)

        if entity and role and value:
            success = self.reasoner.store_fact(entity, role, value)

            if success:
                self.context['entity'] = entity
                return Response(
                    text=f"Got it. {entity}'s {role} is {value}.",
                    learned=True
                )
            else:
                return Response(
                    text="I had trouble remembering that. Can you rephrase?",
                    confidence=0.3
                )

        # Partial parse - ask for clarification
        if entity and not value:
            self.pending_confirmation = {
                'type': 'incomplete_fact',
                'entity': entity,
                'role': role
            }
            return Response(f"What is {entity}'s {role}?")

        return Response(
            text="I'm not sure what fact you want me to remember. Try: 'Alice's job is Engineer'",
            confidence=0.3
        )

    def _handle_teach(self, parsed: ParseResult) -> Response:
        """Handle teaching about knowledge sources."""
        entities = parsed.entities
        topic = entities.get('topic')
        source = entities.get('source')

        if topic and source:
            self.sources[topic.lower()] = source
            return Response(
                text=f"Got it. For {topic} questions, I'll check {source}.",
                learned=True
            )

        return Response("I didn't understand that source. Try: 'for weather, check wttr.in'")

    def _handle_predict(self, parsed: ParseResult) -> Response:
        """Handle prediction queries."""
        context = parsed.entities.get('context', '')

        if not context:
            return Response("What should I predict from? Give me a starting point.")

        result = self.reasoner.predict_next(context)

        if result.found:
            return Response(
                text=f"After {context}, probably {result.value}.",
                confidence=result.confidence
            )
        else:
            return Response(f"I don't have a sequence starting with {context}. Teach me one!")

    def _handle_list(self, parsed: ParseResult) -> Response:
        """Handle list/show queries."""
        concepts = len(self.ai.hippocampus.item_memory)
        facts = len(self.ai.fact_memory.state_vectors)
        sources = len(self.sources)

        lines = [
            f"I know {concepts} concepts and {facts} facts.",
        ]

        if sources:
            lines.append(f"I know how to find info on: {', '.join(self.sources.keys())}")

        if self.context.get('entity'):
            lines.append(f"We were talking about: {self.context['entity']}")

        return Response("\n".join(lines))

    def _handle_greeting(self, parsed: ParseResult) -> Response:
        """Handle greetings."""
        return Response("Hello! I'm here to learn and help. Tell me things or ask me questions.")

    def _handle_farewell(self, parsed: ParseResult) -> Response:
        """Handle goodbyes."""
        return Response("Goodbye! I'll remember what we discussed.", action="quit")

    def _handle_help(self, parsed: ParseResult) -> Response:
        """Handle help requests."""
        return Response("""I can:
- Learn facts: "Alice's job is Engineer"
- Answer questions: "What is Alice's job?"
- Learn sequences: "The order is Wake, Eat, Work, Sleep"
- Predict: "What comes after Wake?"
- Show knowledge: "What do you know?"

I'll be honest when I don't know something.""")

    def _handle_confirm(self, parsed: ParseResult) -> Response:
        """Handle confirmation (yes/ok)."""
        if self.pending_confirmation:
            # Handle pending action
            pending = self.pending_confirmation
            self.pending_confirmation = None
            return Response("What would you like me to remember?")

        return Response("OK!")

    def _handle_deny(self, parsed: ParseResult) -> Response:
        """Handle denial (no/wrong)."""
        self.pending_confirmation = None
        return Response("Alright, never mind that then.")

    def _handle_context(self, parsed: ParseResult) -> Response:
        """Handle context setting (let's talk about X)."""
        entity = parsed.entities.get('context_entity', '')
        if entity:
            self.context['entity'] = entity.lower()
            return Response(f"OK, let's talk about {entity}. What would you like to know or tell me?")

    def _handle_unknown(self, parsed: ParseResult) -> Response:
        """Handle unknown input."""
        # Check if it might be a sequence definition
        words = parsed.original.replace(',', ' ').split()
        if len(words) >= 3 and all(w[0].isupper() for w in words if w.isalpha()):
            # Looks like a sequence
            self.pending_confirmation = {
                'type': 'sequence',
                'items': words
            }
            return Response(
                f"Should I learn this as a sequence? {' → '.join(words)}"
            )

        return Response(
            "I didn't quite understand that. Try asking a question or telling me a fact.",
            confidence=0.2
        )

    def save(self, path: str):
        """Save agent state."""
        import json

        self.ai.save(path)

        # Save sources and context
        meta = {
            'sources': self.sources,
            'context': self.context
        }
        with open(os.path.join(path, 'agent_meta.json'), 'w') as f:
            json.dump(meta, f)

    def load(self, path: str):
        """Load agent state."""
        import json

        self.ai.load(path)

        meta_path = os.path.join(path, 'agent_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                self.sources = meta.get('sources', {})
                self.context = meta.get('context', {})
