"""
Natural Language Parser

Extracts intent and entities from user input.
Designed to be simple but extensible.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Intent(Enum):
    QUERY = "query"           # Asking for information
    STORE = "store"           # Telling a fact
    TEACH = "teach"           # Teaching how to find info
    PREDICT = "predict"       # What happens next?
    LIST = "list"             # Show what you know
    GREETING = "greeting"     # Hello/Hi
    FAREWELL = "farewell"     # Goodbye
    HELP = "help"             # How to use
    CONFIRM = "confirm"       # Yes/OK
    DENY = "deny"             # No/Wrong
    CONTEXT = "context"       # Setting conversation context
    UNKNOWN = "unknown"


@dataclass
class ParseResult:
    intent: Intent
    entities: Dict[str, str]
    confidence: float
    original: str

    def __repr__(self):
        return f"ParseResult({self.intent.value}, {self.entities}, conf={self.confidence:.2f})"


class Parser:
    """
    Rule-based parser with pattern matching.

    Falls back gracefully when patterns don't match.
    """

    def __init__(self):
        # Question patterns
        self.question_patterns = [
            # What is my X? (must be first to catch "my")
            (r"what is my (\w+)\??", self._extract_my_query),
            # What is X's Y?
            (r"what(?:'s| is) (\w+)'s (\w+)\??", self._extract_what_query),
            # What Y does X have?
            (r"what (\w+) does (\w+) have\??", self._extract_what_have),
            # What is the Y of X?
            (r"what is the (\w+) of (\w+)\??", self._extract_the_of),
            # What type/kind is X?
            (r"what (?:type|kind) is (\w+)\??", self._extract_type_query),
            # What does X require/need?
            (r"what does (\w+) (?:require|need)\??", self._extract_require),
            # What color/value is X?
            (r"what (\w+) is (?:the )?(\w+)\??", self._extract_what_attr),
            # What is X? (simple type query)
            (r"what is (\w+)\??$", self._extract_what_is),
            # What is the X? (e.g., "What is the code?")
            (r"what is the (\w+)\??$", self._extract_what_the),
            # How old is X?
            (r"how old is (\w+)\??", self._extract_how_old),
            # What does X have?
            (r"what does (\w+) have\??", self._extract_what_has),
            # Where is/does X?
            (r"where (?:is|does) (\w+)(?: work| live| stay)?\??", self._extract_where),
            # When is X? (time query)
            (r"when is (?:the )?(\w+)\??", self._extract_when),
            # Who is X?
            (r"who is (\w+)\??", self._extract_who),
            # Does X have/know Y?
            (r"does (\w+) (?:have|know) (?:a |an )?(\w+)\??", self._extract_does_have),
            # Is X a Y? (require "a" or "an")
            (r"is (\w+) (?:a|an) (\w+)\??", self._extract_is_a),
            # Tell me about X
            (r"(?:tell me about|describe|explain) (\w+)", self._extract_about),
            # What do you know about X?
            (r"what do you know about (\w+)\??", self._extract_about),
            # Generic what/where/who/how
            (r"(what|where|who|how|why|when) .+\?", self._extract_generic_question),
        ]

        # Statement patterns (facts to learn)
        # IMPORTANT: More specific patterns must come FIRST
        self.statement_patterns = [
            # X is N years old (age)
            (r"(\w+) is (\d+) years? old", self._extract_age),
            # His/Her X is Y (context-dependent)
            (r"(?:his|her|their) (\w+) is (\w+)", self._extract_context_statement),
            # He/She lives in Y (context-dependent)
            (r"(?:he|she|they) lives? (?:in|at) (\w+)", self._extract_context_lives),
            # My X is Y (e.g., "My name is John")
            (r"my (\w+) is (\w+)", self._extract_my_statement),
            # Remember: X is Y
            (r"remember:? (\w+) is (\w+)", self._extract_remember_fact),
            # Note: X is/at Y
            (r"note:? (\w+) (?:is|at) (\w+)", self._extract_note_fact),
            # Important: X is Y
            (r"important:? (\w+) (?:is|under|at) (\w+)", self._extract_note_location),
            # X's Y is Z (most specific, must be first)
            (r"(\w+)'s (\w+) is (\w+)", self._extract_possessive),
            # The Y of X is Z
            (r"the (\w+) of (\w+) is (\w+)", self._extract_of_statement),
            # The X is Y (e.g., "The sky is blue")
            (r"the (\w+) is (\w+)", self._extract_the_is),
            # X works at/in/for Y
            (r"(\w+) works (?:at|in|for) (\w+)", self._extract_works_at),
            # X lives in Y
            (r"(\w+) lives (?:in|at) (\w+)", self._extract_lives_in),
            # X has Y
            (r"(\w+) has (?:a |an )?(.+)", self._extract_has),
            # X knows Y
            (r"(\w+) knows (?:about )?(\w+)", self._extract_knows),
            # Remember that X
            (r"remember (?:that )?(.+)", self._extract_remember),
            # X is under/on/near Y (location)
            (r"(\w+) is (?:under|on|near|behind|beside) (?:the )?(\w+)", self._extract_is_under),
            # X is in/at Y (location)
            (r"(\w+) is (?:in|at) (\w+)", self._extract_is_location),
            # X is Y / X is a Y (least specific, must be last)
            (r"^(\w+) is (?:a |an )?(\w+)$", self._extract_is_statement),
        ]

        # Teaching patterns (how to find info)
        self.teach_patterns = [
            # For X, check Y / use Y
            (r"for (\w+),? (?:check|use|ask|query) (.+)", self._extract_source),
            # To find X, use Y
            (r"to (?:find|get|look up) (\w+),? (?:check|use) (.+)", self._extract_source),
            # X can be found at Y
            (r"(\w+) can be found (?:at|in|on|via) (.+)", self._extract_source),
        ]

        # Prediction patterns
        self.predict_patterns = [
            # What happens after X?
            (r"what (?:happens|comes) (?:after|next after|following) (\w+)\??", self._extract_after),
            # What's next after X?
            (r"what(?:'s| is) next (?:after )?(\w+)?\??", self._extract_next),
            # After X, what?
            (r"after (\w+),? what\??", self._extract_after),
        ]

        # Simple intent patterns
        self.simple_intents = {
            Intent.GREETING: [r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b"],
            Intent.FAREWELL: [r"^(bye|goodbye|see you|farewell|quit|exit)\b"],
            Intent.HELP: [r"^(help|how do|what can you|commands?)\b", r"\?$.*help"],
            Intent.CONFIRM: [r"^(yes|yeah|yep|ok|okay|sure|correct|right)\b"],
            Intent.DENY: [r"^(no|nope|nah|wrong|incorrect)\b"],
            Intent.LIST: [r"^(list|show|what do you know)\b", r"^show me"],
        }

        # Context setting patterns
        self.context_patterns = [
            (r"let'?s talk about (\w+)", self._extract_set_context),
            (r"talking about (\w+)", self._extract_set_context),
            (r"regarding (\w+)", self._extract_set_context),
        ]

    def parse(self, text: str) -> ParseResult:
        """Parse user input into structured intent + entities."""
        original = text
        text = text.strip().lower()

        # Check simple intents first
        for intent, patterns in self.simple_intents.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return ParseResult(intent, {}, 0.9, original)

        # Check context-setting patterns
        for pattern, extractor in self.context_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities = extractor(match)
                return ParseResult(Intent.CONTEXT, entities, 0.9, original)

        # Check if it's a question
        is_question = '?' in text or text.startswith(('what', 'where', 'who', 'how', 'why', 'when', 'is', 'does', 'do', 'can'))

        if is_question:
            # Try prediction patterns
            for pattern, extractor in self.predict_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entities = extractor(match)
                    return ParseResult(Intent.PREDICT, entities, 0.8, original)

            # Try question patterns
            for pattern, extractor in self.question_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entities = extractor(match)
                    return ParseResult(Intent.QUERY, entities, 0.8, original)

        # Try teaching patterns
        for pattern, extractor in self.teach_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities = extractor(match)
                return ParseResult(Intent.TEACH, entities, 0.8, original)

        # Try statement patterns
        for pattern, extractor in self.statement_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities = extractor(match)
                if entities:
                    return ParseResult(Intent.STORE, entities, 0.7, original)

        # Fallback: try to extract any entities
        entities = self._extract_fallback(text)
        if is_question:
            return ParseResult(Intent.QUERY, entities, 0.3, original)
        elif entities:
            return ParseResult(Intent.STORE, entities, 0.3, original)

        return ParseResult(Intent.UNKNOWN, {}, 0.0, original)

    # Extractors for questions
    def _extract_what_query(self, match) -> Dict:
        return {"entity": match.group(1), "role": match.group(2)}

    def _extract_what_have(self, match) -> Dict:
        return {"entity": match.group(2), "role": match.group(1)}

    def _extract_where(self, match) -> Dict:
        return {"entity": match.group(1), "role": "location"}

    def _extract_who(self, match) -> Dict:
        return {"entity": match.group(1), "role": "identity"}

    def _extract_does_have(self, match) -> Dict:
        return {"entity": match.group(1), "role": match.group(2), "check_type": "has"}

    def _extract_is_a(self, match) -> Dict:
        return {"entity": match.group(1), "role": "type", "expected": match.group(2)}

    def _extract_about(self, match) -> Dict:
        return {"entity": match.group(1), "role": "*"}  # Wildcard role

    def _extract_generic_question(self, match) -> Dict:
        return {"question_type": match.group(1)}

    def _extract_what_is(self, match) -> Dict:
        return {"entity": match.group(1), "role": "type"}

    def _extract_the_of(self, match) -> Dict:
        return {"entity": match.group(2), "role": match.group(1)}

    def _extract_type_query(self, match) -> Dict:
        return {"entity": match.group(1), "role": "type"}

    def _extract_require(self, match) -> Dict:
        return {"entity": match.group(1), "role": "requires"}

    def _extract_my_query(self, match) -> Dict:
        return {"entity": "user", "role": match.group(1)}

    def _extract_what_attr(self, match) -> Dict:
        # "What color is the sky?" -> sky.color
        role = match.group(1)  # color
        entity = match.group(2)  # sky
        return {"entity": entity, "role": role}

    def _extract_what_the(self, match) -> Dict:
        # "What is the code?" -> code.value
        entity = match.group(1)
        return {"entity": entity, "role": "value"}

    # Extractors for statements
    def _extract_is_statement(self, match) -> Dict:
        entity = match.group(1)
        value = match.group(2)
        # Skip if it looks like a question fragment
        if entity.lower() in ('what', 'where', 'who', 'that', 'this', 'it'):
            return {}
        return {"entity": entity, "role": "type", "value": value}

    def _extract_possessive(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": match.group(2),
            "value": match.group(3)
        }

    def _extract_of_statement(self, match) -> Dict:
        return {
            "entity": match.group(2),
            "role": match.group(1),
            "value": match.group(3)
        }

    def _extract_works_at(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": "workplace",
            "value": match.group(2)
        }

    def _extract_lives_in(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": "location",
            "value": match.group(2)
        }

    def _extract_has(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": "has",
            "value": match.group(2)
        }

    def _extract_knows(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": "knows",
            "value": match.group(2)
        }

    def _extract_remember(self, match) -> Dict:
        # Re-parse the content after "remember"
        content = match.group(1)
        return {"raw_statement": content}

    def _extract_my_statement(self, match) -> Dict:
        return {
            "entity": "user",
            "role": match.group(1),
            "value": match.group(2)
        }

    def _extract_context_statement(self, match) -> Dict:
        # "His job is X" - uses context entity
        return {
            "entity": "_context_",  # Placeholder for context entity
            "role": match.group(1),
            "value": match.group(2)
        }

    def _extract_remember_fact(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": "value",
            "value": match.group(2)
        }

    def _extract_note_fact(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": "value",
            "value": match.group(2)
        }

    def _extract_the_is(self, match) -> Dict:
        # "The sky is blue" -> sky.color = blue (infer role)
        entity = match.group(1)
        value = match.group(2)
        # Try to infer the role
        color_words = ['blue', 'red', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown', 'gray', 'grey']
        if value.lower() in color_words:
            role = 'color'
        else:
            role = 'type'
        return {
            "entity": entity,
            "role": role,
            "value": value
        }

    def _extract_is_location(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": "location",
            "value": match.group(2)
        }

    def _extract_set_context(self, match) -> Dict:
        return {
            "context_entity": match.group(1)
        }

    def _extract_age(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": "age",
            "value": match.group(2)
        }

    def _extract_context_lives(self, match) -> Dict:
        return {
            "entity": "_context_",
            "role": "location",
            "value": match.group(1)
        }

    def _extract_note_location(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": "location",
            "value": match.group(2)
        }

    def _extract_is_under(self, match) -> Dict:
        return {
            "entity": match.group(1),
            "role": "location",
            "value": match.group(2)
        }

    def _extract_how_old(self, match) -> Dict:
        return {"entity": match.group(1), "role": "age"}

    def _extract_what_has(self, match) -> Dict:
        return {"entity": match.group(1), "role": "has"}

    def _extract_when(self, match) -> Dict:
        return {"entity": match.group(1), "role": "time"}

    # Extractors for teaching
    def _extract_source(self, match) -> Dict:
        return {
            "topic": match.group(1),
            "source": match.group(2).strip()
        }

    # Extractors for prediction
    def _extract_after(self, match) -> Dict:
        return {"context": match.group(1)}

    def _extract_next(self, match) -> Dict:
        context = match.group(1) if match.group(1) else ""
        return {"context": context}

    # Fallback extraction
    def _extract_fallback(self, text: str) -> Dict:
        """Try to extract something useful from unmatched text."""
        words = re.findall(r'\b[A-Z][a-z]+\b', text)  # Capitalized words
        if words:
            return {"entities": words}
        return {}


# Singleton for easy import
_parser = None

def get_parser() -> Parser:
    global _parser
    if _parser is None:
        _parser = Parser()
    return _parser


def parse(text: str) -> ParseResult:
    """Convenience function."""
    return get_parser().parse(text)
