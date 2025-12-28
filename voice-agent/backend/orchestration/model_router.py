"""
Model Router for Kotak Gold Loan Voice Agent.

Decides when to use thinking model vs fast model based on query complexity.
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Available model types."""
    FAST = "fast"           # qwen3:8b-q4_K_M (non-thinking)
    THINKING = "thinking"   # qwen3:8b-q4_K_M (thinking mode)


@dataclass
class RoutingDecision:
    """Decision from model router."""
    model_type: ModelType
    reason: str
    confidence: float  # 0-1


class ModelRouter:
    """
    Routes queries to appropriate model based on complexity.

    Thinking model used for:
    - Complex calculations
    - Ambiguous situations requiring judgment
    - Objection handling
    - Multi-step reasoning

    Fast model used for:
    - Simple Q&A
    - Greetings/confirmations
    - Straightforward information retrieval
    """

    def __init__(
        self,
        thinking_model: str = "qwen3:8b-q4_K_M",
        fast_model: str = "qwen3:8b-q4_K_M"
    ):
        self.thinking_model = thinking_model
        self.fast_model = fast_model

        # Patterns that suggest thinking is needed
        self.thinking_patterns = [
            # Calculations
            r'\b(calculate|compute|how much|kitna|savings?|emi|compare)\b',
            # Objections/concerns
            r'\b(but|however|worried|concern|safe|auction|risk|problem|issue)\b',
            r'\b(lekin|par|chinta|dar|suraksha)\b',  # Hindi
            # Decision making
            r'\b(should i|which is better|recommend|suggest|advice)\b',
            r'\b(kya karoon|konsa behtar|salah)\b',  # Hindi
            # Complex comparisons
            r'\b(vs|versus|compare|difference|muthoot|manappuram|iifl)\b',
            # Negotiation
            r'\b(rate kam|discount|special offer|negotiate)\b',
            # Complaints
            r'\b(complaint|unhappy|not satisfied|bad experience)\b',
            r'\b(shikayat|pareshani)\b',  # Hindi
        ]

        # Patterns for fast responses
        self.fast_patterns = [
            # Greetings
            r'^(hi|hello|hey|namaste|good morning|good evening)\b',
            # Simple confirmations
            r'^(yes|no|ok|okay|haan|nahi|theek hai)\b',
            # Simple questions with clear answers
            r'\b(what is your name|who are you|thank you|dhanyavaad)\b',
            # Direct information requests
            r'^(what|where|when|how).*\?$',
            r'\b(documents|timing|branch|address|phone|contact)\b',
        ]

    def route(self, query: str, context: dict | None = None) -> RoutingDecision:
        """
        Route a query to the appropriate model.

        Args:
            query: User's query
            context: Optional conversation context

        Returns:
            RoutingDecision with model type and reason
        """
        query_lower = query.lower().strip()

        # Check for tool-related queries (need thinking for parameter extraction)
        if self._needs_tool_call(query_lower):
            return RoutingDecision(
                model_type=ModelType.THINKING,
                reason="Query requires tool usage with parameter extraction",
                confidence=0.9
            )

        # Check thinking patterns
        thinking_score = 0
        for pattern in self.thinking_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                thinking_score += 1

        # Check fast patterns
        fast_score = 0
        for pattern in self.fast_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                fast_score += 1

        # Check query length (longer = more complex)
        word_count = len(query_lower.split())
        if word_count > 15:
            thinking_score += 1
        elif word_count < 5:
            fast_score += 1

        # Check for question complexity
        if query_lower.count('?') > 1:
            thinking_score += 1  # Multiple questions

        # Context-based routing
        if context:
            # If there's an unresolved objection, use thinking
            if context.get("has_objection", False):
                thinking_score += 2

            # If calculating something, use thinking
            if context.get("pending_calculation", False):
                thinking_score += 2

        # Make decision
        if thinking_score > fast_score:
            confidence = min(0.5 + (thinking_score - fast_score) * 0.15, 0.95)
            return RoutingDecision(
                model_type=ModelType.THINKING,
                reason=f"Complex query (thinking_score={thinking_score}, fast_score={fast_score})",
                confidence=confidence
            )
        else:
            confidence = min(0.5 + (fast_score - thinking_score) * 0.15, 0.95)
            return RoutingDecision(
                model_type=ModelType.FAST,
                reason=f"Simple query (fast_score={fast_score}, thinking_score={thinking_score})",
                confidence=confidence
            )

    def _needs_tool_call(self, query: str) -> bool:
        """Check if query likely needs a tool call."""
        tool_indicators = [
            r'\b(calculate|compute|kitna milega|kitni emi|kitna bachega)\b',
            r'\b(nearest branch|branch.*city|support ticket|callback)\b',
            r'\b(loan amount|eligible|savings)\b',
            r'\b\d+\s*(lakh|crore|rupees|rs|₹)\b',  # Amount mentioned
        ]

        for pattern in tool_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def get_model_name(self, model_type: ModelType) -> str:
        """Get actual model name for a model type."""
        if model_type == ModelType.THINKING:
            return self.thinking_model
        return self.fast_model
