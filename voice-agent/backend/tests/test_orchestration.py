"""
Tests for orchestration: model router, orchestrator.
"""
import pytest
from orchestration.model_router import ModelRouter, ModelType, RoutingDecision
from orchestration.orchestrator import VoiceAgentOrchestrator, ConversationContext
from rag.vector_store import ChromaVectorStore


class TestModelRouter:
    """Test model routing logic."""

    @pytest.fixture
    def router(self):
        return ModelRouter()

    def test_fast_model_for_simple_queries(self, router):
        """Test fast model selected for simple queries."""
        simple_queries = [
            "What is the interest rate?",
            "Tell me about gold loan",
            "How much can I borrow?",
        ]
        for query in simple_queries:
            decision = router.route(query)
            assert decision.model_type == ModelType.FAST, f"Failed for: {query}"

    def test_thinking_model_for_complex_queries(self, router):
        """Test thinking model for complex queries."""
        complex_queries = [
            "Compare Kotak with Muthoot Finance rates",
            "Why should I trust Kotak after the news?",
            "Calculate my savings if I switch",
        ]
        for query in complex_queries:
            decision = router.route(query)
            assert decision.model_type == ModelType.THINKING, f"Failed for: {query}"

    def test_thinking_model_for_objections(self, router):
        """Test thinking model when objection context present."""
        decision = router.route(
            "I'm not sure about this",
            context={"has_objection": True}
        )
        assert decision.model_type == ModelType.THINKING

    def test_thinking_model_for_calculations(self, router):
        """Test thinking model for calculation requests."""
        decision = router.route(
            "What will be my EMI?",
            context={"pending_calculation": True}
        )
        assert decision.model_type == ModelType.THINKING

    def test_get_model_name(self, router):
        """Test getting model names."""
        fast_name = router.get_model_name(ModelType.FAST)
        thinking_name = router.get_model_name(ModelType.THINKING)

        assert "instruct" in fast_name.lower()
        assert "instruct" not in thinking_name.lower() or thinking_name != fast_name


class TestConversationContext:
    """Test conversation context tracking."""

    def test_default_values(self):
        """Test default context values."""
        ctx = ConversationContext()
        assert ctx.customer_segment is None
        assert ctx.detected_language == "en"
        assert ctx.has_objection is False
        assert ctx.turn_count == 0
        assert ctx.history == []

    def test_context_updates(self):
        """Test context can be updated."""
        ctx = ConversationContext()
        ctx.customer_segment = "P1"
        ctx.has_objection = True
        ctx.turn_count = 5

        assert ctx.customer_segment == "P1"
        assert ctx.has_objection is True
        assert ctx.turn_count == 5


class TestVoiceAgentOrchestrator:
    """Test voice agent orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        vector_store = ChromaVectorStore(
            persist_dir="./data/chroma",
            collection_name="kotak_knowledge"
        )
        return VoiceAgentOrchestrator(vector_store=vector_store)

    def test_detect_language_english(self, orchestrator):
        """Test English language detection."""
        lang = orchestrator.detect_language("What is the interest rate?")
        assert lang == "en"

    def test_detect_language_hindi(self, orchestrator):
        """Test Hindi language detection."""
        lang = orchestrator.detect_language("ब्याज दर क्या है?")
        assert lang == "hi"

    def test_detect_language_mixed(self, orchestrator):
        """Test mixed language detection."""
        # Mostly English with some Hindi
        lang = orchestrator.detect_language("What is the byaj rate?")
        assert lang == "en"

    def test_detect_objection_safety(self, orchestrator):
        """Test safety objection detection."""
        obj = orchestrator.detect_objection("Is my gold safe with you?")
        assert obj == "safety"

    def test_detect_objection_rate(self, orchestrator):
        """Test rate objection detection."""
        obj = orchestrator.detect_objection("Your interest rate seems too high")
        assert obj == "rate"

    def test_detect_objection_auction(self, orchestrator):
        """Test auction objection detection."""
        obj = orchestrator.detect_objection("What if you auction my gold?")
        assert obj == "auction"

    def test_detect_objection_trust(self, orchestrator):
        """Test trust objection detection."""
        obj = orchestrator.detect_objection("Can I trust Kotak after the news?")
        assert obj == "trust"

    def test_detect_no_objection(self, orchestrator):
        """Test no objection detected."""
        obj = orchestrator.detect_objection("Tell me about gold loan")
        assert obj is None

    def test_detect_segment_p1(self, orchestrator):
        """Test P1 (high-value) segment detection."""
        segment = orchestrator.detect_segment(
            "I need 10 lakh for my business",
            []
        )
        assert segment == "P1"

    def test_detect_segment_p2(self, orchestrator):
        """Test P2 (trust-seeker) segment detection."""
        segment = orchestrator.detect_segment(
            "I'm worried about safety after the news",
            []
        )
        assert segment == "P2"

    def test_detect_segment_p4(self, orchestrator):
        """Test P4 (young professional) segment detection."""
        segment = orchestrator.detect_segment(
            "Can I apply through the app quickly?",
            []
        )
        assert segment == "P4"

    def test_retrieve_context(self, orchestrator):
        """Test context retrieval."""
        docs = orchestrator.retrieve_context(
            "What is the interest rate?",
            "en",
            None
        )
        assert len(docs) > 0
        assert all("content" in d for d in docs)

    def test_retrieve_context_with_objection(self, orchestrator):
        """Test context retrieval with objection."""
        docs = orchestrator.retrieve_context(
            "Is my gold safe?",
            "en",
            "safety"
        )
        assert len(docs) > 0
        # Should have safety-related content
        contents = " ".join(d["content"].lower() for d in docs)
        assert any(word in contents for word in ["safe", "vault", "security"])

    def test_build_system_prompt(self, orchestrator):
        """Test system prompt building."""
        prompt = orchestrator.build_system_prompt("en", "P1", None)
        assert "Kotak" in prompt
        assert "High-Value" in prompt

    def test_build_system_prompt_with_objection(self, orchestrator):
        """Test system prompt with objection context."""
        prompt = orchestrator.build_system_prompt("en", None, "safety")
        assert "OBJECTION DETECTED: safety" in prompt

    def test_build_system_prompt_hindi(self, orchestrator):
        """Test Hindi system prompt."""
        prompt = orchestrator.build_system_prompt("hi", None, None)
        assert "Hindi" in prompt

    def test_reset_context(self, orchestrator):
        """Test context reset."""
        orchestrator.context.turn_count = 10
        orchestrator.context.has_objection = True

        orchestrator.reset_context()

        assert orchestrator.context.turn_count == 0
        assert orchestrator.context.has_objection is False


class TestIntegration:
    """Integration tests for the full orchestration flow."""

    @pytest.fixture
    def orchestrator(self):
        vector_store = ChromaVectorStore(
            persist_dir="./data/chroma",
            collection_name="kotak_knowledge"
        )
        return VoiceAgentOrchestrator(vector_store=vector_store)

    def test_full_retrieval_flow(self, orchestrator):
        """Test complete retrieval flow."""
        # Simulate user asking about interest rates
        query = "What is Kotak's gold loan interest rate?"
        language = orchestrator.detect_language(query)
        objection = orchestrator.detect_objection(query)
        docs = orchestrator.retrieve_context(query, language, objection)

        assert language == "en"
        assert objection is None or objection == "rate"
        assert len(docs) > 0

    def test_objection_handling_flow(self, orchestrator):
        """Test objection handling flow."""
        query = "I don't trust banks, what if you auction my gold?"

        language = orchestrator.detect_language(query)
        objection = orchestrator.detect_objection(query)
        segment = orchestrator.detect_segment(query, [])
        docs = orchestrator.retrieve_context(query, language, objection)

        assert objection in ["auction", "trust"]
        assert len(docs) > 0

        # Build prompt should include objection context
        prompt = orchestrator.build_system_prompt(language, segment, objection)
        assert "OBJECTION DETECTED" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
