"""
Voice Agent Orchestrator for Kotak Gold Loan.

Coordinates:
- RAG retrieval for context
- Model routing (thinking vs fast)
- Tool execution
- Response generation
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

from rag.vector_store import ChromaVectorStore
from rag.retriever import HybridRetriever
from tools.executor import ToolExecutor, get_tool_executor
from orchestration.model_router import ModelRouter, ModelType, RoutingDecision

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Tracks conversation state."""
    customer_segment: Optional[str] = None  # P1, P2, P3, P4
    detected_language: str = "en"
    has_objection: bool = False
    current_objection_type: Optional[str] = None
    pending_calculation: bool = False
    mentioned_competitor: Optional[str] = None
    loan_amount_mentioned: Optional[float] = None
    interest_rate_mentioned: Optional[float] = None
    turn_count: int = 0
    history: list[dict] = field(default_factory=list)


@dataclass
class OrchestratorResponse:
    """Response from orchestrator."""
    response_text: str
    model_used: str
    rag_context_used: bool
    tools_used: list[str]
    routing_decision: Optional[RoutingDecision] = None
    retrieved_docs: list[dict] = field(default_factory=list)


class VoiceAgentOrchestrator:
    """
    Main orchestrator for the voice agent.

    Flow:
    1. Detect language from input
    2. Route to appropriate model
    3. Retrieve relevant context from RAG
    4. Generate response with LLM
    5. Execute any tool calls
    6. Format final response
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        chroma_persist_dir: str = "./data/chroma",
        max_context_tokens: int = 2000,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize orchestrator.

        Args:
            vector_store: Pre-initialized vector store (optional)
            chroma_persist_dir: Path to ChromaDB persistence
            max_context_tokens: Max tokens for RAG context
            ollama_base_url: Ollama API base URL
        """
        # Initialize components
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = ChromaVectorStore(
                persist_dir=chroma_persist_dir,
                collection_name="kotak_knowledge"
            )

        self.retriever = HybridRetriever(self.vector_store)
        self.router = ModelRouter()
        self.tool_executor = get_tool_executor()
        self.ollama_base_url = ollama_base_url
        self.max_context_tokens = max_context_tokens

        # Conversation state
        self.context = ConversationContext()

    def detect_language(self, text: str) -> str:
        """
        Detect language from text (simple heuristic).

        Returns 'hi' for Hindi, 'en' for English.
        """
        # Hindi Unicode range
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        total_chars = len(text.replace(' ', ''))

        if total_chars > 0 and hindi_chars / total_chars > 0.3:
            return "hi"
        return "en"

    def detect_segment(self, text: str, history: list[dict]) -> Optional[str]:
        """
        Detect customer segment from conversation.

        P1: High-Value (mentions large amounts, business)
        P2: Trust-Seeker (mentions safety, news, concern)
        P3: Women/Shakti (mentions women, SHG, business women)
        P4: Young Professional (digital-first, quick, app)
        """
        text_lower = text.lower()
        all_text = text_lower + " ".join(
            m.get("content", "").lower() for m in history
        )

        # P1 indicators
        if any(x in all_text for x in ["lakh", "crore", "business", "msme", "vyapar"]):
            return "P1"

        # P2 indicators
        if any(x in all_text for x in ["safe", "news", "worried", "trust", "suraksha", "chinta"]):
            return "P2"

        # P3 indicators
        if any(x in all_text for x in ["mahila", "women", "shakti", "shg"]):
            return "P3"

        # P4 indicators
        if any(x in all_text for x in ["app", "online", "quick", "jaldi", "digital"]):
            return "P4"

        return None

    def detect_objection(self, text: str) -> Optional[str]:
        """Detect if user is raising an objection."""
        text_lower = text.lower()

        objection_map = {
            "safety": ["safe", "suraksha", "theft", "chori", "lost", "vault"],
            "rate": ["interest", "byaj", "rate", "expensive", "mehnga"],
            "auction": ["auction", "nilaam", "seize", "default", "late"],
            "loyalty": ["happy", "satisfied", "used to", "aadat", "relationship"],
            "time": ["time", "samay", "fast", "jaldi", "slow", "wait"],
            "trust": ["trust", "bharosa", "news", "problem", "rbi"],
        }

        for obj_type, keywords in objection_map.items():
            if any(kw in text_lower for kw in keywords):
                return obj_type

        return None

    def retrieve_context(
        self,
        query: str,
        language: str,
        objection_type: Optional[str] = None
    ) -> list[dict]:
        """
        Retrieve relevant context from RAG.

        Args:
            query: User query
            language: Detected language
            objection_type: If objection detected, type of objection

        Returns:
            List of relevant documents
        """
        retrieved = []

        # If objection, prioritize objection handling docs
        if objection_type:
            objection_docs = self.retriever.retrieve_for_objection(
                query, n_results=2, language=language
            )
            retrieved.extend(objection_docs)

        # General retrieval
        general_docs = self.retriever.retrieve(
            query, n_results=3, language=language
        )

        # Deduplicate
        seen_ids = {d["id"] for d in retrieved}
        for doc in general_docs:
            if doc["id"] not in seen_ids:
                retrieved.append(doc)
                seen_ids.add(doc["id"])

        return retrieved[:5]  # Max 5 docs

    def build_system_prompt(
        self,
        language: str,
        segment: Optional[str],
        objection_type: Optional[str]
    ) -> str:
        """Build system prompt based on context."""

        base_prompt = """You are a helpful Kotak Mahindra Bank gold loan assistant.

IMPORTANT GUIDELINES:
1. Be conversational and friendly, like a helpful bank representative
2. Focus on helping customers understand Kotak's gold loan benefits
3. If asked about competitors, subtly highlight Kotak's advantages without being negative
4. Always emphasize: bank-grade security, competitive rates (9-12%), same-day disbursement

RESPONSE FORMAT:
- Keep responses concise (2-3 sentences for voice)
- Use simple language the customer can understand
- If calculations are needed, use the provided tools

AVAILABLE TOOLS:
You can use these tools by including a <tool_call> block in your response:
"""
        # Add tools description
        base_prompt += self.tool_executor.get_tools_prompt()

        base_prompt += """

TOOL USAGE FORMAT:
<tool_call>
  <name>tool_name</name>
  <parameters>
    <param1>value1</param1>
  </parameters>
</tool_call>

"""

        # Add segment-specific guidance
        if segment == "P1":
            base_prompt += """
CUSTOMER SEGMENT: High-Value Business Owner
- Focus on savings calculations and business benefits
- Emphasize relationship banking advantages
- Mention higher loan limits available
"""
        elif segment == "P2":
            base_prompt += """
CUSTOMER SEGMENT: Trust-Seeker
- Emphasize bank-grade security and RBI regulation
- Highlight zero regulatory issues track record
- Reassure about gold safety with insurance coverage
"""
        elif segment == "P3":
            base_prompt += """
CUSTOMER SEGMENT: Women Entrepreneur (Shakti)
- Be respectful and empowering
- Mention special benefits for women borrowers
- Highlight lower documentation requirements
"""
        elif segment == "P4":
            base_prompt += """
CUSTOMER SEGMENT: Young Professional
- Keep it quick and digital-focused
- Mention app features and fast processing
- Use modern, relatable language
"""

        # Add objection handling guidance
        if objection_type:
            base_prompt += f"""
OBJECTION DETECTED: {objection_type}
- Address this concern directly but reassuringly
- Use the retrieved context to provide accurate information
- Don't be defensive; acknowledge the concern and solve it
"""

        # Language guidance
        if language == "hi":
            base_prompt += """
LANGUAGE: Respond in Hindi (Devanagari script).
Use conversational Hindi that's easy to understand.
"""
        else:
            base_prompt += """
LANGUAGE: Respond in English.
Keep it conversational and avoid banking jargon.
"""

        return base_prompt

    def format_context_for_prompt(self, docs: list[dict]) -> str:
        """Format retrieved documents for prompt injection."""
        if not docs:
            return ""

        context = "\n\nRELEVANT INFORMATION:\n"
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "")
            # Truncate if too long
            if len(content) > 500:
                content = content[:500] + "..."
            context += f"\n[{i}] {content}\n"

        return context

    async def process_message(
        self,
        user_message: str,
        llm_client  # OllamaClient instance
    ) -> OrchestratorResponse:
        """
        Process a user message and generate response.

        Args:
            user_message: User's input
            llm_client: Ollama client for LLM calls

        Returns:
            OrchestratorResponse with response and metadata
        """
        # Update turn count
        self.context.turn_count += 1

        # Detect language
        language = self.detect_language(user_message)
        self.context.detected_language = language

        # Detect segment if not already known
        if not self.context.customer_segment:
            segment = self.detect_segment(user_message, self.context.history)
            if segment:
                self.context.customer_segment = segment

        # Detect objection
        objection_type = self.detect_objection(user_message)
        if objection_type:
            self.context.has_objection = True
            self.context.current_objection_type = objection_type

        # Route to model
        routing = self.router.route(user_message, {
            "has_objection": self.context.has_objection,
            "pending_calculation": self.context.pending_calculation
        })
        model_name = self.router.get_model_name(routing.model_type)

        # Retrieve context
        retrieved_docs = self.retrieve_context(
            user_message,
            language,
            objection_type
        )

        # Build prompts
        system_prompt = self.build_system_prompt(
            language,
            self.context.customer_segment,
            objection_type
        )

        # Add retrieved context
        rag_context = self.format_context_for_prompt(retrieved_docs)
        if rag_context:
            system_prompt += rag_context

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent history (last 4 turns)
        for msg in self.context.history[-8:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})

        # Generate response
        logger.info(f"Using model: {model_name} (reason: {routing.reason})")

        raw_response = await llm_client.generate(
            prompt=user_message,
            system_prompt=system_prompt,
            model=model_name
        )

        # Process tool calls
        cleaned_response, tool_results = self.tool_executor.process_llm_output(
            raw_response
        )

        # If tools were used, append results
        tools_used = []
        if tool_results:
            for result in tool_results:
                if result.success:
                    cleaned_response += f"\n\n{result.display_text}"
                    tools_used.append("tool_executed")

        # Update history
        self.context.history.append({"role": "user", "content": user_message})
        self.context.history.append({"role": "assistant", "content": cleaned_response})

        # Clear objection if addressed
        if objection_type and cleaned_response:
            self.context.has_objection = False
            self.context.current_objection_type = None

        return OrchestratorResponse(
            response_text=cleaned_response,
            model_used=model_name,
            rag_context_used=bool(retrieved_docs),
            tools_used=tools_used,
            routing_decision=routing,
            retrieved_docs=retrieved_docs
        )

    def reset_context(self) -> None:
        """Reset conversation context."""
        self.context = ConversationContext()
