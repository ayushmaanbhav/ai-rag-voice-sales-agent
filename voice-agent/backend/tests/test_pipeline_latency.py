"""
Pipeline Latency Tests for Voice Agent.

Tests the full STT -> LLM (with RAG) -> TTS pipeline latency.
"""
import asyncio
import time
import base64
import wave
import io
import pytest


def create_test_audio(duration_sec: float = 1.0) -> str:
    """Create test audio WAV file as base64."""
    sample_rate = 16000
    samples = int(sample_rate * duration_sec)
    audio_data = b'\x00\x00' * samples  # 16-bit silence

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    return base64.b64encode(wav_buffer.getvalue()).decode()


@pytest.fixture
def audio_base64():
    """Fixture for test audio."""
    return create_test_audio()


@pytest.fixture
def reset_singletons():
    """Reset singleton instances before each test."""
    import speech.whisper_stt as wm
    import speech.piper_tts as pm
    wm._whisper_instance = None
    pm._piper_instance = None
    yield
    wm._whisper_instance = None
    pm._piper_instance = None


class TestSTTLatency:
    """Test STT (Whisper) latency."""

    @pytest.mark.asyncio
    async def test_whisper_stt_latency(self, audio_base64, reset_singletons):
        """Test Whisper STT transcription latency."""
        from speech.providers import get_stt_provider

        stt = get_stt_provider()

        # First call (includes model loading)
        start = time.time()
        result = await stt.transcribe(audio_base64, language="hi", audio_format="wav")
        first_call_time = time.time() - start

        # Second call (model cached)
        start = time.time()
        result = await stt.transcribe(audio_base64, language="hi", audio_format="wav")
        cached_time = time.time() - start

        print(f"\nSTT Latency:")
        print(f"  First call (with model load): {first_call_time:.2f}s")
        print(f"  Cached call: {cached_time:.2f}s")

        # Cached call should be under 5 seconds
        assert cached_time < 5.0, f"STT too slow: {cached_time:.2f}s"


class TestTTSLatency:
    """Test TTS (Piper) latency."""

    @pytest.mark.asyncio
    async def test_piper_tts_latency(self, reset_singletons):
        """Test TTS synthesis latency."""
        from speech.providers import get_tts_provider
        from config.settings import settings

        tts = get_tts_provider()
        test_text = "नमस्ते, मैं प्रिया कोटक बैंक से बोल रही हूँ"

        # First call (includes model loading)
        start = time.time()
        result = await tts.synthesize(test_text, language="hi")
        first_call_time = time.time() - start

        # Second call (model cached)
        start = time.time()
        result = await tts.synthesize(test_text, language="hi")
        cached_time = time.time() - start

        print(f"\nTTS Latency ({settings.tts_provider}):")
        print(f"  First call (with model load): {first_call_time:.2f}s")
        print(f"  Cached call: {cached_time:.2f}s")

        assert result is not None
        # Different providers have different performance characteristics
        # Piper: ~1s on CPU, IndicF5: 60-90s on CPU (heavy model)
        if settings.tts_provider == "piper":
            assert cached_time < 1.0, f"Piper TTS too slow: {cached_time:.2f}s"
        elif settings.tts_provider == "indicf5":
            # IndicF5 is expected to be slower on CPU (no GPU)
            assert cached_time < 120.0, f"IndicF5 TTS too slow: {cached_time:.2f}s"
        else:
            assert cached_time < 5.0, f"TTS too slow: {cached_time:.2f}s"


class TestLLMLatency:
    """Test LLM (Ollama) latency."""

    @pytest.mark.asyncio
    async def test_ollama_llm_latency(self):
        """Test Ollama LLM response latency."""
        from llm.ollama_client import OllamaClient

        llm = OllamaClient()

        start = time.time()
        response = await llm.generate_response(
            system_prompt="You are Priya from Kotak Bank. Reply briefly in Hindi.",
            conversation_history=[],
            user_message="User said: हां",
            temperature=0.7
        )
        elapsed = time.time() - start

        print(f"\nLLM Latency: {elapsed:.2f}s")
        print(f"Response: {response[:50]}...")

        assert len(response) > 0
        assert elapsed < 10.0, f"LLM too slow: {elapsed:.2f}s"


class TestConversationWithRAG:
    """Test conversation with RAG context."""

    @pytest.mark.asyncio
    async def test_conversation_with_rag(self):
        """Test conversation response time with RAG enabled."""
        from llm.ollama_client import OllamaClient
        from conversation.state_machine import ConversationManager
        from personalization.customer_profile import CustomerDB
        from orchestration.orchestrator import VoiceAgentOrchestrator

        db = CustomerDB()
        customer = db.get_customer("C001")
        orchestrator = VoiceAgentOrchestrator()
        llm = OllamaClient()

        cm = ConversationManager(
            customer=customer,
            language="hi",
            sarvam_client=None,
            llm_client=llm,
            orchestrator=orchestrator
        )

        start = time.time()
        response = await cm.process_user_input("हां, बताइये")
        elapsed = time.time() - start

        print(f"\nConversation with RAG Latency: {elapsed:.2f}s")
        print(f"Response: {response}")

        assert len(response) > 0
        assert elapsed < 15.0, f"Conversation too slow: {elapsed:.2f}s"


class TestFullPipeline:
    """Test full voice pipeline latency."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, audio_base64, reset_singletons):
        """Test complete STT -> LLM (RAG) -> TTS pipeline."""
        from speech.providers import get_stt_provider, get_tts_provider
        from llm.ollama_client import OllamaClient
        from conversation.state_machine import ConversationManager
        from personalization.customer_profile import CustomerDB
        from orchestration.orchestrator import VoiceAgentOrchestrator

        stt = get_stt_provider()
        tts = get_tts_provider()

        db = CustomerDB()
        customer = db.get_customer("C001")
        orchestrator = VoiceAgentOrchestrator()
        llm = OllamaClient()

        cm = ConversationManager(
            customer=customer,
            language="hi",
            sarvam_client=None,
            llm_client=llm,
            orchestrator=orchestrator
        )

        # Warm up models
        await stt.transcribe(audio_base64, language="hi", audio_format="wav")
        await tts.synthesize("टेस्ट", language="hi")

        print("\n" + "=" * 55)
        print("FULL PIPELINE TEST")
        print("=" * 55)

        # STT
        start = time.time()
        stt_result = await stt.transcribe(audio_base64, language="hi", audio_format="wav")
        stt_time = time.time() - start

        # Conversation (with RAG + LLM)
        start = time.time()
        response = await cm.process_user_input("हां, बताइये")
        conv_time = time.time() - start

        # TTS
        start = time.time()
        tts_result = await tts.synthesize(response[:60], language="hi")
        tts_time = time.time() - start

        total = stt_time + conv_time + tts_time

        from config.settings import settings
        print(f"\nSTT ({settings.stt_provider}):  {stt_time:.2f}s")
        print(f"Conversation:                  {conv_time:.2f}s")
        print(f"TTS ({settings.tts_provider}):  {tts_time:.2f}s")
        print(f"---")
        print(f"TOTAL:                         {total:.2f}s")
        print("=" * 55)
        print(f"\nResponse: {response}")

        assert tts_result is not None
        # Different providers have different performance characteristics
        if settings.tts_provider == "indicf5":
            # IndicF5 is expected to be slower on CPU (no GPU)
            assert total < 200.0, f"Pipeline too slow: {total:.2f}s"
        else:
            assert total < 20.0, f"Pipeline too slow: {total:.2f}s"


if __name__ == "__main__":
    # Run specific test
    asyncio.run(TestFullPipeline().test_full_pipeline(
        create_test_audio(),
        None  # reset_singletons handled manually
    ))
