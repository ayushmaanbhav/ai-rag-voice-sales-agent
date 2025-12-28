<p align="center">
  <img src="https://img.shields.io/badge/Kotak-Gold%20Loan%20Voice%20Agent-FFD700?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI0ZGRDcwMCI+PHBhdGggZD0iTTEyIDJMMTUuMDkgOC4yNkwyMiA5LjI3TDE3IDEzLjE0TDE4LjE4IDIwLjAyTDEyIDE2Ljc3TDUuODIgMjAuMDJMNyAxMy4xNEwyIDkuMjdMOC45MSA4LjI2TDEyIDJ6Ii8+PC9zdmc+" alt="Kotak Gold Loan">
</p>

<h1 align="center">Kotak Gold Loan Voice Agent</h1>

<p align="center">
  <strong>AI-Powered Multilingual Voice Assistant for Gold Loan Customer Acquisition</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/React-18.2-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/TypeScript-5.2-3178C6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=flat-square" alt="Ollama">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Languages-7%20Indian%20Languages-FF9933?style=flat-square" alt="Languages">
  <img src="https://img.shields.io/badge/STT-3%20Providers-4CAF50?style=flat-square" alt="STT">
  <img src="https://img.shields.io/badge/TTS-3%20Providers-2196F3?style=flat-square" alt="TTS">
  <img src="https://img.shields.io/badge/LLM-3%20Options-9C27B0?style=flat-square" alt="LLM">
</p>

---

## Overview

A sophisticated **AI-powered voice agent** designed to help Kotak Mahindra Bank acquire gold loan customers from competitors (Muthoot Finance, Manappuram, IIFL). The system provides personalized, empathetic conversations in **7 Indian languages** to understand customer needs and present compelling offers.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Multilingual** | Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali + English |
| **Personalized** | 4 customer segments with tailored messaging |
| **Intelligent** | RAG-powered knowledge base with competitor insights |
| **Real-time** | WebSocket-based voice streaming with low latency |
| **Extensible** | Plugin architecture for easy provider swapping |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai) (for local LLM)
- 16GB+ RAM recommended

### 1. Clone & Setup

```bash
git clone --recurse-submodules <repo-url>
cd goldloan-study
```

### 2. Backend Setup

```bash
cd voice-agent/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Pull Ollama model
ollama pull qwen3:8b-q4_K_M

# Start backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
cd voice-agent/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Access Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## Supported Languages

<table>
<tr>
<td align="center"><h3>हिंदी</h3><sub>Hindi</sub></td>
<td align="center"><h3>தமிழ்</h3><sub>Tamil</sub></td>
<td align="center"><h3>తెలుగు</h3><sub>Telugu</sub></td>
<td align="center"><h3>ಕನ್ನಡ</h3><sub>Kannada</sub></td>
</tr>
<tr>
<td align="center"><h3>മലയാളം</h3><sub>Malayalam</sub></td>
<td align="center"><h3>বাংলা</h3><sub>Bengali</sub></td>
<td align="center"><h3>English</h3><sub>English</sub></td>
<td align="center"><h3>🇮🇳</h3><sub>+15 more via IndicConformer</sub></td>
</tr>
</table>

---

## Customer Segments

The agent personalizes conversations based on **4 customer segments**:

| Segment | Profile | Loan Range | Key Messaging |
|---------|---------|------------|---------------|
| **P1: High-Value** | MSME owners, business expansion | ₹5-25 Lakhs | Speed, high LTV, business growth |
| **P2: Trust-Seeker** | Safety-focused, 40-55 years | ₹1-5 Lakhs | Security, transparent terms, trust |
| **P3: Shakti** | Women entrepreneurs | ₹50K-3 Lakhs | Empowerment, special rates, respect |
| **P4: Young Pro** | Digital-native, 21-35 years | ₹50K-2 Lakhs | App-first, quick process, flexibility |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VOICE AGENT SYSTEM                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐         ┌──────────────────────────────────────┐ │
│   │   FRONTEND  │ WebSocket│           BACKEND (FastAPI)          │ │
│   │   (React)   │◄────────►│                                      │ │
│   │             │         │  ┌────────────────────────────────┐  │ │
│   │ • Mic Input │         │  │        VOICE PIPELINE          │  │ │
│   │ • Playback  │         │  │                                │  │ │
│   │ • Transcript│         │  │   Audio → STT → LLM → TTS     │  │ │
│   │ • Customer  │         │  │                                │  │ │
│   │   Selector  │         │  └────────────────────────────────┘  │ │
│   └─────────────┘         │                                      │ │
│                           │  ┌────────────────────────────────┐  │ │
│                           │  │      PLUGIN ARCHITECTURE       │  │ │
│                           │  │                                │  │ │
│                           │  │  ┌──────┐ ┌──────┐ ┌────────┐ │  │ │
│                           │  │  │ STT  │ │ TTS  │ │  LLM   │ │  │ │
│                           │  │  └──┬───┘ └──┬───┘ └───┬────┘ │  │ │
│                           │  │     │        │         │      │  │ │
│                           │  │  ┌──▼────────▼─────────▼───┐  │  │ │
│                           │  │  │    Plugin Registry      │  │  │ │
│                           │  │  └─────────────────────────┘  │  │ │
│                           │  └────────────────────────────────┘  │ │
│                           │                                      │ │
│                           │  ┌────────────────────────────────┐  │ │
│                           │  │       KNOWLEDGE (RAG)          │  │ │
│                           │  │                                │  │ │
│                           │  │  ChromaDB + BM25 Hybrid Search │  │ │
│                           │  │  • FAQs, Products, Competitors │  │ │
│                           │  │  • Regulations, Objections     │  │ │
│                           │  └────────────────────────────────┘  │ │
│                           └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Plugin System

The agent uses a **pluggable architecture** allowing easy swapping of providers:

### Speech-to-Text (STT)

| Provider | Languages | Use Case |
|----------|-----------|----------|
| **IndicConformer** | 22 Indian languages | Production (best accuracy) |
| **Whisper** | Multilingual | Fallback |
| **Sarvam AI** | Indian languages | API-based alternative |

### Text-to-Speech (TTS)

| Provider | Languages | Quality |
|----------|-----------|---------|
| **IndicF5** | 11 Indian languages | Natural, expressive |
| **Piper** | Multiple | Fast, lightweight |
| **Parler TTS** | English | High quality |

### Language Models (LLM)

| Provider | Model | Deployment |
|----------|-------|------------|
| **Ollama** | qwen3:8b | Local (default) |
| **Claude** | claude-3.5-haiku | API fallback |
| **OpenAI** | gpt-4 | API option |

---

## Configuration

### Environment Variables

```bash
# Required
SARVAM_API_KEY=your_key          # For Sarvam AI speech services
ANTHROPIC_API_KEY=your_key       # For Claude LLM fallback

# Optional overrides
DEFAULT_LANGUAGE=hi              # Default conversation language
STT_PROVIDER=indicconformer      # Speech-to-text provider
TTS_PROVIDER=indicf5             # Text-to-speech provider
LLM_PROVIDER=ollama              # Language model provider
```

### Feature Flags (`config/features.yaml`)

```yaml
stt:
  provider: indicconformer       # Primary STT
  fallback: whisper              # Fallback STT

tts:
  provider: indicf5              # Primary TTS
  fallback: piper                # Fallback TTS

llm:
  provider: ollama
  model: qwen3:8b-q4_K_M
  temperature: 0.3
  max_tokens: 150

rag:
  enabled: true
  similarity_threshold: 0.7
  hybrid_weight: 0.6             # Semantic vs BM25 balance

experiments:
  mode: native                   # native | translation | ab_test
```

---

## Project Structure

```
goldloan-study/
├── README.md                    # This file
├── ARCHITECTURE.md              # Detailed architecture docs
├── CLAUDE.md                    # Project context
├── .gitignore                   # Git ignore rules
├── .gitmodules                  # Submodule configuration
│
├── voice-agent/                 # Main application
│   ├── backend/                 # Python FastAPI backend
│   │   ├── main.py              # Application entry point
│   │   ├── config/              # Configuration management
│   │   ├── core/                # Plugin system & pipeline
│   │   ├── plugins/             # STT, TTS, LLM, Translation providers
│   │   ├── conversation/        # State machine & flow management
│   │   ├── personalization/     # Customer segmentation
│   │   ├── rag/                 # Knowledge retrieval (RAG)
│   │   ├── tools/               # Function calling tools
│   │   ├── data/knowledge/      # YAML knowledge base
│   │   └── ai4bharat-nemo/      # Indian language models (submodule)
│   │
│   └── frontend/                # React TypeScript frontend
│       ├── src/components/      # UI components
│       └── src/hooks/           # React hooks (WebSocket voice agent)
│
└── research/                    # Research & documentation
    ├── research_docs/           # Strategy reports & analysis
    ├── presentation-content/    # Presentation materials
    └── kotak-gold-loan-presentation/  # React presentation app
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws/conversation/{id}` | WebSocket | Real-time voice conversation |
| `/api/customers` | GET | List customer profiles |
| `/api/languages` | GET | List supported languages |
| `/api/conversations/start` | POST | Start new conversation |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |

---

## Extending the System

### Adding a New Language

1. Update `config/languages.py`:
```python
LANGUAGE_CONFIG = {
    "new_lang": {
        "name": "Language Name",
        "native_name": "नेटिव नाम",
        "tts_voice": "voice_id",
        "greeting": "नमस्ते..."
    }
}
```

2. Add prompts in `conversation/prompts/new_lang/`
3. Update frontend language selector

### Adding a New Speech Provider

1. Create provider in `plugins/stt/` or `plugins/tts/`:
```python
from core.interfaces import STTPlugin

class NewSTTProvider(STTPlugin):
    async def transcribe(self, audio: bytes, language: str) -> str:
        # Implementation
        pass
```

2. Register in `plugins/__init__.py`
3. Add to `config/features.yaml`

### Adding New Knowledge

Add YAML files to `data/knowledge/`:
```yaml
# data/knowledge/new_topic.yaml
- question: "What is...?"
  answer: "The answer is..."
  category: "general"
  keywords: ["key1", "key2"]
```

---

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| **End-to-end latency** | < 2s | ~1.5s |
| **STT accuracy (Hindi)** | > 90% | 94% |
| **Concurrent conversations** | 10 | 10 |
| **Model memory** | < 8GB | ~6GB |

---

## Research & Documentation

The `research/` directory contains comprehensive analysis:

- **Market Analysis**: India gold loan market (₹7.3L crore opportunity)
- **Competitor Study**: Muthoot, Manappuram, IIFL positioning
- **Regulatory Framework**: RBI guidelines compliance
- **Customer Psychology**: Trust factors & decision drivers
- **AI Strategy**: Voice AI differentiation

---

## License

Proprietary - Kotak Mahindra Bank

---

<p align="center">
  <sub>Built with AI for India's gold loan market</sub>
</p>
