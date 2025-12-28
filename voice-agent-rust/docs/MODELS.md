# Voice Agent ONNX Models

This document describes the ONNX models required for the Voice Agent and how to obtain/export them.

## Overview

| Component | Model | Purpose | Size | Latency Target |
|-----------|-------|---------|------|----------------|
| VAD | Silero VAD | Voice activity detection | 1.5MB | <5ms |
| STT | IndicConformer | Hindi/English speech-to-text | ~100MB | <100ms |
| TTS | Piper Hindi | Hindi text-to-speech | ~50MB | <100ms first chunk |
| Reranker | MiniLM Cross-encoder | Document reranking | ~25MB | <20ms |
| Embedding | e5-small | Dense embeddings for RAG | ~100MB | <50ms |

## Quick Start

```bash
# Download models (creates placeholders for manual models)
./scripts/download_models.sh

# Set models path
export MODELS_PATH=./models
```

## Model Details

### 1. Silero VAD (Voice Activity Detection)

**Status**: Auto-downloaded

The Silero VAD model detects speech segments in audio. It's lightweight and fast.

```bash
# Already downloaded by script
ls models/vad/silero_vad.onnx
```

**Input**: 16kHz mono audio (512 samples = 32ms chunks)
**Output**: Speech probability (0.0 - 1.0)

### 2. IndicConformer (Speech-to-Text)

**Status**: Manual download required

IndicConformer is the recommended STT model for Hindi. It provides excellent accuracy on Indian languages.

#### Option A: AI4Bharat Model (Recommended)

1. Visit [AI4Bharat IndicConformer](https://github.com/AI4Bharat/IndicConformer)
2. Download the pre-trained model
3. Export to ONNX:

```python
from indicconformer import IndicConformerASR
import torch

model = IndicConformerASR.from_pretrained("ai4bharat/indicconformer-hi")

# Export encoder
dummy_audio = torch.randn(1, 16000)  # 1 second of audio
torch.onnx.export(
    model.encoder,
    dummy_audio,
    "models/stt/indicconformer_encoder.onnx",
    input_names=["audio"],
    output_names=["features"],
    dynamic_axes={"audio": {1: "length"}}
)
```

#### Option B: Whisper (Alternative)

```bash
pip install whisper-onnx
python -m whisper_onnx.export --model small --output models/stt/whisper-small
```

#### Option C: Wav2Vec2 Hindi

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx \
    --model facebook/wav2vec2-large-xlsr-53-hindi \
    --task automatic-speech-recognition \
    models/stt/wav2vec2-hindi
```

### 3. Piper TTS (Text-to-Speech)

**Status**: Partially auto-downloaded

Piper provides fast, high-quality neural TTS with Hindi support.

```bash
# Hindi voice (auto-downloaded)
ls models/tts/hi_IN-swara-medium.onnx

# Alternative: Download other voices
# See https://github.com/rhasspy/piper for voice list
```

**Input**: Phoneme sequence (IPA)
**Output**: 22050Hz audio samples

#### G2P (Grapheme-to-Phoneme)

The TTS requires phoneme input. The G2P module converts Hindi text to IPA phonemes.

See `crates/pipeline/src/tts/g2p.rs` for the implementation.

### 4. Cross-Encoder Reranker

**Status**: Manual download required

The reranker uses a cross-encoder model to score query-document relevance.

#### Standard Export

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx \
    --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    models/reranker/minilm
```

#### With Early Exit Support

For true layer-by-layer early exit (not currently used, cascaded reranking is faster):

```python
from transformers import AutoModel
import torch.onnx

model = AutoModel.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    output_hidden_states=True
)

# Custom export with all hidden states
# See scripts/export_reranker_with_layers.py
```

### 5. Embedding Model (RAG)

**Status**: Manual download required

Dense embeddings for semantic search.

```bash
pip install optimum[onnxruntime]
optimum-cli export onnx \
    --model intfloat/e5-small-v2 \
    models/embedding/e5-small
```

For multilingual (Hindi + English):

```bash
optimum-cli export onnx \
    --model intfloat/multilingual-e5-small \
    models/embedding/me5-small
```

## Configuration

Set the model paths in your configuration:

```toml
[models]
vad_path = "models/vad/silero_vad.onnx"
stt_path = "models/stt/indicconformer"
tts_path = "models/tts/hi_IN-swara-medium.onnx"
reranker_path = "models/reranker/minilm"
embedding_path = "models/embedding/e5-small"
```

Or via environment variables:

```bash
export VAD_MODEL_PATH=models/vad/silero_vad.onnx
export STT_MODEL_PATH=models/stt/indicconformer
export TTS_MODEL_PATH=models/tts/hi_IN-swara-medium.onnx
export RERANKER_MODEL_PATH=models/reranker/minilm
export EMBEDDING_MODEL_PATH=models/embedding/e5-small
```

## Model Requirements

| Model | ONNX Runtime Version | Execution Provider | Notes |
|-------|---------------------|-------------------|-------|
| All | 2.0.0-rc.9+ | CPU | Default, always works |
| All | 2.0.0-rc.9+ | CUDA | Requires CUDA 11.x |
| All | 2.0.0-rc.9+ | TensorRT | Best GPU performance |

## Latency Optimization

1. **Use INT8 Quantization** for faster inference:
   ```bash
   python -m onnxruntime.quantization.quantize \
       --model model.onnx \
       --output model_int8.onnx \
       --quant_format QDQ
   ```

2. **Use ONNX Runtime Graph Optimization**:
   - Set `GraphOptimizationLevel::Level3` in code

3. **Use IOBinding** for zero-copy GPU inference

4. **Batch Processing** for embedding/reranking

## Troubleshooting

### Model Loading Fails

```
Error: Failed to load model: Invalid model
```

- Verify ONNX file is not corrupted
- Check ONNX opset version compatibility
- Try reconverting with latest optimum

### Out of Memory

- Use INT8 quantized models
- Reduce batch size
- Enable memory arena shrinking

### Slow Performance

- Enable GPU execution provider
- Check model is actually using ONNX Runtime (not Python fallback)
- Profile with ONNX Runtime profiler
