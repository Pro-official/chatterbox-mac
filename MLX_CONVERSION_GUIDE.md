# Chatterbox TTS → MLX Conversion Guide

## Overview

Converting Chatterbox TTS from PyTorch to MLX for native Apple Silicon performance.

**Goal**: Achieve ~0.5-1x RTF (real-time or faster) vs current ~3x RTF on MPS.

**Source**: `/Users/promise/dev/truenaad-ai/chatterbox-tts/src/chatterbox/`

**Target**: `/Users/promise/dev/truenaad-ai/chatterbox-mac/mlx_chatterbox/`

---

## Architecture Overview

```
Text Input
    ↓
┌─────────────────────────────────────┐
│  T3 Model (Text → Speech Tokens)    │  ← Llama/GPT2 backbone
│  - Text Embedding                    │
│  - Speech Embedding                  │
│  - Transformer (30 layers)           │
│  - Conditioning (VoiceEncoder)       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  S3Gen (Tokens → Mel Spectrogram)   │  ← Conformer + Flow Matching
│  - S3Tokenizer                       │
│  - UpsampleConformerEncoder          │
│  - ConditionalDecoder                │
│  - CAMPPlus (speaker encoder)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  HiFiGAN (Mel → Waveform)           │  ← Neural Vocoder
│  - Upsampling convolutions           │
│  - Residual blocks                   │
│  - Snake activation                  │
└─────────────────────────────────────┘
    ↓
Audio Output (24kHz)
```

---

## Conversion Phases

### Phase 1: VoiceEncoder (LSTM) ✅ Start Here
**Source**: `models/voice_encoder/voice_encoder.py`
**Complexity**: Easy
**Time**: 2-3 hours

**Components**:
- `nn.LSTM` (3 layers, bidirectional=False)
- `nn.Linear` projection
- Mel spectrogram extraction

**MLX Equivalents**:
| PyTorch | MLX |
|---------|-----|
| `nn.LSTM` | `nn.LSTM` (exists in MLX) |
| `nn.Linear` | `nn.Linear` |
| `torch.Tensor` | `mx.array` |

**Weight Conversion**:
```python
# PyTorch LSTM weights
weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0

# MLX LSTM weights
Wx, Wh, bias (combined)
```

---

### Phase 2: T3 Model (Transformer)
**Source**: `models/t3/t3.py`
**Complexity**: Medium
**Time**: 1-2 days

**Components**:
- `nn.Embedding` (text + speech)
- `LlamaModel` or `GPT2Model` backbone
- `nn.Linear` heads
- RoPE (Rotary Position Embeddings)
- KV-Cache for generation

**MLX Equivalents**:
| PyTorch | MLX |
|---------|-----|
| `nn.Embedding` | `nn.Embedding` |
| `LlamaModel` | Use `mlx-lm` patterns |
| `nn.Linear` | `nn.Linear` |
| `F.scaled_dot_product_attention` | `mx.fast.scaled_dot_product_attention` |

**Key Files to Reference**:
- `mlx-lm` library for Llama implementation
- `/Users/promise/dev/truenaad-ai/mlx-audio/mlx_audio/tts/models/` for patterns

---

### Phase 3: S3Gen (Flow Matching Decoder)
**Source**: `models/s3gen/s3gen.py`, `flow.py`
**Complexity**: Medium-Hard
**Time**: 2-3 days

**Components**:
- `S3Tokenizer` (external lib - challenging)
- `UpsampleConformerEncoder`
- `CausalMaskedDiffWithXvec`
- `CAMPPlus` speaker encoder

**MLX Equivalents**:
| PyTorch | MLX |
|---------|-----|
| `Conformer` | Need to implement |
| `Conv1d` | `nn.Conv1d` |
| `MultiHeadAttention` | `nn.MultiHeadAttention` |
| ODE solver | Implement in MLX |

**S3Tokenizer Challenge**:
- External library with potential CUDA dependencies
- Options:
  1. Check if pure Python version exists
  2. Keep as PyTorch, convert tensors at boundary
  3. Reimplement in MLX

---

### Phase 4: HiFiGAN Vocoder
**Source**: `models/s3gen/hifigan.py`
**Complexity**: Medium
**Time**: 1 day

**Components**:
- `ConvTranspose1d` for upsampling
- Residual blocks with dilations
- Snake activation (custom)
- F0 predictor

**MLX Equivalents**:
| PyTorch | MLX |
|---------|-----|
| `nn.Conv1d` | `nn.Conv1d` |
| `nn.ConvTranspose1d` | `nn.ConvTranspose1d` |
| `weight_norm` | Implement manually |
| Snake activation | Implement: `x + (1/a) * sin(a*x)^2` |

---

## Weight Conversion Strategy

### General Pattern
```python
import torch
import mlx.core as mx
from safetensors.torch import load_file

def convert_pytorch_to_mlx(pt_weights_path):
    """Convert PyTorch weights to MLX format."""
    pt_state = load_file(pt_weights_path)

    mlx_state = {}
    for key, tensor in pt_state.items():
        # Convert to numpy then to MLX
        np_array = tensor.cpu().numpy()
        mlx_state[key] = mx.array(np_array)

    return mlx_state
```

### LSTM Weight Conversion
```python
def convert_lstm_weights(pt_lstm_state):
    """Convert PyTorch LSTM weights to MLX format."""
    # PyTorch: weight_ih_l{layer}, weight_hh_l{layer}, bias_ih_l{layer}, bias_hh_l{layer}
    # MLX: Wx, Wh, bias (per layer)

    mlx_weights = {}
    for layer in range(num_layers):
        # Input-hidden weights
        Wx = pt_lstm_state[f'weight_ih_l{layer}'].numpy()
        # Hidden-hidden weights
        Wh = pt_lstm_state[f'weight_hh_l{layer}'].numpy()
        # Biases (combine ih and hh)
        bias = (pt_lstm_state[f'bias_ih_l{layer}'] +
                pt_lstm_state[f'bias_hh_l{layer}']).numpy()

        mlx_weights[f'layers.{layer}.Wx'] = mx.array(Wx)
        mlx_weights[f'layers.{layer}.Wh'] = mx.array(Wh)
        mlx_weights[f'layers.{layer}.bias'] = mx.array(bias)

    return mlx_weights
```

---

## Directory Structure

```
mlx_chatterbox/
├── __init__.py
├── voice_encoder.py      # Phase 1
├── t3/
│   ├── __init__.py
│   ├── t3.py             # Phase 2
│   ├── config.py
│   └── modules/
│       ├── cond_enc.py
│       └── perceiver.py
├── s3gen/
│   ├── __init__.py
│   ├── s3gen.py          # Phase 3
│   ├── flow.py
│   ├── conformer.py
│   └── hifigan.py        # Phase 4
├── tokenizer.py
├── convert_weights.py    # Weight conversion utility
└── tts.py                # Main TTS interface
```

---

## Testing Strategy

### Unit Tests (per component)
```python
def test_voice_encoder():
    """Compare MLX vs PyTorch output."""
    # Load same weights in both
    pt_model = load_pytorch_voice_encoder()
    mlx_model = load_mlx_voice_encoder()

    # Same input
    audio = load_test_audio()

    # Compare outputs
    pt_out = pt_model(audio)
    mlx_out = mlx_model(audio)

    assert np.allclose(pt_out.numpy(), np.array(mlx_out), rtol=1e-4)
```

### Integration Tests
1. Generate audio with PyTorch version
2. Generate audio with MLX version (same text, same seed)
3. Compare waveforms and spectrograms

---

## Progress Tracking

### Phase 1: VoiceEncoder ✅ COMPLETE
- [x] Create `mlx_chatterbox/` directory structure
- [x] Implement `voice_encoder.py` in MLX (stacked LSTM layers)
- [x] Write weight conversion for LSTM
- [x] Test against PyTorch version
- [x] Validate speaker embeddings match (cosine similarity = 1.0)

### Phase 2: T3 Model ✅ COMPLETE
- [x] Implement T3Config dataclass
- [x] Port LlamaModel using mlx-lm
- [x] Implement text/speech embeddings
- [x] Implement conditioning encoder (Perceiver, T3CondEnc)
- [x] Implement KV-cache for generation
- [x] Test token generation

### Phase 3: S3Gen ⚠️ HYBRID APPROACH
- [x] Investigated - Too complex for full MLX conversion
- [x] Using PyTorch/MPS for S3Gen (efficient on Apple Silicon)
- [x] Conformer, Flow matching, CAMPPlus stay in PyTorch
- [x] HiFiGAN stays in PyTorch

### Phase 4: Integration ✅ COMPLETE
- [x] Create unified `ChatterboxMLX` interface
- [x] End-to-end testing working
- [x] Hybrid approach: MLX VoiceEncoder + T3, PyTorch S3Gen + HiFiGAN
- [x] Performance: ~4x RTF (improved from ~10x with MPS-only)

## Current Status

**ChatterboxMLX is functional!**

```python
from mlx_chatterbox import ChatterboxMLX

model = ChatterboxMLX.from_pretrained(turbo=False)
audio = model.generate(
    text="Hello world!",
    audio_prompt="reference.wav"
)
```

### Architecture (Hybrid)
```
Text Input
    ↓
┌─────────────────────────────────────┐
│  MLX VoiceEncoder (speaker emb)     │  ← MLX (~10x faster)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  MLX T3 Model (text → tokens)       │  ← MLX (~5x faster)
│  - Llama backbone via mlx-lm        │
│  - Custom embeddings & conditioning │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  PyTorch S3Gen (tokens → mel)       │  ← PyTorch/MPS
│  - Conformer encoder                │
│  - Flow matching decoder            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  PyTorch HiFiGAN (mel → wav)        │  ← PyTorch/MPS
└─────────────────────────────────────┘
    ↓
Audio Output (24kHz)
```

---

## References

### MLX Documentation
- https://ml-explore.github.io/mlx/build/html/index.html
- https://github.com/ml-explore/mlx-examples

### Existing MLX TTS (for patterns)
- `/Users/promise/dev/truenaad-ai/mlx-audio/mlx_audio/tts/models/kokoro/`
- `/Users/promise/dev/truenaad-ai/f5-tts-mlx/f5_tts_mlx/`

### Chatterbox Source
- `/Users/promise/dev/truenaad-ai/chatterbox-tts/src/chatterbox/`

---

## Notes

### Known Challenges
1. **S3Tokenizer**: External lib, may have CUDA deps
2. **Perth Watermarking**: Skip for now, use DummyWatermarker
3. **Memory**: Keep models lazy-loaded to manage memory
4. **Precision**: MLX defaults to float32, match PyTorch behavior

### Performance Targets
| Metric | Current (MPS) | Target (MLX) |
|--------|---------------|--------------|
| RTF (Turbo) | ~3x | ~0.5-1x |
| RTF (Standard) | ~10x | ~2-3x |
| Memory | ~4GB | ~3GB |
| Load time | ~15s | ~10s |
