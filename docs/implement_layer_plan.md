# Whisper `large-v3` Implementation Layer Plan (Milestone 1)

## Goal
Build a CUDA-only, end-to-end Whisper baseline from scratch for long audio, using primitive layers first and composing upward.

## Baseline Scope
- Model: `whisper-large-v3`
- Language: fixed Japanese (`ja`), no language detection
- Runtime: CUDA only
- Pipeline order: encode all chunks first, then decode
- No optimization work in this milestone beyond correctness and stable execution

## Fixed Constants (load once)
1. Model dimensions:
   - `d_model=1280`
   - `n_heads=20`
   - `d_head=64`
   - `encoder_layers=32`
   - `decoder_layers=32`
   - `vocab_size=51866`
2. Audio/frontend:
   - `sample_rate=16000`
   - `n_fft=400`
   - `hop_length=160`
   - `chunk_length=30s`
   - `mel_bins=128`
   - `nb_max_frames=3000`
3. Context:
   - `encoder_ctx=1500`
   - `decoder_ctx=448`

## Layer Build Order (primitive to full model)

### 1) Asset + Config Layer
1. Parse and validate:
   - `config.json`
   - `generation_config.json`
   - `preprocessor_config.json`
   - packed weight `manifest.json` + `weights.bin`
2. Resolve token IDs used by fixed JA decode:
   - `<|ja|>`, `<|transcribe|>`, `bos/eos`, `no_timestamps`

### 2) Memory Pool Layer (add early)
1. Implement host and device allocators:
   - Host pinned pool for H2D/D2H staging
   - Device pool for runtime tensors
2. Implement allocation modes:
   - Static region: weights, positional embeddings, constant masks
   - Reusable workspace: per-step temporaries (`QK`, softmax scratch, logits scratch)
   - Sequence state region: decoder KV cache
3. Allocator behavior:
   - Alignment: at least 64B (128B preferred for tensor cores)
   - Deterministic free-list ordering
   - Stream-safe lifetime tracking (free only after stream completion event)
4. Baseline API:
   - `alloc(size, alignment, stream, tag)`
   - `free(handle)`
   - `reset_workspace(epoch_id)` for per-step reuse
   - `stats()` for used/peak bytes by tag
5. Initial sizing checkpoints:
   - Compute expected bytes for:
     - all model weights on device
     - max encoder activation/workspace
     - KV cache for `decoder_layers * heads * d_head * max_tokens`
   - Fail fast if pool capacity is insufficient

### 3) Tensor + Math Primitive Layer
1. Tensor object: dtype, shape, stride, contiguous view, device pointer
2. Core ops:
   - `matmul` (`bf16` input with `fp32` accumulation)
   - `conv1d`
   - elementwise add/mul/exp/log/max
   - reduce sum/max
   - masked `softmax` (stable, `fp32`)
   - `layer_norm` (`fp32` stats/reduction)
   - `gelu`
   - `embedding_gather`
   - `argmax` / `topk`
3. KV cache primitives:
   - append/write K,V by layer and timestep
   - gather/read history range for attention

### 4) Audio Frontend Layer
1. Audio decode and mono conversion
2. Resample to 16kHz
3. STFT
4. Power spectrogram
5. Mel projection (`128` bins)
6. Log-mel normalization and clamp
7. Pad/trim to chunk shape (`30s`, `3000` frames)

### 5) Encoder Layer Stack
1. Input stem:
   - `conv1 -> gelu -> conv2 -> gelu`
   - add encoder positional embedding
2. 32x encoder blocks:
   - `LN -> self-attn -> residual`
   - `LN -> MLP(fc1 -> gelu -> fc2) -> residual`
3. Final encoder layer norm

### 6) Decoder Layer Stack
1. Decoder input:
   - token embedding + positional embedding
   - causal mask
2. 32x decoder blocks:
   - `LN -> causal self-attn (with KV cache) -> residual`
   - `LN -> cross-attn(encoder states) -> residual`
   - `LN -> MLP(fc1 -> gelu -> fc2) -> residual`
3. Final decoder layer norm
4. Output logits projection (tied to token embedding matrix)

### 7) Tokenizer + Text Layer
1. Implement/port Whisper normalizer behavior
2. Implement BPE encode/decode using model assets
3. Handle special tokens and suppression lists

### 8) Decode Control Layer
1. Build fixed JA prompt prefix:
   - `<|startoftranscript|> <|ja|> <|transcribe|> ...`
2. Autoregressive loop:
   - forward one token step
   - apply suppression/timestamp rules
   - pick next token (greedy or fixed beam size)
3. Stop conditions:
   - EOS, max length, safety limits

### 9) End-to-End Pipeline Layer
1. Long-audio chunking
2. Encode-all stage for all chunks
3. Decode-all stage for all chunks
4. Stitch text output

## Implementation Sequence (recommended)
1. Layer 1 + Layer 2
2. Layer 3 kernels with unit tests
3. Layer 4 audio frontend parity checks
4. Layer 5 encoder forward parity checks
5. Layer 6 decoder one-step parity checks
6. Layer 7 tokenizer round-trip tests
7. Layer 8 decode loop correctness tests
8. Layer 9 full transcription smoke test

## Milestone 1 Exit Criteria
1. Runs end-to-end on CUDA with `large-v3` weights
2. Produces Japanese transcription from long audio input
3. Uses fixed-language JA decode path (no language detection)
4. Memory pool reports stable peak usage and no leaks
