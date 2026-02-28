# C++/CUDA Whisper Inference Requirements

Having model weights is necessary but not sufficient. A Whisper runtime also needs:

1. Tokenizer assets:
   - `vocab.json`
   - `merges.txt`
   - `tokenizer.json` (or equivalent BPE implementation)
   - `normalizer.json`
2. Model architecture/config:
   - `config.json`
   - Dimensions (`d_model`, heads, layers, vocab size, max positions)
   - Special token IDs (BOS/EOS, language/task tokens, timestamp controls)
3. Audio frontend parameters:
   - log-Mel settings (`n_mels=80`, FFT/hop/window, sample rate 16 kHz)
   - chunking/stride policy for long audio
4. Decode policy:
   - fixed beam size per run (reproducibility)
   - temperature / fallback / repetition controls
   - timestamp rules and suppression tables
5. Runtime behavior:
   - KV cache layout and paging policy
   - precision rules (BF16 storage, FP32 accumulation points)

## Converter Added In This Repo

Subproject: `tools/weights_converter`

It converts Whisper `.safetensors` (single file or sharded index) into:

- `weights.bin`: contiguous tensor payloads
- `manifest.json`: tensor metadata (`name`, `dtype`, `shape`, `offset`, `nbytes`)

This format is intentionally simple for C++ mmap/read + CUDA upload logic.

## Example

```bash
cd tools/weights_converter
uv run weights-converter \
  --input ../../third_party/whisper-large-v3 \
  --output-dir ../../artifacts/whisper-large-v3-packed
```

Smoke test with a small subset:

```bash
cd tools/weights_converter
uv run weights-converter \
  --input ../../third_party/whisper-large-v3 \
  --output-dir /tmp/packed-test \
  --max-tensors 16
```

## Manifest Schema (v1)

```json
{
  "format": "longwhisper.packed_weights.v1",
  "weights_file": "weights.bin",
  "alignment": 64,
  "num_tensors": 0,
  "tensors": [
    {
      "name": "encoder.conv1.weight",
      "dtype": "float16",
      "shape": [1280, 80, 3],
      "offset": 0,
      "nbytes": 614400,
      "source_file": "model.safetensors"
    }
  ]
}
```
